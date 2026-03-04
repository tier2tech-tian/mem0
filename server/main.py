import hashlib
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.responses import Response

from mem0 import Memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "")
QDRANT_PORT = os.environ.get("QDRANT_PORT", "")
QDRANT_PATH = os.environ.get("QDRANT_PATH", "")  # local file-based Qdrant (no Docker needed)
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "memories")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIMS = os.environ.get("EMBEDDING_DIMS", "1024")
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "ollama")  # "ollama" or "openai"
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")  # "ollama" or "openai"
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "history/history.db")
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "")
_admin_session_token = secrets.token_urlsafe(32)  # regenerated on restart

# --- API Key Management ---
API_KEYS_FILE = Path(os.environ.get("API_KEYS_FILE", Path(__file__).parent / "api_keys.json"))
_api_keys_cache: Dict[str, Any] = {}
_api_keys_mtime: float = 0.0


def _load_api_keys() -> Dict[str, Any]:
    """Load api_keys.json, with mtime-based hot reload."""
    global _api_keys_cache, _api_keys_mtime
    try:
        mtime = API_KEYS_FILE.stat().st_mtime
    except FileNotFoundError:
        _api_keys_cache = {}
        _api_keys_mtime = 0.0
        return _api_keys_cache
    if mtime != _api_keys_mtime:
        with open(API_KEYS_FILE, "r") as f:
            data = json.load(f)
        _api_keys_cache = data.get("keys", {})
        _api_keys_mtime = mtime
        logging.info("Reloaded API keys (%d keys)", len(_api_keys_cache))
    return _api_keys_cache


def _save_api_keys(keys: Dict[str, Any]) -> None:
    global _api_keys_cache, _api_keys_mtime
    with open(API_KEYS_FILE, "w") as f:
        json.dump({"keys": keys}, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _api_keys_cache = keys
    _api_keys_mtime = API_KEYS_FILE.stat().st_mtime


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


from datetime import datetime

CUSTOM_FACT_EXTRACTION_PROMPT = f"""You are an Information Organizer, specialized in accurately storing facts, memories, and knowledge from conversations.
Your primary role is to extract ALL relevant pieces of information and organize them into distinct, manageable facts.

Types of Information to Remember:
1. Personal Preferences and Details
2. Technical Architecture and System Design decisions
3. Configuration, Deployment, and Infrastructure details
4. Code patterns, Libraries, Tools and their usage
5. Bug reports, Known issues and Solutions
6. Project structure and Module responsibilities
7. Plans, Decisions, and Workflows
8. Any other factual information worth remembering

Here are some few shot examples:

Input: Hi.
Output: {{"facts" : []}}

Input: There are branches in trees.
Output: {{"facts" : []}}

Input: 后端用 FastAPI，数据库用 PostgreSQL，部署在 AWS。
Output: {{"facts" : ["后端用 FastAPI", "数据库用 PostgreSQL", "部署在 AWS"]}}

Input: The auth middleware uses JWT with RS256. Token expiry is 24 hours.
Output: {{"facts" : ["Auth middleware uses JWT with RS256", "Token expiry is 24 hours"]}}

Return the facts in a json format as shown above.

Remember:
- Today's date is {datetime.now().strftime("%Y-%m-%d")}.
- Do not return anything from the custom few shot example prompts provided above.
- If you do not find anything relevant, return an empty list for "facts".
- Create the facts based on the user messages only.
- Return response in json with key "facts" and value as list of strings.
- Detect the language of input and record facts in the same language.
- Extract ALL factual information, including technical details, architecture, configurations, and system design.
"""

DEFAULT_CONFIG = {
    "version": "v1.1",
    "custom_fact_extraction_prompt": CUSTOM_FACT_EXTRACTION_PROMPT,
    "vector_store": {
        "provider": "qdrant",
        "config": {
            **( {"path": QDRANT_PATH, "on_disk": True} if QDRANT_PATH
                else {"host": QDRANT_HOST, "port": int(QDRANT_PORT or 6333)} ),
            "collection_name": COLLECTION_NAME,
            "embedding_model_dims": int(EMBEDDING_DIMS),
        },
    },
    "llm": {
        "provider": LLM_PROVIDER,
        "config": (
            {
                "model": LLM_MODEL,
                "temperature": 0,
                "max_tokens": 2000,
                "api_key": LLM_API_KEY,
                "openai_base_url": LLM_BASE_URL,
            }
            if LLM_PROVIDER == "openai"
            else {
                "model": LLM_MODEL,
                "temperature": 0,
                "max_tokens": 2000,
                "ollama_base_url": OLLAMA_BASE_URL,
            }
        ),
    },
    "embedder": {
        "provider": EMBEDDING_PROVIDER,
        "config": (
            {
                "model": EMBEDDING_MODEL,
                "embedding_dims": int(EMBEDDING_DIMS),
                "api_key": EMBEDDING_API_KEY or LLM_API_KEY,
                "openai_base_url": EMBEDDING_BASE_URL or LLM_BASE_URL,
            }
            if EMBEDDING_PROVIDER == "openai"
            else {
                "model": EMBEDDING_MODEL,
                "ollama_base_url": OLLAMA_BASE_URL,
            }
        ),
    },
    "history_db_path": HISTORY_DB_PATH,
}


# --- Monkey-patch: disable thinking mode for OpenAI-compatible LLMs (e.g. Qwen3.5) ---
if os.environ.get("OPENAI_DISABLE_THINKING", "").lower() in ("1", "true"):
    from mem0.llms.openai import OpenAILLM

    _orig_generate = OpenAILLM.generate_response

    def _patched_generate(self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
        _orig_create = self.client.chat.completions.create

        def _create_with_no_thinking(**kw):
            kw.setdefault("extra_body", {})["enable_thinking"] = False
            return _orig_create(**kw)

        self.client.chat.completions.create = _create_with_no_thinking
        try:
            return _orig_generate(self, messages, response_format, tools, tool_choice, **kwargs)
        finally:
            self.client.chat.completions.create = _orig_create

    OpenAILLM.generate_response = _patched_generate
    logging.info("Patched OpenAILLM: thinking mode disabled")

MEMORY_INSTANCE = Memory.from_config(DEFAULT_CONFIG)

# --- Concurrency limiter (500 = DashScope QPS upper bound) ---
_MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "500"))
_ollama_semaphore = threading.Semaphore(_MAX_CONCURRENT)
_queue_waiting = 0
_queue_processing = 0  # now a counter, not bool
_queue_lock = threading.Lock()


def _call_with_semaphore(fn, max_retries=2):
    global _queue_waiting, _queue_processing
    with _queue_lock:
        _queue_waiting += 1
    try:
        with _ollama_semaphore:
            with _queue_lock:
                _queue_waiting -= 1
                _queue_processing += 1
            try:
                for attempt in range(max_retries):
                    try:
                        return fn()
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logging.warning("Retry %d: %s", attempt + 1, e)
                            time.sleep(1)
                        else:
                            raise
            finally:
                with _queue_lock:
                    _queue_processing -= 1
    except:
        with _queue_lock:
            _queue_waiting = max(0, _queue_waiting - 1)
        raise

app = FastAPI(
    title="Mem0 REST APIs",
    description="A REST API for managing and searching memories for your AI Agents and Apps.",
    version="1.0.0",
)


# --- Authentication Middleware ---
_LOCAL_IPS = {"127.0.0.1", "::1"}
_PUBLIC_PATHS = {"/docs", "/openapi.json", "/"}


def _is_local(request: Request) -> bool:
    client = request.client
    return client is not None and client.host in _LOCAL_IPS


async def _extract_user_id(request: Request) -> Optional[str]:
    """Extract user_id from query params or JSON body."""
    # Try query params first
    user_id = request.query_params.get("user_id")
    if user_id:
        return user_id
    # Try JSON body (for POST/PUT/PATCH)
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            body = await request.body()
            if body:
                data = json.loads(body)
                return data.get("user_id")
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return None


@app.middleware("http")
async def auth_middleware(request: Request, call_next) -> Response:
    # 1. Local access → allow unconditionally
    if _is_local(request):
        return await call_next(request)

    # 2. Public paths (docs, openapi)
    if request.url.path in _PUBLIC_PATHS:
        return await call_next(request)

    # 3. Admin endpoints: require login session (cookie) or local access
    if request.url.path.startswith("/admin/"):
        # Allow login page and login POST without session
        if request.url.path in ("/admin/login", "/admin/login/"):
            return await call_next(request)
        # Check session cookie
        session = request.cookies.get("mem0_admin")
        if session == _admin_session_token and ADMIN_PASS:
            return await call_next(request)
        # Not authenticated → redirect to login
        if request.url.path.endswith(("/", "")):
            return RedirectResponse(url="/admin/login")
        return JSONResponse(status_code=401, content={"detail": "Admin login required"})

    # 4. Extract Bearer token
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"detail": "Missing or invalid Authorization header"})
    token = auth_header[7:]

    # 5. Look up key hash
    keys = _load_api_keys()
    key_hash = _hash_key(token)
    key_entry = keys.get(key_hash)
    if not key_entry:
        return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

    # 6. Check user_id binding
    bound_user_id = key_entry.get("user_id")
    if bound_user_id:
        # We need to read the body to check user_id, but the body stream is consumed.
        # Cache it so downstream handlers can still read it.
        body = await request.body()
        req_user_id = request.query_params.get("user_id")
        if not req_user_id and body:
            try:
                data = json.loads(body)
                req_user_id = data.get("user_id")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        if req_user_id and req_user_id != bound_user_id:
            return JSONResponse(
                status_code=403,
                content={"detail": f"API key is bound to user_id '{bound_user_id}', cannot access '{req_user_id}'"},
            )

    return await call_next(request)


# --- Admin Endpoints (session-authenticated, enforced by middleware) ---


class AdminLoginRequest(BaseModel):
    username: str
    password: str


ADMIN_LOGIN_HTML = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mem0 Admin Login</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#FAF9F7;color:#1a1a1a;
  min-height:100vh;display:flex;align-items:center;justify-content:center;
}
.login-card{
  background:#fff;border:1px solid #e8e6e3;border-radius:12px;
  padding:32px;width:340px;box-shadow:0 2px 8px rgba(0,0,0,.06);
}
.login-card h1{font-size:20px;font-weight:600;margin-bottom:4px;text-align:center}
.login-card .subtitle{color:#666;font-size:13px;text-align:center;margin-bottom:24px}
.field{margin-bottom:14px}
.field label{display:block;font-size:13px;font-weight:500;color:#555;margin-bottom:4px}
.field input{
  width:100%;padding:9px 12px;border:1px solid #ddd;border-radius:8px;
  font-size:14px;outline:none;transition:border .15s;background:#fafafa;
}
.field input:focus{border-color:#D97757;background:#fff}
.btn-login{
  width:100%;padding:10px;border:none;border-radius:8px;
  font-size:14px;font-weight:500;cursor:pointer;
  background:#D97757;color:#fff;transition:background .15s;
}
.btn-login:hover{background:#c4684a}
.btn-login:disabled{opacity:.5;cursor:not-allowed}
.error-msg{color:#c44;font-size:13px;text-align:center;margin-top:12px;display:none}
</style>
</head>
<body>
<div class="login-card">
  <h1>Mem0 Admin</h1>
  <p class="subtitle">请登录以管理 API Keys</p>
  <form id="login-form">
    <div class="field">
      <label>用户名</label>
      <input id="inp-user" type="text" autocomplete="username" autofocus>
    </div>
    <div class="field">
      <label>密码</label>
      <input id="inp-pass" type="password" autocomplete="current-password">
    </div>
    <button class="btn-login" type="submit">登录</button>
    <div class="error-msg" id="error-msg"></div>
  </form>
</div>
<script>
document.getElementById('login-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector('.btn-login');
  const err = document.getElementById('error-msg');
  btn.disabled = true;
  btn.textContent = '登录中…';
  err.style.display = 'none';
  try {
    const res = await fetch('/admin/login', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        username: document.getElementById('inp-user').value,
        password: document.getElementById('inp-pass').value
      })
    });
    if (res.ok) {
      window.location.href = '/admin/';
    } else {
      const data = await res.json();
      err.textContent = data.detail || '登录失败';
      err.style.display = 'block';
    }
  } catch(ex) {
    err.textContent = '请求失败';
    err.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = '登录';
  }
});
</script>
</body>
</html>
"""


@app.get("/admin/login", response_class=HTMLResponse, include_in_schema=False)
def admin_login_page():
    return HTMLResponse(content=ADMIN_LOGIN_HTML)


@app.post("/admin/login", include_in_schema=False)
def admin_login(body: AdminLoginRequest):
    if not ADMIN_PASS:
        raise HTTPException(status_code=403, detail="Admin login not configured (set ADMIN_PASS)")
    if body.username != ADMIN_USER or body.password != ADMIN_PASS:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    response = JSONResponse(content={"message": "OK"})
    response.set_cookie(
        key="mem0_admin",
        value=_admin_session_token,
        httponly=True,
        max_age=86400 * 7,  # 7 days
        samesite="lax",
    )
    return response


@app.get("/admin/logout", include_in_schema=False)
def admin_logout():
    response = RedirectResponse(url="/admin/login")
    response.delete_cookie("mem0_admin")
    return response


class CreateKeyRequest(BaseModel):
    user_id: str = Field(..., description="User ID to bind this key to")
    label: str = Field("", description="Human-readable label for this key")


@app.post("/admin/keys", summary="Create API key")
def create_api_key(body: CreateKeyRequest):
    """Generate a new API key bound to a user_id. Returns the plaintext key (shown only once)."""
    keys = _load_api_keys()
    for entry in keys.values():
        if entry.get("user_id") == body.user_id:
            raise HTTPException(status_code=409, detail=f"user_id '{body.user_id}' 已有 Key，请先删除旧 Key 再创建")
    raw_key = "mk_" + secrets.token_urlsafe(32)
    key_hash = _hash_key(raw_key)
    keys[key_hash] = {
        "user_id": body.user_id,
        "label": body.label,
        "raw_key": raw_key,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    _save_api_keys(keys)
    logging.info("Created API key for user_id=%s label=%s", body.user_id, body.label)
    return {"api_key": raw_key, "user_id": body.user_id, "label": body.label}


@app.get("/admin/keys", summary="List API keys")
def list_api_keys():
    """List all API keys (showing only hash prefix, not full key)."""
    keys = _load_api_keys()
    result = []
    for hash_val, entry in keys.items():
        result.append({
            "key_prefix": hash_val[:12],
            "user_id": entry.get("user_id"),
            "label": entry.get("label"),
            "raw_key": entry.get("raw_key", ""),
            "created_at": entry.get("created_at"),
        })
    return {"keys": result}


@app.delete("/admin/keys/{key_prefix}", summary="Delete API key")
def delete_api_key(key_prefix: str):
    """Delete an API key by its hash prefix."""
    keys = _load_api_keys()
    to_delete = [h for h in keys if h.startswith(key_prefix)]
    if not to_delete:
        raise HTTPException(status_code=404, detail="No key found with that prefix")
    if len(to_delete) > 1:
        raise HTTPException(status_code=409, detail=f"Ambiguous prefix, matches {len(to_delete)} keys. Use a longer prefix.")
    del keys[to_delete[0]]
    _save_api_keys(keys)
    logging.info("Deleted API key with prefix %s", key_prefix)
    return {"message": "Key deleted", "key_prefix": key_prefix}


@app.get("/admin/queue", summary="Queue status")
def queue_status():
    return {"waiting": _queue_waiting, "processing": _queue_processing, "max_concurrent": _MAX_CONCURRENT}


@app.get("/admin/users", summary="List all users with memory counts")
def admin_list_users():
    """Scroll through all Qdrant points and aggregate by user_id/agent_id."""
    vs = MEMORY_INSTANCE.vector_store
    user_stats: Dict[str, Dict[str, Any]] = {}  # key → {count, latest}
    next_offset = None
    while True:
        points, next_offset = vs.client.scroll(
            collection_name=vs.collection_name,
            limit=1000,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            uid = p.payload.get("user_id") or p.payload.get("agent_id") or "unknown"
            kind = "user_id" if p.payload.get("user_id") else ("agent_id" if p.payload.get("agent_id") else "unknown")
            key = f"{kind}:{uid}"
            if key not in user_stats:
                user_stats[key] = {"id_type": kind, "id_value": uid, "count": 0, "latest": ""}
            user_stats[key]["count"] += 1
            ts = p.payload.get("updated_at") or p.payload.get("created_at") or ""
            if ts > user_stats[key]["latest"]:
                user_stats[key]["latest"] = ts
        if next_offset is None:
            break
    result = sorted(user_stats.values(), key=lambda x: x["count"], reverse=True)
    return {"users": result, "total_memories": sum(u["count"] for u in result)}


@app.get("/admin/memories/browse", summary="Browse memories by user/agent")
def admin_browse_memories(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List memories with pagination, optionally filtered by user_id or agent_id."""
    vs = MEMORY_INSTANCE.vector_store
    filters = {}
    if user_id:
        filters["user_id"] = user_id
    if agent_id:
        filters["agent_id"] = agent_id

    query_filter = None
    if filters:
        from qdrant_client.models import FieldCondition, MatchValue, Filter as QFilter
        conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
        query_filter = QFilter(must=conditions)

    # Get total count
    total = vs.client.count(collection_name=vs.collection_name, count_filter=query_filter).count

    # Scroll with offset simulation (qdrant scroll doesn't support numeric offset well, so we fetch and skip)
    all_points = []
    next_off = None
    needed = offset + limit
    while len(all_points) < needed:
        batch, next_off = vs.client.scroll(
            collection_name=vs.collection_name,
            scroll_filter=query_filter,
            limit=min(1000, needed - len(all_points)),
            offset=next_off,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(batch)
        if next_off is None:
            break

    page = all_points[offset:offset + limit]
    memories = []
    for p in page:
        memories.append({
            "id": str(p.id),
            "memory": p.payload.get("data", ""),
            "user_id": p.payload.get("user_id"),
            "agent_id": p.payload.get("agent_id"),
            "hash": p.payload.get("hash"),
            "created_at": p.payload.get("created_at"),
            "updated_at": p.payload.get("updated_at"),
            "metadata": {k: v for k, v in p.payload.items()
                         if k not in ("data", "user_id", "agent_id", "run_id", "hash", "created_at", "updated_at")},
        })
    return {"memories": memories, "total": total, "offset": offset, "limit": limit}


@app.delete("/admin/memories/{memory_id}", summary="Delete memory (admin)")
def admin_delete_memory(memory_id: str):
    """Delete a memory via admin dashboard."""
    try:
        MEMORY_INSTANCE.delete(memory_id=memory_id)
        return {"message": "Memory deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/", summary="Admin dashboard", response_class=HTMLResponse, include_in_schema=False)
def admin_page():
    return HTMLResponse(content=ADMIN_HTML)


ADMIN_HTML = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mem0 Admin</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#FAF9F7;color:#1a1a1a;line-height:1.6;
  min-height:100vh;padding:48px 16px;
}
.container{max-width:800px;margin:0 auto}
h1{font-size:22px;font-weight:600;margin-bottom:4px}
.subtitle{color:#666;font-size:14px;margin-bottom:32px}

/* Card */
.card{
  background:#fff;border:1px solid #e8e6e3;border-radius:12px;
  padding:24px;margin-bottom:20px;
  box-shadow:0 1px 3px rgba(0,0,0,.04);
}
.card h2{font-size:15px;font-weight:600;margin-bottom:16px;color:#333}

/* Form */
.form-row{display:flex;gap:10px;margin-bottom:12px}
.form-row input{
  flex:1;padding:9px 12px;border:1px solid #ddd;border-radius:8px;
  font-size:14px;outline:none;transition:border .15s;
  background:#fafafa;
}
.form-row input:focus{border-color:#D97757;background:#fff}
.form-row input.error{border-color:#c44;background:#fef8f7}
.form-row input.error::placeholder{color:#daa}
.btn{
  padding:9px 18px;border:none;border-radius:8px;font-size:14px;
  font-weight:500;cursor:pointer;transition:all .15s;
  display:inline-flex;align-items:center;gap:6px;
}
.btn-primary{background:#D97757;color:#fff}
.btn-primary:hover:not(:disabled){background:#c4684a}
.btn-primary:active:not(:disabled){transform:scale(.97)}
.btn-primary:disabled{opacity:.5;cursor:not-allowed}
.btn-copy-key{background:#D97757;color:#fff;border:none;border-radius:6px;padding:5px 12px;font-size:12px;cursor:pointer;transition:all .15s;white-space:nowrap}
.btn-copy-key:hover{background:#c4684a}
.btn-copy-key.copied{background:#5a9a6a}
.btn-danger{background:none;color:#999;padding:6px 10px;font-size:13px}
.btn-danger:hover{color:#c44;background:#fef2f0;border-radius:6px}

/* Key display */
.key-display{
  margin-top:14px;padding:14px 16px;
  background:#FDF6F0;border:1px solid #f0d9c8;border-radius:10px;
  display:none;animation:fadeIn .25s;
}
.key-display .label{font-size:12px;color:#996644;font-weight:500;margin-bottom:6px}
.key-display .key-value{
  font-family:'SF Mono',Monaco,Consolas,monospace;font-size:13px;
  word-break:break-all;color:#1a1a1a;
  display:flex;align-items:flex-start;gap:8px;
}
.key-display .key-value code{flex:1}
.btn-copy{
  background:#D97757;color:#fff;border:none;border-radius:6px;
  padding:4px 10px;font-size:12px;cursor:pointer;white-space:nowrap;
  transition:all .15s;flex-shrink:0;
}
.btn-copy:hover{background:#c4684a}
.btn-copy.copied{background:#5a9a6a}

/* Table */
.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
thead th{
  text-align:left;padding:8px 10px;color:#888;font-weight:500;
  font-size:12px;text-transform:uppercase;letter-spacing:.5px;
  border-bottom:1px solid #eee;
}
tbody td{padding:10px;border-bottom:1px solid #f4f3f1;vertical-align:middle}
tbody tr:hover{background:#faf9f7}
.mono{font-family:'SF Mono',Monaco,Consolas,monospace;font-size:12px;color:#666}
.empty{text-align:center;padding:32px;color:#aaa;font-size:14px}

/* Misc */
@keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
.loading{opacity:.5;pointer-events:none}
.toast{
  position:fixed;bottom:24px;left:50%;transform:translateX(-50%);
  background:#333;color:#fff;padding:10px 20px;border-radius:8px;
  font-size:13px;opacity:0;transition:opacity .2s;pointer-events:none;
}
.toast.show{opacity:1}

/* Queue status */
.queue-bar{
  display:flex;align-items:center;gap:8px;
  padding:10px 16px;border-radius:10px;
  background:#fff;border:1px solid #e8e6e3;
  margin-bottom:20px;font-size:13px;color:#666;
  box-shadow:0 1px 3px rgba(0,0,0,.04);
}
.queue-dot{
  width:8px;height:8px;border-radius:50%;flex-shrink:0;
}
.queue-dot.idle{background:#5a9a6a}
.queue-dot.busy{background:#d4882a;animation:pulse 1.2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* Memory items */
.mem-item{
  padding:12px 14px;border-bottom:1px solid #f0efed;
  font-size:13px;animation:fadeIn .2s;
}
.mem-item:last-child{border-bottom:none}
.mem-item:hover{background:#faf9f7}
.mem-content{margin-bottom:4px;line-height:1.5}
.mem-meta{color:#999;font-size:11px;display:flex;gap:12px;flex-wrap:wrap}
.mem-meta span{white-space:nowrap}
.badge{
  display:inline-block;padding:1px 6px;border-radius:4px;
  font-size:11px;font-weight:500;cursor:pointer;
}
.badge-user{background:#E8F0FE;color:#1967D2}
.badge-agent{background:#FEF3E0;color:#E37400}
.btn-del-mem{
  background:none;border:none;color:#ccc;cursor:pointer;font-size:12px;
  padding:2px 6px;border-radius:4px;transition:all .15s;
}
.btn-del-mem:hover{color:#c44;background:#fef2f0}
</style>
</head>
<body>
<div class="container">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div><h1>Mem0 Admin</h1><p class="subtitle">管理 API Keys、用户与记忆</p></div>
    <a href="/admin/logout" style="color:#999;font-size:13px;text-decoration:none">退出登录</a>
  </div>

  <!-- Queue status -->
  <div class="queue-bar" id="queue-bar">
    <span class="queue-dot idle" id="queue-dot"></span>
    <span id="queue-text">空闲</span>
  </div>

  <!-- Create -->
  <div class="card">
    <h2>创建 Key</h2>
    <div class="form-row">
      <input id="inp-uid" type="text" placeholder="user_id" value="heasenbug">
      <input id="inp-label" type="text" placeholder="标签（如：Mac-mini）">
      <button class="btn btn-primary" id="btn-create">创建</button>
    </div>
    <div class="key-display" id="key-display">
      <div class="label">API Key（仅显示一次，请立即复制）</div>
      <div class="key-value">
        <code id="key-text"></code>
        <button class="btn-copy" id="btn-copy">复制</button>
      </div>
    </div>
  </div>

  <!-- List -->
  <div class="card">
    <h2>已有 Keys</h2>
    <div class="table-wrap" id="key-list"></div>
  </div>

  <!-- Users -->
  <div class="card">
    <h2>用户概览 <span id="total-memories" style="font-weight:400;color:#888;font-size:13px"></span></h2>
    <div class="table-wrap" id="user-list"><div class="empty">加载中…</div></div>
  </div>

  <!-- Memory browser -->
  <div class="card">
    <h2>记忆浏览</h2>
    <div class="form-row" style="margin-bottom:14px">
      <input id="inp-browse-uid" type="text" placeholder="user_id（可选）">
      <input id="inp-browse-aid" type="text" placeholder="agent_id（可选）">
      <button class="btn btn-primary" id="btn-browse">查询</button>
    </div>
    <div id="mem-pager" style="display:none;margin-bottom:10px;font-size:13px;color:#888;display:flex;justify-content:space-between;align-items:center">
      <span id="mem-info"></span>
      <span>
        <button class="btn" id="btn-prev" style="padding:4px 10px;font-size:12px" disabled>上一页</button>
        <button class="btn" id="btn-next" style="padding:4px 10px;font-size:12px" disabled>下一页</button>
      </span>
    </div>
    <div class="table-wrap" id="mem-list"><div class="empty">输入条件后点击查询</div></div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const $ = s => document.querySelector(s);
let _cachedKeys = [];

function toast(msg) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 2000);
}

function clearErrors() {
  document.querySelectorAll('.form-row input').forEach(i => i.classList.remove('error'));
}

async function loadKeys() {
  const wrap = $('#key-list');
  try {
    const res = await fetch('/admin/keys');
    const data = await res.json();
    _cachedKeys = data.keys || [];
    if (_cachedKeys.length === 0) {
      wrap.innerHTML = '<div class="empty">暂无 Key</div>';
      return;
    }
    let html = '<table><thead><tr><th>Prefix</th><th>User</th><th>标签</th><th>创建时间</th><th></th><th></th></tr></thead><tbody>';
    for (const k of _cachedKeys) {
      const t = k.created_at ? new Date(k.created_at).toLocaleString('zh-CN', {dateStyle:'short',timeStyle:'short'}) : '-';
      const copyBtn = k.raw_key
        ? `<button class="btn btn-copy-key" onclick="copyKey(this,'${esc(k.raw_key)}')">复制 Key</button>`
        : '<span style="color:#aaa;font-size:12px">旧 Key</span>';
      html += `<tr>
        <td class="mono">${esc(k.key_prefix)}</td>
        <td>${esc(k.user_id||'')}</td>
        <td>${esc(k.label||'-')}</td>
        <td style="color:#888">${t}</td>
        <td>${copyBtn}</td>
        <td><button class="btn btn-danger" onclick="delKey('${esc(k.key_prefix)}')">删除</button></td>
      </tr>`;
    }
    html += '</tbody></table>';
    wrap.innerHTML = html;
  } catch(e) {
    wrap.innerHTML = '<div class="empty">加载失败</div>';
  }
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

$('#btn-create').addEventListener('click', async () => {
  clearErrors();
  const inpUid = $('#inp-uid'), inpLabel = $('#inp-label');
  const uid = inpUid.value.trim();
  const label = inpLabel.value.trim();

  // Validate required fields
  let hasError = false;
  if (!uid) { inpUid.classList.add('error'); hasError = true; }
  if (!label) { inpLabel.classList.add('error'); hasError = true; }
  if (hasError) { toast(!uid ? '请输入 user_id' : '请输入标签'); return; }

  // Check user_id uniqueness
  const dup = _cachedKeys.find(k => k.user_id === uid);
  if (dup) {
    inpUid.classList.add('error');
    toast('该 user_id 已有 Key，请先删除旧 Key');
    return;
  }

  const btn = $('#btn-create');
  btn.disabled = true;
  btn.textContent = '创建中…';
  try {
    const res = await fetch('/admin/keys', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({user_id: uid, label: label})
    });
    const data = await res.json();
    if (data.api_key) {
      const display = $('#key-display');
      $('#key-text').textContent = data.api_key;
      display.style.display = 'block';
      display.style.animation = 'none';
      display.offsetHeight;
      display.style.animation = '';
      $('#btn-copy').textContent = '复制';
      $('#btn-copy').classList.remove('copied');
      inpLabel.value = '';
      loadKeys();
    } else {
      toast('创建失败: ' + (data.detail||'未知错误'));
    }
  } catch(e) {
    toast('请求失败');
  } finally {
    btn.disabled = false;
    btn.textContent = '创建';
  }
});

// Clear error style on input
document.querySelectorAll('.form-row input').forEach(inp => {
  inp.addEventListener('input', () => inp.classList.remove('error'));
});

$('#btn-copy').addEventListener('click', () => {
  const key = $('#key-text').textContent;
  navigator.clipboard.writeText(key).then(() => {
    $('#btn-copy').textContent = '已复制';
    $('#btn-copy').classList.add('copied');
  });
});

function copyKey(btn, key) {
  navigator.clipboard.writeText(key).then(() => {
    const orig = btn.textContent;
    btn.textContent = '已复制';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = orig; btn.classList.remove('copied'); }, 1500);
  });
}

async function delKey(prefix) {
  if (!confirm('确定删除 Key ' + prefix + '？')) return;
  try {
    const res = await fetch('/admin/keys/' + prefix, {method:'DELETE'});
    if (res.ok) {
      toast('已删除');
      loadKeys();
    } else {
      const d = await res.json();
      toast('删除失败: ' + (d.detail||''));
    }
  } catch(e) {
    toast('请求失败');
  }
}

loadKeys();

// Queue status polling
async function pollQueue() {
  try {
    const res = await fetch('/admin/queue');
    const data = await res.json();
    const dot = $('#queue-dot');
    const txt = $('#queue-text');
    const active = data.processing || 0;
    const waiting = data.waiting || 0;
    if (active > 0 || waiting > 0) {
      dot.className = 'queue-dot busy';
      let s = active + ' 个处理中';
      if (waiting > 0) s += '，' + waiting + ' 个排队';
      txt.textContent = s;
    } else {
      dot.className = 'queue-dot idle';
      txt.textContent = '空闲';
    }
  } catch(e) {}
}
pollQueue();
setInterval(pollQueue, 2000);

// ===== Users module =====
async function loadUsers() {
  const wrap = $('#user-list');
  try {
    const res = await fetch('/admin/users');
    const data = await res.json();
    $('#total-memories').textContent = '共 ' + data.total_memories + ' 条记忆';
    const users = data.users || [];
    if (users.length === 0) {
      wrap.innerHTML = '<div class="empty">暂无用户</div>';
      return;
    }
    let html = '<table><thead><tr><th>类型</th><th>ID</th><th>记忆数</th><th>最近更新</th><th></th></tr></thead><tbody>';
    for (const u of users) {
      const t = u.latest ? new Date(u.latest).toLocaleString('zh-CN', {dateStyle:'short',timeStyle:'short'}) : '-';
      const badge = u.id_type === 'user_id' ? 'badge-user' : 'badge-agent';
      html += '<tr>' +
        '<td><span class="badge ' + badge + '">' + esc(u.id_type) + '</span></td>' +
        '<td>' + esc(u.id_value) + '</td>' +
        '<td style="font-weight:600">' + u.count + '</td>' +
        '<td style="color:#888">' + t + '</td>' +
        '<td><button class="btn" style="padding:4px 10px;font-size:12px" ' +
          'onclick="browseUser(\'' + esc(u.id_type) + '\',\'' + esc(u.id_value) + '\')">查看</button></td>' +
        '</tr>';
    }
    html += '</tbody></table>';
    wrap.innerHTML = html;
  } catch(e) {
    wrap.innerHTML = '<div class="empty">加载失败</div>';
  }
}
loadUsers();

function browseUser(idType, idValue) {
  if (idType === 'user_id') {
    $('#inp-browse-uid').value = idValue;
    $('#inp-browse-aid').value = '';
  } else {
    $('#inp-browse-aid').value = idValue;
    $('#inp-browse-uid').value = '';
  }
  _memOffset = 0;
  loadMemories();
  // Scroll to memory browser
  $('#mem-list').scrollIntoView({behavior:'smooth', block:'start'});
}

// ===== Memory browser module =====
let _memOffset = 0;
const _memLimit = 20;

$('#btn-browse').addEventListener('click', () => {
  _memOffset = 0;
  loadMemories();
});

$('#btn-prev').addEventListener('click', () => {
  _memOffset = Math.max(0, _memOffset - _memLimit);
  loadMemories();
});

$('#btn-next').addEventListener('click', () => {
  _memOffset += _memLimit;
  loadMemories();
});

async function loadMemories() {
  const wrap = $('#mem-list');
  const uid = $('#inp-browse-uid').value.trim();
  const aid = $('#inp-browse-aid').value.trim();
  let url = '/admin/memories/browse?limit=' + _memLimit + '&offset=' + _memOffset;
  if (uid) url += '&user_id=' + encodeURIComponent(uid);
  if (aid) url += '&agent_id=' + encodeURIComponent(aid);

  wrap.innerHTML = '<div class="empty">加载中…</div>';
  try {
    const res = await fetch(url);
    const data = await res.json();
    const mems = data.memories || [];
    const total = data.total || 0;
    $('#mem-info').textContent = '共 ' + total + ' 条，当前 ' + (total > 0 ? _memOffset + 1 : 0) + '-' + Math.min(_memOffset + _memLimit, total);
    $('#mem-pager').style.display = total > 0 ? 'flex' : 'none';
    $('#btn-prev').disabled = _memOffset === 0;
    $('#btn-next').disabled = _memOffset + _memLimit >= total;

    if (mems.length === 0) {
      wrap.innerHTML = '<div class="empty">无记忆</div>';
      return;
    }
    let html = '';
    for (const m of mems) {
      const t = (m.updated_at || m.created_at || '');
      const tStr = t ? new Date(t).toLocaleString('zh-CN', {dateStyle:'short',timeStyle:'short'}) : '';
      html += '<div class="mem-item">' +
        '<div class="mem-content">' + esc(m.memory) + '</div>' +
        '<div class="mem-meta">' +
          (m.user_id ? '<span class="badge badge-user">' + esc(m.user_id) + '</span>' : '') +
          (m.agent_id ? '<span class="badge badge-agent">' + esc(m.agent_id) + '</span>' : '') +
          '<span>' + tStr + '</span>' +
          '<span class="mono">' + esc((m.id||'').substring(0,8)) + '</span>' +
          '<button class="btn-del-mem" onclick="delMem(\'' + esc(m.id) + '\',this)">删除</button>' +
        '</div>' +
      '</div>';
    }
    wrap.innerHTML = html;
  } catch(e) {
    wrap.innerHTML = '<div class="empty">加载失败</div>';
  }
}

async function delMem(id, btn) {
  if (!confirm('确定删除此条记忆？')) return;
  try {
    const res = await fetch('/admin/memories/' + id, {method:'DELETE'});
    if (res.ok) {
      btn.closest('.mem-item').style.opacity = '0';
      setTimeout(() => { btn.closest('.mem-item').remove(); }, 200);
      toast('已删除');
      loadUsers(); // refresh counts
    } else {
      toast('删除失败');
    }
  } catch(e) { toast('请求失败'); }
}
</script>
</body>
</html>
"""


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    limit: Optional[int] = None
    filters: Optional[Dict[str, Any]] = None


class MetadataUpdateRequest(BaseModel):
    metadata: Dict[str, Any]


class BatchMetadataItem(BaseModel):
    memory_id: str
    metadata: Dict[str, Any]


class BatchMetadataUpdateRequest(BaseModel):
    updates: List[BatchMetadataItem]


@app.post("/configure", summary="Configure Mem0")
def set_config(config: Dict[str, Any]):
    """Set memory configuration."""
    global MEMORY_INSTANCE
    MEMORY_INSTANCE = Memory.from_config(config)
    return {"message": "Configuration set successfully"}


@app.post("/memories", summary="Create memories")
def add_memory(memory_create: MemoryCreate):
    """Store new memories."""
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
    try:
        response = _call_with_semaphore(
            lambda: MEMORY_INSTANCE.add(messages=[m.model_dump() for m in memory_create.messages], **params)
        )
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories", summary="Get memories")
def get_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    limit: Optional[int] = None,
):
    """Retrieve stored memories."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        result = MEMORY_INSTANCE.get_all(**params)
        if limit:
            if isinstance(result, dict) and "results" in result:
                result["results"] = result["results"][:limit]
            elif isinstance(result, list):
                result = result[:limit]
        return result
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}", summary="Get a memory")
def get_memory(memory_id: str):
    """Retrieve a specific memory by ID."""
    try:
        return MEMORY_INSTANCE.get(memory_id)
    except Exception as e:
        logging.exception("Error in get_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", summary="Search memories")
def search_memories(search_req: SearchRequest):
    """Search for memories based on a query."""
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        return _call_with_semaphore(
            lambda: MEMORY_INSTANCE.search(query=search_req.query, **params)
        )
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memories/{memory_id}", summary="Update a memory")
def update_memory(memory_id: str, updated_memory: Dict[str, Any]):
    """Update an existing memory with new content.
    
    Args:
        memory_id (str): ID of the memory to update
        updated_memory (str): New content to update the memory with
        
    Returns:
        dict: Success message indicating the memory was updated
    """
    try:
        return MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory.get("data", str(updated_memory)))
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}/history", summary="Get memory history")
def memory_history(memory_id: str):
    """Retrieve memory history."""
    try:
        return MEMORY_INSTANCE.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}", summary="Delete a memory")
def delete_memory(memory_id: str):
    """Delete a specific memory by ID."""
    try:
        MEMORY_INSTANCE.delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories", summary="Delete all memories")
def delete_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
):
    """Delete all memories for a given identifier."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        MEMORY_INSTANCE.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/memories/batch/metadata", summary="Batch update metadata")
def batch_update_metadata(body: BatchMetadataUpdateRequest):
    """Merge metadata into multiple memories at once."""
    results = []
    for item in body.updates:
        try:
            existing = MEMORY_INSTANCE.get(item.memory_id)
            current = existing.get("metadata", {}) or {}
            for k, v in item.metadata.items():
                if v is None:
                    current.pop(k, None)
                else:
                    current[k] = v
            MEMORY_INSTANCE.update(memory_id=item.memory_id, data=existing.get("memory", ""))
            results.append({"memory_id": item.memory_id, "status": "updated"})
        except Exception as e:
            results.append({"memory_id": item.memory_id, "status": "error", "detail": str(e)})
    return {"results": results}


@app.patch("/memories/{memory_id}/metadata", summary="Update memory metadata")
def update_metadata(memory_id: str, body: MetadataUpdateRequest):
    """Merge new metadata keys into an existing memory."""
    try:
        existing = MEMORY_INSTANCE.get(memory_id)
        current = existing.get("metadata", {}) or {}
        for k, v in body.metadata.items():
            if v is None:
                current.pop(k, None)
            else:
                current[k] = v
        MEMORY_INSTANCE.update(memory_id=memory_id, data=existing.get("memory", ""))
        return {"status": "updated", "metadata": current}
    except Exception as e:
        logging.exception("Error in update_metadata:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", summary="Reset all memories")
def reset_memory():
    """Completely reset stored memories."""
    try:
        MEMORY_INSTANCE.reset()
        return {"message": "All memories reset"}
    except Exception as e:
        logging.exception("Error in reset_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")
