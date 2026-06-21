import sys
import asyncio

# Windows + Python 3.12+: Playwright (and asyncio subprocesses in general) require
# ProactorEventLoop to spawn child processes. Must be set before uvicorn initialises
# its own event loop, which is why this sits above every other import.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

"""
main.py  —  FastAPI Application Entry Point

WebSocket message types (client → server):
    { "type": "message",               "text": "…" }
    { "type": "confirm",               "confirmation_id": "…", "approved": bool }
    { "type": "clear" }
    { "type": "stop_task" }                                      ← Phase 3b
    { "type": "plan_response",         "plan_id": "…", "approved": bool,
                                       "edited_steps": [{…}] | null }   ← Phase 3e
    { "type": "set_optimizer",         "data": {"enabled": bool} }
    { "type": "set_local_mode",        "data": {"enabled": bool} }
    { "type": "set_model",             "data": {"model": "…"} }
    { "type": "set_local_agent_model", "data": {"model": "…"} }
    { "type": "set_config",            "data": {"key": "context.recent_turns", "value": N} }
    { "type": "stop_process",          "data": {"name": "…"} }          ← Phase 7
    { "type": "cancel_schedule",       "data": {"task_id": "…"} }       ← Phase 7
    { "type": "schedule_task",         "data": {"task_id","message","schedule"} } ← Phase 7

WebSocket event types (server → client):
    status | prompt_optimized | tool_call | tool_result | tool_denied
    confirm_required | message | error | cleared | optimizer_status
    tree_update | local_mode_status | model_status | local_agent_model_status | config_ack
    task_started | task_progress | task_stopped                   ← Phase 3b
    task_plan                                                      ← Phase 3e
    process_stopped | schedule_updated                             ← Phase 7
"""

import asyncio
import json
import logging
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from agent_core import AgentCore
from task_runner import TaskRunner                                   # Phase 3b
from memory.context import load_history, save_history, trim_history
from memory.embeddings import store_turn, search_similar, clear_all as clear_vectors
from agent_tools.filesystem import register_all as register_filesystem_tools
from agent_tools.capabilities import register_capabilities_tools
from agent_tools.local_llm import is_ollama_available, summarize_history, unload_model
from agent_tools.web import register_web_tools
from agent_tools.system_info import register_system_tools
from agent_tools.file_analysis import register_file_analysis_tools
from agent_tools.code_executor import register_code_executor_tools, set_send_event_callback
from agent_tools.tool_writer import register_tool_writer_tools          # Phase 3c
from agent_tools.hot_reload import hot_reload_tool, list_generated_tools  # Phase 3c
from agent_tools.memory_tool import register_memory_tools                 # Phase 3f
from agent_tools.self_knowledge import register_self_knowledge_tools      # Phase 3g
from agent_tools.profile_updater import register_profile_updater_tools    # Phase 3g
from agent_tools.research_mode import register_research_tools              # Phase 3h
from agent_tools.project_scaffold import register_scaffold_tools            # Phase 4a
from agent_tools.project_manager import register_project_manager_tools      # Phase 4b
from agent_tools.project_tester import register_project_tester_tools        # Phase 4b/4c
from agent_tools.github_tool import register_github_tools                   # Phase 5a
from agent_tools.credentials import register_credential_tools               # Phase 5b
from agent_tools.youtube_tool import register_youtube_tools                 # Phase 5c
from agent_tools.process_manager import register_process_tools, cleanup_all_processes  # Phase 5d
from task_scheduler import TaskScheduler                                    # Phase 5e
from agent_tools.scheduler_tool import register_scheduler_tools, set_scheduler  # Phase 5e
from agent_tools.interaction import (                                        # Phase 6a
    register_interaction_tools,
    set_task_runner as set_interaction_runner,
    set_send_event as set_interaction_event,
)

# Phase 9 — Media, notifications, file watching, email inbox tools
try:
    from agent_tools.media_tool import register_media_tools                  # Phase 9a
except ImportError:
    register_media_tools = None  # type: ignore

try:
    from agent_tools.notification_tool import register_notification_tools    # Phase 9b
except ImportError:
    register_notification_tools = None  # type: ignore

try:
    from agent_tools.file_watcher import (                                   # Phase 9c
        register_file_watcher_tools,
        _manager as file_watcher_manager,
    )
except ImportError:
    register_file_watcher_tools = None  # type: ignore
    file_watcher_manager        = None  # type: ignore

try:
    from agent_tools.email_tool import register_email_tools                  # Phase 9d
except ImportError:
    register_email_tools = None  # type: ignore

# Phase 3i — browser tools (optional; silently skipped if Playwright not installed)
_browser_available = False
try:
    from agent_tools.browser import register_browser_tools, browser_tools_registered  # Phase 3i
except ImportError:
    register_browser_tools = None  # type: ignore
    browser_tools_registered = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent.parent / "config.json"

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.json not found at {CONFIG_PATH}.")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)

config = load_config()

# ---------------------------------------------------------------------------
# Phase 10 — Remote access auth token
# ---------------------------------------------------------------------------

def _ensure_auth_token(cfg: dict, cfg_path: Path) -> str:
    """
    Generate and persist a random auth token on first startup.
    Prints the token to the console so the user can copy it.
    The token is a 32-byte URL-safe random string (~256 bits of entropy).
    """
    remote_cfg = cfg.setdefault("remote_access", {})
    token = remote_cfg.get("auth_token", "")
    if not token:
        token = secrets.token_urlsafe(32)
        remote_cfg["auth_token"] = token
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        logger.info("=" * 60)
        logger.info("[remote] AUTH TOKEN GENERATED (first run):")
        logger.info(f"[remote] {token}")
        logger.info("[remote] Copy this token — you need it to connect remotely.")
        logger.info("[remote] It is saved in config.json and won't change.")
        logger.info("=" * 60)
    return token

AUTH_TOKEN   = _ensure_auth_token(config, CONFIG_PATH)
REQUIRE_AUTH = config.get("remote_access", {}).get("require_auth", True)

# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

register_filesystem_tools()
register_capabilities_tools()
register_web_tools()
register_system_tools()
register_file_analysis_tools()
register_code_executor_tools()
register_tool_writer_tools()                                             # Phase 3c
register_memory_tools()                                                  # Phase 3f
register_self_knowledge_tools()                                          # Phase 3g
register_profile_updater_tools()                                         # Phase 3g
register_research_tools()                                                 # Phase 3h
register_scaffold_tools()                                                   # Phase 4a
register_project_manager_tools()                                            # Phase 4b
register_project_tester_tools()                                             # Phase 4b/4c

# Phase 5a — GitHub integration (registered even without a token; tools return
# a helpful error message if GITHUB_TOKEN is not set when called)
try:
    register_github_tools()
except Exception as _gh_err:
    logger.warning(f"[startup] GitHub tool registration failed (non-fatal): {_gh_err}")
_github_token_status = "set" if os.getenv("GITHUB_TOKEN") else "NOT SET"
logger.info(f"[startup] github (token: {_github_token_status})")

# Phase 5b — Credential manager (Fernet-encrypted local storage)
try:
    register_credential_tools()
except Exception as _cred_err:
    logger.warning(f"[startup] Credential tool registration failed (non-fatal): {_cred_err}")

# Phase 5c — YouTube Data API integration
try:
    register_youtube_tools()
except Exception as _yt_err:
    logger.warning(f"[startup] YouTube tool registration failed (non-fatal): {_yt_err}")

# Phase 5d — Process manager (start/stop/monitor background processes)
try:
    register_process_tools()
except Exception as _pm_err:
    logger.warning(f"[startup] Process manager tool registration failed (non-fatal): {_pm_err}")

# Phase 5e — Scheduled tasks
# set_scheduler must happen before register_scheduler_tools so the tool
# handlers have a reference to the scheduler when first called.
task_scheduler = TaskScheduler()
set_scheduler(task_scheduler)
try:
    register_scheduler_tools()
except Exception as _sched_err:
    logger.warning(f"[startup] Scheduler tool registration failed (non-fatal): {_sched_err}")

# Phase 9a — ffmpeg media tools
if register_media_tools is not None:
    try:
        register_media_tools()
    except Exception as _media_err:
        logger.warning(f"[startup] Media tool registration failed (non-fatal): {_media_err}")
else:
    logger.info("[startup] media_tool.py not found — skipping media tools.")

# Phase 9b — email notification tools
if register_notification_tools is not None:
    try:
        register_notification_tools()
    except Exception as _notif_err:
        logger.warning(f"[startup] Notification tool registration failed (non-fatal): {_notif_err}")
else:
    logger.info("[startup] notification_tool.py not found — skipping notification tools.")

# Phase 9c — file watcher tools
if register_file_watcher_tools is not None:
    try:
        register_file_watcher_tools()
    except Exception as _fw_err:
        logger.warning(f"[startup] File watcher tool registration failed (non-fatal): {_fw_err}")
else:
    logger.info("[startup] file_watcher.py not found — skipping file watcher tools.")

# Phase 9d — IMAP email inbox management tools
if register_email_tools is not None:
    try:
        register_email_tools()
    except Exception as _email_err:
        logger.warning(f"[startup] Email tool registration failed (non-fatal): {_email_err}")
else:
    logger.info("[startup] email_tool.py not found — skipping email tools.")

# Phase 3i — register browser tools if Playwright is installed
if register_browser_tools is not None:
    try:
        register_browser_tools()
        _browser_available = browser_tools_registered
    except Exception as _br_err:
        logger.warning(f"[startup] Browser tool registration failed (non-fatal): {_br_err}")

try:
    from agent_tools.browser import browser_tools_registered as _btr
    _browser_available = _btr
except Exception:
    pass
_browser_log = ", browser (browser_open, browser_read, browser_screenshot)" if _browser_available else ""
logger.info(
    "[startup] Registered tools: filesystem (read_file, write_file, list_directory), "
    "capabilities (list_capabilities), "
    "web (search_web, fetch_page), "
    "system (get_system_info), "
    "file_analysis (analyze_file), "
    "code_executor (execute_code), "
    "tool_writer (write_tool, reload_tool), "
    "memory (log_research, recall_memory, log_fact), "
    "self_knowledge (read_user_profile, scan_system), "
    "profile_updater (update_user_profile), "
    "research (deep_research), "
    "project_scaffold (scaffold_project), "
    f"github (github_list_repos, github_create_repo, github_push_file, "
    f"github_read_file, github_list_files, github_create_issue), "
    f"credentials (store_credential, get_credential, list_credentials), "
    f"youtube (youtube_search, youtube_get_video_stats, youtube_get_trending, "
    f"youtube_get_video_comments, youtube_get_channel_info, youtube_search_captions), "
    f"interaction (ask_user), "
    f"project_manager (get_project_status, mark_file_complete, read_project_state)"
    f"{_browser_log}"
)

# ---------------------------------------------------------------------------
# Phase 3c — Auto-load agent-written tools from agent_tools/generated/
# ---------------------------------------------------------------------------

async def _autoload_generated_tools() -> None:
    """Import every .py file in agent_tools/generated/ that passes validation."""
    from pathlib import Path
    generated_dir = Path(__file__).parent / "agent_tools" / "generated"
    files = list_generated_tools()
    if not files:
        logger.info("[startup] No agent-generated tools to auto-load.")
        return
    for filename in files:
        path = generated_dir / filename
        success, msg = await hot_reload_tool(path, send_event=None)
        if success:
            logger.info(f"[startup] Auto-loaded generated tool: {filename}")
        else:
            logger.warning(f"[startup] Failed to auto-load {filename}: {msg}")

# ---------------------------------------------------------------------------
# Agent + TaskRunner instances
# Phase 3d: TaskRunner receives config so it can read compression_threshold.
# ---------------------------------------------------------------------------

agent       = AgentCore(config)
task_runner = TaskRunner(config=config)   # Phase 3b / 3d

# Phase 6a — wire interaction module refs now that both objects exist.
set_interaction_runner(task_runner)
register_interaction_tools()            # Phase 6a

# ---------------------------------------------------------------------------
# Phase 4.5 — Streaming execution output
# ---------------------------------------------------------------------------

_active_send_event: list = [None]

def _execution_output_callback(event_type: str, data: dict) -> None:
    fn = _active_send_event[0]
    if fn is None:
        return
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(fn(event_type, data))
    except Exception as e:
        logger.debug(f"[main] Streaming callback dispatch error (non-fatal): {e}")

set_send_event_callback(_execution_output_callback)
logger.info("[startup] Execution streaming callback registered.")

# ---------------------------------------------------------------------------
# Phase 5e — Active WebSocket connection tracking + broadcast helper
# ---------------------------------------------------------------------------

_active_connections: set = set()


async def _broadcast(event_type: str, data: dict) -> None:
    for ws in list(_active_connections):
        try:
            await ws.send_json({"type": event_type, "data": data})
        except Exception:
            _active_connections.discard(ws)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Personal AI Agent",
    description="Phase 10 — Remote Access",
    version="4.0.0",
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ---------------------------------------------------------------------------
# Phase 10 — HTTP auth middleware
# Skips auth for the root page, static assets, and the /login page itself.
# All other HTTP endpoints require a valid token via query param or header.
# ---------------------------------------------------------------------------

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not REQUIRE_AUTH:
        return await call_next(request)

    # Always allow: root page (so redirect to /login works), static files, login page
    if request.url.path in ("/", "/login", "/favicon.ico") or request.url.path.startswith("/static"):
        return await call_next(request)

    # Accept token via query param, X-Auth-Token header, or Authorization: Bearer header
    token = (
        request.query_params.get("token")
        or request.headers.get("X-Auth-Token")
        or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    )
    if token != AUTH_TOKEN:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Phase 10 — Login page (public, no auth required)
# ---------------------------------------------------------------------------

@app.get("/login", include_in_schema=False)
async def login_page():
    return HTMLResponse("""
<!DOCTYPE html><html><head><title>Assistant — Connect</title>
<style>
  body { font-family: monospace; background: #1e1e2e; color: #cdd6f4;
         display: flex; align-items: center; justify-content: center;
         height: 100vh; margin: 0; }
  .box { background: #313244; padding: 32px; border-radius: 12px; min-width: 320px; }
  h2 { margin-top: 0; }
  input { width: 100%; padding: 10px; margin: 8px 0 16px;
          background: #1e1e2e; color: #cdd6f4;
          border: 1px solid #45475a; border-radius: 6px;
          box-sizing: border-box; font-family: monospace; font-size: 14px; }
  button { width: 100%; padding: 10px; background: #89b4fa; color: #1e1e2e;
           border: none; border-radius: 6px; cursor: pointer;
           font-weight: bold; font-size: 14px; }
  button:hover { background: #b4d0fa; }
  .err { color: #f38ba8; font-size: 13px; margin-top: 8px; display: none; }
</style></head><body>
<div class="box">
  <h2>🤖 Assistant</h2>
  <p>Enter your access token to connect.</p>
  <input type="password" id="tok" placeholder="Paste token here" autofocus>
  <button onclick="go()">Connect</button>
  <div class="err" id="err">Invalid token — check config.json or server console.</div>
</div>
<script>
function go() {
  const t = document.getElementById('tok').value.trim();
  if (!t) return;
  // Validate by hitting /status with the token before redirecting
  fetch('/status?token=' + encodeURIComponent(t))
    .then(r => {
      if (r.ok) {
        window.location.href = '/?token=' + encodeURIComponent(t);
      } else {
        document.getElementById('err').style.display = 'block';
      }
    })
    .catch(() => { document.getElementById('err').style.display = 'block'; });
}
document.getElementById('tok').addEventListener('keydown', e => {
  if (e.key === 'Enter') go();
});
// If already have a token in URL, auto-connect
const t = new URLSearchParams(location.search).get('token');
if (t) window.location.href = '/?token=' + encodeURIComponent(t);
</script></body></html>
""")


@app.on_event("startup")
async def _on_startup() -> None:
    await _autoload_generated_tools()
    task_scheduler.set_refs(
        agent=agent,
        task_runner=task_runner,
        send_event=_broadcast,
        pending_confirmations=pending_confirmations,
    )
    task_scheduler.start()
    logger.info("[startup] Task scheduler started.")
    # Phase 9c: wire file watcher refs
    if file_watcher_manager is not None:
        file_watcher_manager.set_refs(
            task_runner=task_runner,
            agent=agent,
            send_event=_broadcast,
        )
        logger.info("[startup] File watcher refs set.")


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    task_scheduler.shutdown()
    cleanup_all_processes()
    # Phase 9c: stop all file watchers
    if file_watcher_manager is not None:
        file_watcher_manager.shutdown()
    await unload_model(agent.local_model, agent.ollama_url)
    logger.info("[shutdown] Local model unloaded.")
    try:
        from agent_tools.browser import close_browser
        await close_browser()
    except Exception as e:
        logger.warning(f"[shutdown] Browser cleanup failed (non-fatal): {e}")


@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


def _get_embeddings_count() -> int:
    try:
        from memory.embeddings import _get_collection
        col = _get_collection()
        return col.count()
    except Exception:
        return 0


@app.get("/status")
async def status():
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    ollama_ok   = await is_ollama_available(config.get("ollama_base_url", "http://localhost:11434"))
    return JSONResponse({
        "claude_api":              has_api_key,
        "ollama":                  ollama_ok,
        "use_prompt_optimizer":    agent.use_prompt_optimizer,
        "use_intent_routing":      agent.use_intent_routing,
        "use_tool_compression":    agent.use_tool_compression,
        "use_code_prevalidation":  agent.use_code_prevalidation,
        "use_tool_prefilter":      agent.use_tool_prefilter,
        "local_fallback":          config.get("local_fallback", True),
        "local_mode":              agent.local_mode,
        "primary_model":           agent.primary_model,
        "local_agent_model":       agent.local_agent_model,
        "models":                  config.get("llm", {}),
        "context":                 config.get("context", {}),
        "embeddings":              config.get("embeddings", {}),
        "local_agent_timeout":     agent.local_agent_timeout,
        "tree_root":               config.get("tree_root", "."),
        "embeddings_count":        _get_embeddings_count(),
        "profile_loaded":          (Path(__file__).resolve().parent.parent / "memory" / "user_profile.json").exists(),
        "browser_available":       _browser_available,
        "local_sufficient_default": agent.local_sufficient_default,   # Phase 9
        "auto_approve_code_execution": config.get("auto_approve_code_execution", False),
    })


@app.get("/ollama-models")
async def get_ollama_models():
    """
    Fetch all locally available Ollama models and return them as a list.
    Used by the settings panel to populate model dropdowns dynamically.
    Returns name, size in GB, and modification date for each model.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{config.get('ollama_base_url', 'http://localhost:11434')}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return {
                "success": True,
                "models": [
                    {
                        "name": m["name"],
                        "size_gb": round(m.get("size", 0) / 1e9, 1),
                        "modified": m.get("modified_at", ""),
                    }
                    for m in models
                ]
            }
    except Exception as e:
        return {"success": False, "models": [], "error": str(e)}


@app.get("/task")
async def get_task(history: int = 0):
    data = task_runner.load_last_task()
    result = data if data is not None else {}

    if history > 0:
        try:
            from memory.long_term import load as load_long_term
            lt = load_long_term()
            tasks = lt.get("tasks", [])
            result["history"] = tasks[-history:] if len(tasks) > history else tasks
        except Exception as e:
            logger.warning(f"[task] Could not load task history: {e}")
            result["history"] = []

    return JSONResponse(result)


@app.get("/memory")
async def get_memory():
    from memory.long_term import load as load_long_term
    try:
        data = load_long_term()
        return JSONResponse({
            "tasks":    data.get("tasks",    []),
            "facts":    data.get("facts",    []),
            "research": data.get("research", []),
        })
    except Exception as e:
        logger.warning(f"[memory] Could not load long-term store: {e}")
        return JSONResponse({"tasks": [], "facts": [], "research": []})


# ---------------------------------------------------------------------------
# Phase 7 — New REST endpoints
# ---------------------------------------------------------------------------

@app.get("/processes")
async def get_processes():
    """Return all tracked background processes and their current status."""
    from agent_tools.process_manager import list_processes
    try:
        result = await list_processes()
        return JSONResponse(result)
    except Exception as e:
        logger.warning(f"[processes] Could not list processes: {e}")
        return JSONResponse({"processes": []})


@app.get("/scheduled")
async def get_scheduled():
    """Return all scheduled tasks with next-run metadata."""
    try:
        return JSONResponse({"tasks": task_scheduler.list_scheduled()})
    except Exception as e:
        logger.warning(f"[scheduled] Could not list tasks: {e}")
        return JSONResponse({"tasks": []})


@app.get("/credentials")
async def get_credentials():
    """Return stored credential service names (never values)."""
    from agent_tools.credentials import list_credentials
    try:
        result = await list_credentials()
        return JSONResponse(result)
    except Exception as e:
        logger.warning(f"[credentials] Could not list credentials: {e}")
        return JSONResponse({"credentials": [], "count": 0})


@app.get("/analytics")
async def get_analytics():
    """
    Compute agent analytics from long_term.json.

    Returns:
        total          — total task count
        success        — number of successful tasks
        rate           — success rate as a percentage (0–100)
        avg_duration   — mean duration in seconds (rounded)
        top_tools      — list of {name, count} for the 5 most-used tools
    """
    from memory.long_term import load as load_long_term
    try:
        data = load_long_term()
    except Exception as e:
        logger.warning(f"[analytics] Could not load long-term store: {e}")
        return JSONResponse({"total": 0, "success": 0, "rate": 0, "avg_duration": 0, "top_tools": []})

    tasks = data.get("tasks", [])
    if not tasks:
        return JSONResponse({"total": 0, "success": 0, "rate": 0, "avg_duration": 0, "top_tools": []})

    success   = sum(1 for t in tasks if t.get("outcome") == "success")
    durations = [t.get("duration_seconds", 0) for t in tasks]

    tool_counts: dict[str, int] = {}
    for t in tasks:
        for tool in t.get("tools_used", []):
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    top_tools = sorted(tool_counts.items(), key=lambda x: -x[1])[:5]

    return JSONResponse({
        "total":        len(tasks),
        "success":      success,
        "rate":         round(success / len(tasks) * 100, 1),
        "avg_duration": round(sum(durations) / len(durations)),
        "top_tools":    [{"name": k, "count": v} for k, v in top_tools],
    })


# ---------------------------------------------------------------------------
# set_config helper
# ---------------------------------------------------------------------------

def _apply_config(key: str, value) -> None:
    parts = key.split(".")
    node  = config

    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]

    leaf = parts[-1]
    node[leaf] = value
    logger.info(f"[config] Set {key} = {value!r}")

    if key == "context.max_iterations_per_turn":
        agent.max_iterations = int(value)
    elif key == "local_agent_timeout":
        agent.local_agent_timeout = float(value)
    elif key == "use_intent_routing":
        agent.use_intent_routing = bool(value)
    elif key == "use_tool_compression":
        agent.use_tool_compression = bool(value)
    elif key == "use_code_prevalidation":
        agent.use_code_prevalidation = bool(value)
    elif key == "use_tool_prefilter":
        agent.use_tool_prefilter = bool(value)
    elif key == "context.compression_threshold":
        task_runner._compression_threshold = int(value)
    elif key == "llm.max_tokens_primary":
        agent.max_tokens_primary = int(value)
    elif key == "llm.max_tokens_complex":
        agent.max_tokens_complex = int(value)
    elif key == "local_sufficient_default":
        if value in ("ask", "local", "claude"):
            agent.local_sufficient_default = value
        else:
            raise ValueError(f"Invalid local_sufficient_default value: {value!r}. Must be 'ask', 'local', or 'claude'.")


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

async def build_context(
    history: list[dict],
    current_message: str,
) -> tuple[str, list[dict]]:
    ctx_cfg  = config.get("context", {})
    emb_cfg  = config.get("embeddings", {})

    recent_turns       = ctx_cfg.get("recent_turns", 4)
    summary_threshold  = ctx_cfg.get("summary_threshold", 4)
    retrieval_n        = ctx_cfg.get("retrieval_n", 3)
    similarity_cutoff  = ctx_cfg.get("similarity_cutoff", 0.45)
    embed_enabled      = emb_cfg.get("enabled", True)
    embed_model        = emb_cfg.get("model", "nomic-embed-text")

    cutoff = recent_turns * 2
    recent = history[-cutoff:] if len(history) >= cutoff else history
    old    = history[:-cutoff]  if len(history) >  cutoff else []

    notes = []

    if len(old) >= summary_threshold * 2:
        summary = await summarize_history(
            turns=old,
            model=agent.local_model,
            base_url=agent.ollama_url,
        )
        if summary:
            notes.append(f"Summary of earlier conversation:\n{summary}")
            logger.debug(f"[ctx] Summary: {len(old)} old entries → {len(summary)} chars")

    if embed_enabled and current_message:
        recent_timestamps = {e.get("timestamp", "") for e in recent if e.get("timestamp")}

        retrieved = await search_similar(
            query=current_message,
            n_results=retrieval_n,
            model=embed_model,
            base_url=agent.ollama_url,
        )

        relevant = [
            r for r in retrieved
            if r["distance"] < similarity_cutoff
            and r["timestamp"] not in recent_timestamps
        ]

        if relevant:
            lines = []
            for r in relevant:
                u = r["user_content"][:200].replace("\n", " ")
                a = r["assistant_content"][:200].replace("\n", " ")
                lines.append(f"• User: {u}\n  Agent: {a}")
            notes.append(
                "Relevant context retrieved from memory:\n" + "\n".join(lines)
            )

    context_note   = "\n\n".join(notes)
    agent_messages = [{"role": e["role"], "content": e["content"]} for e in recent]

    return context_note, agent_messages


# ---------------------------------------------------------------------------
# Background embedding
# ---------------------------------------------------------------------------

async def _embed_turn_bg(
    timestamp: str,
    user_content: str,
    assistant_content: str,
) -> None:
    emb_cfg = config.get("embeddings", {})
    if not emb_cfg.get("enabled", True):
        return

    try:
        ok = await store_turn(
            timestamp=timestamp,
            user_content=user_content,
            assistant_content=assistant_content,
            model=emb_cfg.get("model", "nomic-embed-text"),
            base_url=agent.ollama_url,
        )
        if ok:
            logger.debug("[main] Background embed complete.")
    except Exception as e:
        logger.warning(f"[main] Background embed failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

pending_confirmations: dict = {}
pending_plans:         dict = {}   # Phase 3e


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = ""):
    # Phase 10 — reject connections with an invalid token
    if REQUIRE_AUTH and token != AUTH_TOKEN:
        await websocket.close(code=4401, reason="Unauthorized")
        logger.warning("[ws] Rejected connection — invalid or missing token")
        return

    await websocket.accept()
    _active_connections.add(websocket)
    logger.info("[ws] New WebSocket connection.")

    history: list[dict] = load_history()
    if history:
        logger.info(f"[ws] Resumed session — {len(history)} history entries loaded.")

    async def send_event(event_type: str, data) -> None:
        await websocket.send_json({"type": event_type, "data": data})

    try:
        from agent_tools.filesystem import _build_tree, _get_tree_root
        tree_str = _build_tree(_get_tree_root())
        await send_event("tree_update", {"tree": tree_str})
    except Exception as _te:
        logger.debug(f"[ws] Initial tree emit failed (non-fatal): {_te}")

    _active_send_event[0] = send_event
    set_interaction_event(send_event)

    async def dispatch(raw: dict):
        msg_type = raw.get("type")

        if msg_type == "message":
            user_text = raw.get("text", "").strip()
            if not user_text:
                return None

            if task_runner.is_running():
                logger.info(f"[ws] Task running — queuing mid-task message: {user_text[:60]!r}")
                await task_runner.inject_message(user_text)
                await send_event("status", {
                    "text": "Message queued — agent will read it after the current step."
                })
                return None

            logger.info(f"[ws] User: {user_text[:80]!r}")
            context_note, agent_messages = await build_context(history, user_text)
            return (user_text, context_note, agent_messages)

        elif msg_type == "plan_response":
            plan_id      = raw.get("plan_id")
            approved     = bool(raw.get("approved", False))
            edited_steps = raw.get("edited_steps", None)
            if plan_id in pending_plans:
                pending_plans[plan_id]["result"] = {
                    "approved":     approved,
                    "edited_steps": edited_steps,
                }
                pending_plans[plan_id]["event"].set()
            else:
                logger.warning(f"[ws] Unknown plan_id: {plan_id!r}")

        elif msg_type == "stop_task":
            logger.info("[ws] stop_task received.")
            task_runner.cancel()

        elif msg_type == "question_answer":
            question_id = raw.get("data", {}).get("question_id", "")
            answer      = raw.get("data", {}).get("answer", "")
            if question_id:
                task_runner.answer_question(question_id, answer)

        elif msg_type == "requeue_message":
            # Emitted by task_runner when a message arrived during the race window
            # between task completion and _is_running becoming False.
            # Re-process it as a fresh user message.
            requeue_text = raw.get("content", "")
            if requeue_text:
                logger.info(
                    f"[ws] Re-processing race-condition queued message: {requeue_text[:60]!r}"
                )
                context_note, agent_messages = await build_context(history, requeue_text)
                asyncio.create_task(
                    agent.run_with_task_runner(
                        task_runner=task_runner,
                        user_message=requeue_text,
                        history=agent_messages,
                        send_event=send_event,
                        pending_confirmations=pending_confirmations,
                        context_summary=context_note,
                        pending_plans=pending_plans,
                    )
                )

        # Phase 9 — tier choice response from the frontend banner
        elif msg_type == "tier_response":
            message_id = raw.get("data", {}).get("message_id", "")
            use_local  = bool(raw.get("data", {}).get("use_local", False))
            if message_id:
                agent.resolve_tier_choice(message_id, use_local)
            else:
                logger.warning("[ws] tier_response received with no message_id")

        elif msg_type == "confirm":
            confirmation_id = raw.get("confirmation_id")
            approved        = bool(raw.get("approved", False))
            if confirmation_id in pending_confirmations:
                pending_confirmations[confirmation_id]["result"] = approved
                pending_confirmations[confirmation_id]["event"].set()

        elif msg_type == "clear":
            history.clear()
            save_history(history)
            clear_vectors()
            await send_event("cleared", {"text": "Conversation history cleared."})

        elif msg_type == "set_optimizer":
            enabled = bool(raw.get("data", {}).get("enabled", True))
            agent.use_prompt_optimizer = enabled
            await send_event("optimizer_status", {"enabled": enabled})

        elif msg_type == "set_local_mode":
            enabled          = bool(raw.get("data", {}).get("enabled", False))
            agent.local_mode = enabled
            await send_event("local_mode_status", {"enabled": enabled})

        elif msg_type == "set_model":
            model = raw.get("data", {}).get("model", "").strip()
            if model:
                agent.primary_model = model
                config.setdefault("llm", {})["primary"] = model
                await send_event("model_status", {"model": model})

        elif msg_type == "set_local_agent_model":
            model = raw.get("data", {}).get("model", "").strip()
            if model:
                agent.local_agent_model = model
                config.setdefault("llm", {})["local_agent"] = model
                await send_event("local_agent_model_status", {"model": model})

        elif msg_type == "set_config":
            data_payload = raw.get("data", {})
            key   = data_payload.get("key", "").strip()
            value = data_payload.get("value")
            if key and value is not None:
                try:
                    _apply_config(key, value)
                    await send_event("config_ack", {"key": key, "value": value})
                except Exception as e:
                    await send_event("error", {"text": f"Could not apply config {key}: {e}"})

        # ----------------------------------------------------------------
        # Phase 7 — Process + Schedule WebSocket handlers
        # ----------------------------------------------------------------

        elif msg_type == "stop_process":
            # Stop a named background process
            name = raw.get("data", {}).get("name", "")
            if name:
                from agent_tools.process_manager import stop_process
                try:
                    result = await stop_process(name)
                    await send_event("process_stopped", {"name": name, "result": result})
                    logger.info(f"[ws] Stopped process '{name}'")
                except Exception as e:
                    await send_event("process_stopped", {"name": name, "result": {"error": str(e)}})

        elif msg_type == "cancel_schedule":
            # Cancel a scheduled task by task_id
            task_id = raw.get("data", {}).get("task_id", "")
            if task_id:
                try:
                    task_scheduler.cancel_task(task_id)
                    await send_event("schedule_updated", {"action": "cancelled", "task_id": task_id})
                    logger.info(f"[ws] Cancelled scheduled task '{task_id}'")
                except Exception as e:
                    await send_event("error", {"text": f"Could not cancel task '{task_id}': {e}"})

        elif msg_type == "schedule_task":
            # Add a new scheduled task
            d = raw.get("data", {})
            task_id      = d.get("task_id", "")
            message      = d.get("message", "")
            schedule_str = d.get("schedule", "")
            if task_id and message and schedule_str:
                try:
                    result = await task_scheduler.schedule_task(
                        task_id=task_id,
                        message=message,
                        schedule_str=schedule_str,
                    )
                    await send_event("schedule_updated", {"action": "added", "result": result})
                    logger.info(f"[ws] Scheduled task '{task_id}': {schedule_str!r}")
                except Exception as e:
                    await send_event("error", {"text": f"Could not schedule task: {e}"})

        else:
            logger.warning(f"[ws] Unknown message type: {msg_type!r}")

        return None

    try:
        while True:
            raw = await websocket.receive_json()
            result = await dispatch(raw)

            if result is None:
                continue

            user_text, context_note, agent_messages = result

            agent_task = asyncio.create_task(
                agent.run_with_task_runner(
                    task_runner=task_runner,
                    user_message=user_text,
                    history=agent_messages,
                    send_event=send_event,
                    pending_confirmations=pending_confirmations,
                    context_summary=context_note,
                    pending_plans=pending_plans,
                )
            )

            while not agent_task.done():
                recv_task = asyncio.create_task(websocket.receive_json())
                done, _ = await asyncio.wait(
                    {agent_task, recv_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if recv_task in done:
                    try:
                        incoming = recv_task.result()
                        await dispatch(incoming)
                    except Exception:
                        pass
                else:
                    recv_task.cancel()
                    try:
                        await recv_task
                    except (asyncio.CancelledError, Exception):
                        pass

            assistant_reply = await agent_task

            ts = datetime.now(timezone.utc).isoformat()
            history.append({"timestamp": ts, "role": "user",      "content": user_text})
            history.append({"timestamp": ts, "role": "assistant", "content": assistant_reply})

            max_turns = config.get("context", {}).get("max_history_turns", 20)
            history   = trim_history(history, max_turns)
            save_history(history)

            asyncio.create_task(_embed_turn_bg(ts, user_text, assistant_reply))

    except WebSocketDisconnect:
        logger.info("[ws] Client disconnected.")
        _active_connections.discard(websocket)
        _active_send_event[0] = None
        await unload_model(agent.local_model, agent.ollama_url)
    except Exception as e:
        logger.exception("[ws] Unhandled error in WebSocket handler")
        try:
            await websocket.send_json({"type": "error", "data": {"text": str(e)}})
        except Exception:
            pass
