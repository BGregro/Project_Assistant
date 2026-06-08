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

WebSocket event types (server → client):
    status | prompt_optimized | tool_call | tool_result | tool_denied
    confirm_required | message | error | cleared | optimizer_status
    tree_update | local_mode_status | model_status | local_agent_model_status | config_ack
    task_started | task_progress | task_stopped                   ← Phase 3b
    task_plan                                                      ← Phase 3e

Phase 3d changes:
    - TaskRunner now receives the config dict so it can read compression_threshold.
    - _apply_config() syncs the three new efficiency flags to agent attributes.
    - /status endpoint includes use_intent_routing, use_tool_compression,
      use_code_prevalidation so the settings panel can read them on page load.

Phase 3e changes:
    - pending_plans dict mirrors pending_confirmations; resolved by plan_response msgs.
    - agent.run_with_task_runner() now accepts pending_plans kwarg.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
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
from agent_tools.code_executor import register_code_executor_tools
from agent_tools.tool_writer import register_tool_writer_tools          # Phase 3c
from agent_tools.hot_reload import hot_reload_tool, list_generated_tools  # Phase 3c
from agent_tools.memory_tool import register_memory_tools                 # Phase 3f
from agent_tools.self_knowledge import register_self_knowledge_tools      # Phase 3g
from agent_tools.profile_updater import register_profile_updater_tools    # Phase 3g
from agent_tools.research_mode import register_research_tools              # Phase 3h

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
    "research (deep_research)"
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

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Personal AI Agent",
    description="Phase 3h — structured research mode",
    version="1.9.0",  # Phase 3h
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# Phase 3c: auto-load agent-written tools once the event loop is running.
@app.on_event("startup")
async def _on_startup() -> None:
    await _autoload_generated_tools()


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    """Unload the local model from RAM when the server process exits."""
    await unload_model(agent.local_model, agent.ollama_url)
    logger.info("[shutdown] Local model unloaded.")


@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


def _get_embeddings_count() -> int:
    """Return number of stored embedding entries, or 0 if unavailable."""
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
        # Phase 3d — new flags readable by the settings panel
        "use_intent_routing":      agent.use_intent_routing,
        "use_tool_compression":    agent.use_tool_compression,
        "use_code_prevalidation":  agent.use_code_prevalidation,
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
        # Phase 3g — user profile presence indicator
        "profile_loaded":          (Path(__file__).parent.parent / "memory" / "user_profile.json").exists(),
    })


# ---------------------------------------------------------------------------
# Phase 3b: GET /task — return last persisted task state
# ---------------------------------------------------------------------------

@app.get("/task")
async def get_task():
    data = task_runner.load_last_task()
    return JSONResponse(data if data is not None else {})


# ---------------------------------------------------------------------------
# Phase 3f: GET /memory — return long-term memory store as JSON
# ---------------------------------------------------------------------------

@app.get("/memory")
async def get_memory():
    """
    Return the full long-term memory store (tasks, facts, research).
    Useful for the Memory tab count display and debugging.
    """
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
# set_config helper — apply a dot-notation key to the live config dict
# and mirror any change to agent / task_runner attributes.
# ---------------------------------------------------------------------------

def _apply_config(key: str, value) -> None:
    """
    Apply a dot-notation config key to the live `config` dict, then mirror
    the change to any agent.* or task_runner.* attribute that caches it.

    Phase 3d additions:
        use_intent_routing       → agent.use_intent_routing
        use_tool_compression     → agent.use_tool_compression
        use_code_prevalidation   → agent.use_code_prevalidation
        context.compression_threshold → task_runner._compression_threshold
    """
    parts = key.split(".")
    node  = config

    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]

    leaf = parts[-1]
    node[leaf] = value
    logger.info(f"[config] Set {key} = {value!r}")

    # Mirror to agent attributes
    if key == "context.max_iterations_per_turn":
        agent.max_iterations = int(value)
    elif key == "local_agent_timeout":
        agent.local_agent_timeout = float(value)
    # Phase 3d — new efficiency flags
    elif key == "use_intent_routing":
        agent.use_intent_routing = bool(value)
    elif key == "use_tool_compression":
        agent.use_tool_compression = bool(value)
    elif key == "use_code_prevalidation":
        agent.use_code_prevalidation = bool(value)
    elif key == "context.compression_threshold":
        task_runner._compression_threshold = int(value)


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

async def build_context(
    history: list[dict],
    current_message: str,
) -> tuple[str, list[dict]]:
    """
    Assemble the minimal context Claude needs for the current turn.

    Returns:
        context_note   — string to append to the system prompt (may be empty)
        agent_messages — [{role, content}] list for the messages[] parameter
    """
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
            logger.debug(
                f"[ctx] Retrieved {len(relevant)} relevant turn(s) "
                f"(distances: {[round(r['distance'],2) for r in relevant]})"
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
pending_plans:         dict = {}   # Phase 3e — keyed by plan_id


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[ws] New WebSocket connection.")

    history: list[dict] = load_history()
    if history:
        logger.info(f"[ws] Resumed session — {len(history)} history entries loaded.")

    async def send_event(event_type: str, data) -> None:
        await websocket.send_json({"type": event_type, "data": data})

    async def dispatch(raw: dict):
        msg_type = raw.get("type")

        if msg_type == "message":
            user_text = raw.get("text", "").strip()
            if not user_text:
                return None

            if task_runner.is_running():
                logger.info(
                    f"[ws] Task running — queuing mid-task message: {user_text[:60]!r}"
                )
                await task_runner.inject_message(user_text)
                await send_event("status", {
                    "text": "Message queued — agent will read it after the current step."
                })
                return None

            logger.info(f"[ws] User: {user_text[:80]!r}")
            context_note, agent_messages = await build_context(history, user_text)
            if context_note:
                logger.info(
                    f"[ws] Context note: {len(context_note)} chars "
                    f"from {len(history)} history entries."
                )
            return (user_text, context_note, agent_messages)

        elif msg_type == "plan_response":
            # Phase 3e — user approved or rejected the plan card
            plan_id      = raw.get("plan_id")
            approved     = bool(raw.get("approved", False))
            edited_steps = raw.get("edited_steps", None)  # list[dict] or null
            if plan_id in pending_plans:
                pending_plans[plan_id]["result"] = {
                    "approved":     approved,
                    "edited_steps": edited_steps,
                }
                pending_plans[plan_id]["event"].set()
                logger.info(
                    f"[ws] Plan '{plan_id}': "
                    f"{'approved' if approved else 'rejected'}"
                    f"{', edited' if edited_steps else ''}"
                )
            else:
                logger.warning(f"[ws] Unknown plan_id: {plan_id!r}")

        elif msg_type == "stop_task":
            logger.info("[ws] stop_task received.")
            task_runner.cancel()

        elif msg_type == "confirm":
            confirmation_id = raw.get("confirmation_id")
            approved        = bool(raw.get("approved", False))
            if confirmation_id in pending_confirmations:
                pending_confirmations[confirmation_id]["result"] = approved
                pending_confirmations[confirmation_id]["event"].set()
                logger.info(
                    f"[ws] Confirmation '{confirmation_id}': "
                    f"{'approved' if approved else 'denied'}"
                )
            else:
                logger.warning(f"[ws] Unknown confirmation_id: {confirmation_id!r}")

        elif msg_type == "clear":
            history.clear()
            save_history(history)
            clear_vectors()
            await send_event("cleared", {"text": "Conversation history cleared."})
            logger.info("[ws] History and vector store cleared by user.")

        elif msg_type == "set_optimizer":
            enabled = bool(raw.get("data", {}).get("enabled", True))
            agent.use_prompt_optimizer = enabled
            logger.info(f"[ws] Prompt optimizer set to: {enabled}")
            await send_event("optimizer_status", {"enabled": enabled})

        elif msg_type == "set_local_mode":
            enabled          = bool(raw.get("data", {}).get("enabled", False))
            agent.local_mode = enabled
            logger.info(f"[ws] Local mode set to: {enabled}")
            await send_event("local_mode_status", {"enabled": enabled})

        elif msg_type == "set_model":
            model = raw.get("data", {}).get("model", "").strip()
            if model:
                agent.primary_model = model
                config.setdefault("llm", {})["primary"] = model
                logger.info(f"[ws] Primary model set to: {model}")
                await send_event("model_status", {"model": model})

        elif msg_type == "set_local_agent_model":
            model = raw.get("data", {}).get("model", "").strip()
            if model:
                agent.local_agent_model = model
                config.setdefault("llm", {})["local_agent"] = model
                logger.info(f"[ws] Local agent model set to: {model}")
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
                    logger.warning(f"[ws] set_config failed for {key!r}: {e}")
                    await send_event("error", {"text": f"Could not apply config {key}: {e}"})

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
                    pending_plans=pending_plans,        # Phase 3e
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
        # Unload model from RAM on disconnect — it will reload automatically
        # on the next local LLM call when a new session starts.
        await unload_model(agent.local_model, agent.ollama_url)
    except Exception as e:
        logger.exception("[ws] Unhandled error in WebSocket handler")
        try:
            await websocket.send_json({"type": "error", "data": {"text": str(e)}})
        except Exception:
            pass
