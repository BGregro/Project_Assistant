"""
main.py  —  FastAPI Application Entry Point

WebSocket message types (client → server):
    { "type": "message",               "text": "…" }
    { "type": "confirm",               "confirmation_id": "…", "approved": bool }
    { "type": "clear" }
    { "type": "set_optimizer",         "data": {"enabled": bool} }
    { "type": "set_local_mode",        "data": {"enabled": bool} }
    { "type": "set_model",             "data": {"model": "…"} }
    { "type": "set_local_agent_model", "data": {"model": "…"} }
    { "type": "set_config",            "data": {"key": "context.recent_turns", "value": N} }

WebSocket event types (server → client):
    status | prompt_optimized | tool_call | tool_result | tool_denied
    confirm_required | message | error | cleared | optimizer_status
    tree_update | local_mode_status | model_status | local_agent_model_status | config_ack
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
from memory.context import load_history, save_history, trim_history
from memory.embeddings import store_turn, search_similar, clear_all as clear_vectors
from agent_tools.filesystem import register_all as register_filesystem_tools
from agent_tools.capabilities import register_capabilities_tools
from agent_tools.local_llm import is_ollama_available, summarize_history
from agent_tools.web import register_web_tools
from agent_tools.system_info import register_system_tools
from agent_tools.file_analysis import register_file_analysis_tools

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
logger.info(
    "[startup] Registered tools: filesystem (read_file, write_file, list_directory), "
    "capabilities (list_capabilities), "
    "web (search_web, fetch_page), "
    "system (get_system_info), "
    "file_analysis (analyze_file)"
)

# ---------------------------------------------------------------------------
# Agent instance
# ---------------------------------------------------------------------------

agent = AgentCore(config)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Personal AI Agent",
    description="Phase 2 — settings panel, configurable models and context",
    version="1.4.0",
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/status")
async def status():
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    ollama_ok   = await is_ollama_available(config.get("ollama_base_url", "http://localhost:11434"))
    return JSONResponse({
        "claude_api":            has_api_key,
        "ollama":                ollama_ok,
        "use_prompt_optimizer":  agent.use_prompt_optimizer,
        "local_fallback":        config.get("local_fallback", True),
        "local_mode":            agent.local_mode,
        "primary_model":         agent.primary_model,
        "local_agent_model":     agent.local_agent_model,
        "models":                config.get("llm", {}),
        "context":               config.get("context", {}),
        "embeddings":            config.get("embeddings", {}),
        "local_agent_timeout":   agent.local_agent_timeout,
        "tree_root":             config.get("tree_root", "."),
    })


# ---------------------------------------------------------------------------
# set_config helper — apply a dot-notation key to the live config dict
# and mirror any change to agent attributes that cache config values.
# ---------------------------------------------------------------------------

def _apply_config(key: str, value) -> None:
    """
    Apply a dot-notation config key (e.g. "context.recent_turns") to the live
    `config` dict, then mirror the change to any `agent.*` attribute that reads
    from config so in-flight requests see the new value immediately.

    Supported keys and their agent mirrors:
        context.recent_turns          — config only (read by build_context each call)
        context.summary_threshold     — config only
        context.retrieval_n           — config only
        context.similarity_cutoff     — config only
        context.max_history_turns     — config only (read by WS handler each turn)
        context.max_iterations_per_turn → agent.max_iterations
        embeddings.enabled            — config only (read by _embed_turn_bg each call)
        tree_root                     — config only (read by filesystem.py each call)
        local_agent_timeout           → agent.local_agent_timeout
    """
    parts = key.split(".")
    node  = config

    # Navigate to the parent of the leaf key
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]

    leaf = parts[-1]
    node[leaf] = value
    logger.info(f"[config] Set {key} = {value!r}")

    # Mirror to agent attributes where applicable
    if key == "context.max_iterations_per_turn":
        agent.max_iterations = int(value)
    elif key == "local_agent_timeout":
        agent.local_agent_timeout = float(value)


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[ws] New WebSocket connection.")

    history: list[dict] = load_history()
    if history:
        logger.info(f"[ws] Resumed session — {len(history)} history entries loaded.")

    async def send_event(event_type: str, data) -> None:
        await websocket.send_json({"type": event_type, "data": data})

    try:
        while True:
            raw      = await websocket.receive_json()
            msg_type = raw.get("type")

            # ---- Incoming chat message ----
            if msg_type == "message":
                user_text = raw.get("text", "").strip()
                if not user_text:
                    continue

                logger.info(f"[ws] User: {user_text[:80]!r}")

                context_note, agent_messages = await build_context(history, user_text)

                if context_note:
                    logger.info(
                        f"[ws] Context note: {len(context_note)} chars "
                        f"from {len(history)} history entries."
                    )

                assistant_reply = await agent.run(
                    user_message=user_text,
                    history=agent_messages,
                    send_event=send_event,
                    pending_confirmations=pending_confirmations,
                    context_summary=context_note,
                )

                ts = datetime.now(timezone.utc).isoformat()
                history.append({"timestamp": ts, "role": "user",      "content": user_text})
                history.append({"timestamp": ts, "role": "assistant", "content": assistant_reply})

                max_turns = config.get("context", {}).get("max_history_turns", 20)
                history   = trim_history(history, max_turns)
                save_history(history)

                asyncio.create_task(_embed_turn_bg(ts, user_text, assistant_reply))

            # ---- Confirmation ----
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

            # ---- Clear history ----
            elif msg_type == "clear":
                history.clear()
                save_history(history)
                clear_vectors()
                await send_event("cleared", {"text": "Conversation history cleared."})
                logger.info("[ws] History and vector store cleared by user.")

            # ---- Optimizer toggle ----
            elif msg_type == "set_optimizer":
                enabled = bool(raw.get("data", {}).get("enabled", True))
                agent.use_prompt_optimizer = enabled
                logger.info(f"[ws] Prompt optimizer set to: {enabled}")
                await send_event("optimizer_status", {"enabled": enabled})

            # ---- Local mode toggle ----
            elif msg_type == "set_local_mode":
                enabled          = bool(raw.get("data", {}).get("enabled", False))
                agent.local_mode = enabled
                logger.info(f"[ws] Local mode set to: {enabled}")
                await send_event("local_mode_status", {"enabled": enabled})

            # ---- Primary Claude model change ----
            elif msg_type == "set_model":
                model = raw.get("data", {}).get("model", "").strip()
                if model:
                    agent.primary_model = model
                    # Also update the live config so /status reflects it
                    config.setdefault("llm", {})["primary"] = model
                    logger.info(f"[ws] Primary model set to: {model}")
                    await send_event("model_status", {"model": model})

            # ---- Local agent model change ----
            elif msg_type == "set_local_agent_model":
                model = raw.get("data", {}).get("model", "").strip()
                if model:
                    agent.local_agent_model = model
                    config.setdefault("llm", {})["local_agent"] = model
                    logger.info(f"[ws] Local agent model set to: {model}")
                    await send_event("local_agent_model_status", {"model": model})

            # ---- Generic config key update ----
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

    except WebSocketDisconnect:
        logger.info("[ws] Client disconnected.")
    except Exception as e:
        logger.exception("[ws] Unhandled error in WebSocket handler")
        try:
            await websocket.send_json({"type": "error", "data": {"text": str(e)}})
        except Exception:
            pass
