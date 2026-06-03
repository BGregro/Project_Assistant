"""
main.py  —  FastAPI Application Entry Point

Responsibilities:
  - Load config and register tools at startup.
  - Serve the frontend (static files + index.html).
  - Expose a WebSocket endpoint at /ws that drives the full agent session.
  - Handle the permission confirmation round-trip from the frontend.
  - Expose a /status endpoint for the frontend to poll Ollama/API availability.

The WebSocket is the only "API" the frontend uses — everything (messages, tool events,
confirmations, status updates) goes over the same connection as typed JSON events.
"""

import json
import logging
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Load .env from the project root (one level above backend/)
load_dotenv(Path(__file__).parent.parent / ".env")

# Add backend/ to sys.path so imports work when running from any directory
sys.path.insert(0, str(Path(__file__).parent))

from agent_core import AgentCore
from tools.filesystem import register_all as register_filesystem_tools
from tools.local_llm import is_ollama_available

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
    """Load config.json from the project root. Fail loudly if missing."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.json not found at {CONFIG_PATH}. Copy and fill it in.")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)

config = load_config()

# ---------------------------------------------------------------------------
# Tool registration (happens once at import time)
# ---------------------------------------------------------------------------

register_filesystem_tools()
logger.info(f"[startup] Registered tools: filesystem (read_file, write_file, list_directory)")

# ---------------------------------------------------------------------------
# Agent instance (shared across all WebSocket sessions)
# ---------------------------------------------------------------------------

agent = AgentCore(config)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Personal AI Agent",
    description="Phase 1 — FastAPI backend with Claude API and Ollama local tier",
    version="1.0.0",
)

# Serve frontend static files (CSS, JS) at /static
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the chat UI at the root URL."""
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/status")
async def status():
    """
    Health check endpoint polled by the frontend on load to show connection indicators.
    Returns availability of Claude API key and Ollama.
    """
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    ollama_ok = await is_ollama_available(config.get("ollama_base_url", "http://localhost:11434"))
    return JSONResponse({
        "claude_api": has_api_key,
        "ollama": ollama_ok,
        "use_prompt_optimizer": config.get("use_prompt_optimizer", True),
        "local_fallback": config.get("local_fallback", True),
        "models": config.get("llm", {}),
    })


# ---------------------------------------------------------------------------
# WebSocket endpoint  —  the core communication channel
# ---------------------------------------------------------------------------

# Stores pending permission confirmations: confirmation_id -> {event, result}
# Defined at module level so the WebSocket handler and agent_core share the same dict.
pending_confirmations: dict = {}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Single persistent WebSocket connection per browser tab.

    Message protocol (JSON):

      Client → Server:
        { "type": "message",  "text": "..." }           User chat message
        { "type": "confirm",  "confirmation_id": "...",  User permission response
                              "approved": true/false }
        { "type": "clear" }                              Clear conversation history

      Server → Client:
        { "type": "status",          "data": {"text": "..."} }    Processing status
        { "type": "prompt_optimized","data": {"original","optimized"} | null }
        { "type": "tool_call",       "data": {"tool","input"} }
        { "type": "tool_result",     "data": {"tool","success","result"} }
        { "type": "tool_denied",     "data": {"tool"} }
        { "type": "confirm_required","data": {"confirmation_id","tool","input","message"} }
        { "type": "message",         "data": {"text","source":"claude"|"local"} }
        { "type": "error",           "data": {"text"} }
        { "type": "cleared",         "data": {"text"} }
    """
    await websocket.accept()
    logger.info("[ws] New WebSocket connection.")

    # Conversation history for this session — grows with each turn.
    # We keep the ORIGINAL user messages in history (not the optimized versions)
    # so the conversation reads naturally.
    history: list[dict] = []

    async def send_event(event_type: str, data) -> None:
        """Helper: send a typed JSON event to the frontend."""
        await websocket.send_json({"type": event_type, "data": data})

    try:
        while True:
            raw = await websocket.receive_json()
            msg_type = raw.get("type")

            # ---- Incoming chat message ----
            if msg_type == "message":
                user_text = raw.get("text", "").strip()
                if not user_text:
                    continue

                logger.info(f"[ws] User: {user_text[:80]!r}")

                # Build history EXCLUDING the current message — agent_core appends it
                assistant_reply = await agent.run(
                    user_message=user_text,
                    history=history,
                    send_event=send_event,
                    pending_confirmations=pending_confirmations,
                )

                # Update history with the original (non-optimized) user text
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": assistant_reply})

                # Keep history bounded: drop oldest turns beyond the configured limit.
                # × 2 because each turn = 1 user + 1 assistant entry.
                max_turns = config.get("context", {}).get("max_history_turns", 20)
                if len(history) > max_turns * 2:
                    history = history[-(max_turns * 2):]

            # ---- Confirmation response from user ----
            elif msg_type == "confirm":
                confirmation_id = raw.get("confirmation_id")
                approved = bool(raw.get("approved", False))

                if confirmation_id in pending_confirmations:
                    pending_confirmations[confirmation_id]["result"] = approved
                    # Unblock the asyncio.Event that agent_core is waiting on
                    pending_confirmations[confirmation_id]["event"].set()
                    logger.info(f"[ws] Confirmation '{confirmation_id}': {'approved' if approved else 'denied'}")
                else:
                    logger.warning(f"[ws] Unknown confirmation_id: {confirmation_id!r}")

            # ---- Clear conversation history ----
            elif msg_type == "clear":
                history.clear()
                await send_event("cleared", {"text": "Conversation history cleared."})
                logger.info("[ws] History cleared by user.")

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
