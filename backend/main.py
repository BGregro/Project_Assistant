"""
main.py  —  FastAPI Application Entry Point

Responsibilities:
  - Load config and register tools at startup.
  - Serve the frontend (static files + index.html).
  - Expose a WebSocket endpoint at /ws that drives the full agent session.
  - Handle the permission confirmation round-trip from the frontend.
  - Expose a /status endpoint for the frontend to poll Ollama/API availability.

Context assembly (build_context):
  Before every agent.run() call, build_context() constructs a minimal but
  rich context from three sources:
    1. Recent turns — last N turns verbatim (always sent, always accurate)
    2. Summary note — Ollama compresses old turns beyond the window
    3. Retrieved turns — ChromaDB semantic search finds relevant past turns
  The combined context_note is injected once into the system prompt (not
  repeated in messages[]), making it O(1) in token cost regardless of N.

Changes vs Phase 1:
  - Persistent memory (load/save history.json on every turn)
  - Runtime optimizer toggle (set_optimizer WS message)
  - Tiered context assembly via build_context()
  - Background embedding of each turn into ChromaDB
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
from agent_tools.local_llm import is_ollama_available, summarize_history

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
logger.info("[startup] Registered tools: filesystem (read_file, write_file, list_directory)")

# ---------------------------------------------------------------------------
# Agent instance (shared across all WebSocket sessions)
# ---------------------------------------------------------------------------

agent = AgentCore(config)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Personal AI Agent",
    description="Phase 1 — FastAPI backend with Claude API, Ollama, and semantic memory",
    version="1.2.0",
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
        "claude_api":          has_api_key,
        "ollama":              ollama_ok,
        "use_prompt_optimizer": agent.use_prompt_optimizer,
        "local_fallback":      config.get("local_fallback", True),
        "models":              config.get("llm", {}),
    })


# ---------------------------------------------------------------------------
# Context assembly — called before every agent.run()
# ---------------------------------------------------------------------------

async def build_context(
    history: list[dict],
    current_message: str,
) -> tuple[str, list[dict]]:
    """
    Assemble the minimal context Claude needs for the current turn.

    Returns:
        context_note  — string to append to the system prompt (may be empty)
        agent_messages — [{role, content}] list for the messages[] parameter,
                         containing only the recent verbatim window (timestamps stripped)

    Token budget design:
      recent_turns × avg_turn_tokens  +  ~150 (summary)  +  ~200 (retrieved)
      e.g. 4 turns × 300 tokens + 350 = ~1550 tokens total context
      vs. 20 turns × 300 = 6000 tokens without tiering  →  ~75% reduction
    """
    ctx_cfg  = config.get("context", {})
    emb_cfg  = config.get("embeddings", {})

    recent_turns       = ctx_cfg.get("recent_turns", 4)        # turns kept verbatim
    summary_threshold  = ctx_cfg.get("summary_threshold", 4)   # min OLD turns before summarizing
    retrieval_n        = ctx_cfg.get("retrieval_n", 3)          # how many turns to retrieve
    similarity_cutoff  = ctx_cfg.get("similarity_cutoff", 0.45) # cosine distance ceiling
    embed_enabled      = emb_cfg.get("enabled", True)
    embed_model        = emb_cfg.get("model", "nomic-embed-text")

    # --- Split history into recent (verbatim) and old (to compress) ---
    cutoff = recent_turns * 2
    recent = history[-cutoff:] if len(history) >= cutoff else history
    old    = history[:-cutoff]  if len(history) >  cutoff else []

    notes = []

    # ---- Source 1: Summary of old turns --------------------------------
    # Only bother if there are enough old turns to justify a local LLM call.
    if len(old) >= summary_threshold * 2:   # threshold is in turns, old is in entries
        await_note = True
        summary = await summarize_history(
            turns=old,
            model=agent.local_model,
            base_url=agent.ollama_url,
        )
        if summary:
            notes.append(f"Summary of earlier conversation:\n{summary}")
            logger.debug(f"[ctx] Summary: {len(old)} old entries → {len(summary)} chars")

    # ---- Source 2: Semantic retrieval from ChromaDB --------------------
    # Skip if embeddings are disabled in config, or if there's nothing to retrieve.
    if embed_enabled and current_message:
        # Timestamps of the turns already in the verbatim window — used to
        # exclude duplicates (don't show a turn in both recent AND retrieved).
        recent_timestamps = {e.get("timestamp", "") for e in recent if e.get("timestamp")}

        retrieved = await search_similar(
            query=current_message,
            n_results=retrieval_n,
            model=embed_model,
            base_url=agent.ollama_url,
        )

        # Filter: must be similar enough AND not already in the verbatim window
        relevant = [
            r for r in retrieved
            if r["distance"] < similarity_cutoff
            and r["timestamp"] not in recent_timestamps
        ]

        if relevant:
            lines = []
            for r in relevant:
                # Truncate to keep token cost bounded
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
# Background embedding — fire-and-forget after each turn
# ---------------------------------------------------------------------------

async def _embed_turn_bg(
    timestamp: str,
    user_content: str,
    assistant_content: str,
) -> None:
    """
    Embed a completed turn and store it in ChromaDB.
    Called via asyncio.create_task() so it never delays the agent response.
    All exceptions are caught here — a failed embed must not bubble up.
    """
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
    """
    Single persistent WebSocket connection per browser tab.

    Client → Server messages:
        { "type": "message",      "text": "..." }
        { "type": "confirm",      "confirmation_id": "...", "approved": bool }
        { "type": "clear" }
        { "type": "set_optimizer","data": {"enabled": bool} }

    Server → Client events:
        status | prompt_optimized | tool_call | tool_result | tool_denied
        confirm_required | message | error | cleared | optimizer_status
    """
    await websocket.accept()
    logger.info("[ws] New WebSocket connection.")

    # Pre-populate from persistent storage so conversations survive restarts.
    # Format: [{"timestamp": ISO, "role": "user|assistant", "content": "..."}]
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

                # Assemble tiered context:
                #   context_note    → injected into system prompt (old summary + retrieved)
                #   agent_messages  → recent verbatim turns (timestamps stripped)
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

                # Stamp and persist the new turn
                ts = datetime.now(timezone.utc).isoformat()
                history.append({"timestamp": ts, "role": "user",      "content": user_text})
                history.append({"timestamp": ts, "role": "assistant", "content": assistant_reply})

                max_turns = config.get("context", {}).get("max_history_turns", 20)
                history   = trim_history(history, max_turns)
                save_history(history)

                # Embed the new turn in the background (doesn't delay the response)
                asyncio.create_task(_embed_turn_bg(ts, user_text, assistant_reply))

            # ---- Confirmation response from user ----
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

            # ---- Clear conversation history ----
            elif msg_type == "clear":
                history.clear()
                save_history(history)   # Wipe history.json
                clear_vectors()         # Wipe ChromaDB vector store
                await send_event("cleared", {"text": "Conversation history cleared."})
                logger.info("[ws] History and vector store cleared by user.")

            # ---- Runtime optimizer toggle ----
            elif msg_type == "set_optimizer":
                enabled                  = bool(raw.get("data", {}).get("enabled", True))
                agent.use_prompt_optimizer = enabled
                logger.info(f"[ws] Prompt optimizer set to: {enabled}")
                await send_event("optimizer_status", {"enabled": enabled})

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
