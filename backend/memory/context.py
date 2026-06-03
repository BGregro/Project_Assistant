"""
memory/context.py  —  Persistent Conversation Memory

Saves and loads conversation history so sessions survive process restarts.

Storage location:
    <project_root>/memory/history.json

    The project root is three levels up from this file:
        backend/memory/context.py  →  backend/memory  →  backend  →  project root

On-disk format (list of objects):
    [
      { "timestamp": "2025-01-15T10:30:00+00:00", "role": "user",      "content": "..." },
      { "timestamp": "2025-01-15T10:30:01+00:00", "role": "assistant", "content": "..." },
      ...
    ]

Public API:
    load_history()                              → list[dict]
    save_history(history: list)                 → None
    trim_history(history: list, max_turns: int) → list[dict]
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# __file__ = backend/memory/context.py
# .parent         = backend/memory/
# .parent.parent  = backend/
# .parent.parent.parent = project root (where config.json lives)
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# The data directory is project_root/memory/ (separate from the Python package
# backend/memory/ — Python package is code, data dir is runtime state)
MEMORY_DIR = _PROJECT_ROOT / "memory"
HISTORY_FILE = MEMORY_DIR / "history.json"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_history() -> list:
    """
    Load the full conversation history from disk.

    Returns:
        A list of dicts with keys "timestamp", "role", "content".
        Returns an empty list if the file doesn't exist or is malformed
        (never raises — a missing/corrupt history file is not fatal).
    """
    if not HISTORY_FILE.exists():
        logger.info("[memory] No history file found — starting with empty history.")
        return []

    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)

        # Basic sanity check: must be a list
        if not isinstance(data, list):
            logger.warning(
                "[memory] history.json is not a JSON array — ignoring corrupt file."
            )
            return []

        # Drop any malformed entries (must have role + content at minimum)
        valid = [
            e for e in data
            if isinstance(e, dict) and "role" in e and "content" in e
        ]
        if len(valid) != len(data):
            logger.warning(
                f"[memory] Dropped {len(data) - len(valid)} malformed entries "
                "from history.json."
            )

        logger.info(f"[memory] Loaded {len(valid)} history entries from disk.")
        return valid

    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[memory] Could not read history.json: {e}")
        return []


def save_history(history: list) -> None:
    """
    Persist the conversation history to disk.

    Each entry should already contain "role" and "content"; a "timestamp"
    key is expected but not required. The memory/ directory is created
    automatically if it doesn't exist.

    This is a best-effort write — a failure logs an error but does NOT
    raise, because a save failure should never crash the agent.
    """
    try:
        # Create project_root/memory/ if it doesn't exist yet
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        logger.debug(f"[memory] Saved {len(history)} entries to history.json.")

    except OSError as e:
        logger.error(f"[memory] Failed to save history.json: {e}")


def trim_history(history: list, max_turns: int) -> list:
    """
    Return a copy of history containing only the most recent `max_turns` turns.

    One "turn" = one user message + one assistant reply = 2 list entries.
    Entries are always trimmed from the front so the most recent context
    is preserved.

    Args:
        history:   Full history list (may include timestamp/role/content dicts).
        max_turns: Maximum number of complete turns to keep.

    Returns:
        Trimmed list. If history is already within limits, the original
        list is returned unchanged (no copy).
    """
    max_entries = max_turns * 2  # Each turn = user entry + assistant entry

    if len(history) <= max_entries:
        return history  # Already within bounds — nothing to trim

    trimmed = history[-max_entries:]
    logger.debug(
        f"[memory] Trimmed history from {len(history)} → {len(trimmed)} entries "
        f"(max_turns={max_turns})."
    )
    return trimmed
