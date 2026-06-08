"""
long_term.py  —  Phase 3f: Long-Term Task Memory

Persistent store for task outcomes, factual knowledge, and research findings
across sessions.  Entirely separate from:
  - history.json  (conversation log — short-lived, session-oriented)
  - ChromaDB      (semantic search over conversation turns)

This store is optimised for:
  - Recalling what the agent tried before and whether it worked
  - Surfacing relevant research findings at the start of a new task
  - Storing key facts the agent discovered (URLs, versions, config values, etc.)

Data lives in  memory/long_term.json  (relative to the project root).

Atomic writes: every save() writes to a .tmp file first, then renames, so a
crash mid-write never corrupts the store.
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# This file lives at  backend/memory/long_term.py
# The data file lives at  memory/long_term.json  (project root / memory /)
_STORE_FILE = Path(__file__).parent.parent.parent / "memory" / "long_term.json"

# Maximum number of entries kept per collection.
_MAX_TASKS    = 100
_MAX_RESEARCH = 200


# ---------------------------------------------------------------------------
# Core I/O
# ---------------------------------------------------------------------------

def load() -> dict:
    """
    Read long_term.json and return its contents as a dict.

    Returns the canonical empty structure if the file does not exist, is empty,
    or is corrupt so callers never need to handle missing keys.
    """
    empty: dict = {"tasks": [], "facts": [], "research": []}
    if not _STORE_FILE.exists():
        return empty
    try:
        with open(_STORE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        # Ensure all three top-level keys exist (forward-compat)
        for key in empty:
            data.setdefault(key, [])
        return data
    except Exception as e:
        logger.warning(f"[long_term] Could not load {_STORE_FILE}: {e} — returning empty store.")
        return empty


def save(data: dict) -> None:
    """
    Write data to long_term.json using an atomic rename so partial writes
    never corrupt the store.
    """
    _STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STORE_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # Atomic rename (on Windows this may overwrite; on POSIX it's truly atomic)
        os.replace(tmp, _STORE_FILE)
    except Exception as e:
        logger.error(f"[long_term] Failed to save store: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Task logging
# ---------------------------------------------------------------------------

def log_task(
    goal: str,
    outcome: str,
    summary: str,
    tools_used: list,
    duration_seconds: int,
) -> None:
    """
    Append a completed (or failed/cancelled) task to the tasks list.

    Args:
        goal:             The original user message / task description.
        outcome:          "success", "failure", or "partial".
        summary:          A short human-readable summary of what happened.
        tools_used:       List of tool names that were called during the task.
        duration_seconds: Wall-clock seconds the task ran for.
    """
    data = load()
    entry = {
        "id":               str(uuid4()),
        "timestamp":        _now_iso(),
        "goal":             goal,
        "outcome":          outcome,
        "summary":          summary,
        "tools_used":       list(tools_used),
        "duration_seconds": int(duration_seconds),
    }
    data["tasks"].append(entry)
    # Keep only the most recent _MAX_TASKS entries (trim oldest)
    if len(data["tasks"]) > _MAX_TASKS:
        data["tasks"] = data["tasks"][-_MAX_TASKS:]
    save(data)
    logger.info(f"[long_term] Task logged: outcome={outcome!r}, goal={goal[:60]!r}")


# ---------------------------------------------------------------------------
# Fact logging
# ---------------------------------------------------------------------------

def log_fact(
    key: str,
    value: str,
    source: str,
    expires_days: int | None = None,
) -> None:
    """
    Store or update a key-value fact.

    If a fact with the same key already exists it is updated in place rather
    than duplicated, keeping the store tidy.

    Args:
        key:          Short identifier for this fact (e.g. "python_version").
        value:        The fact content.
        source:       Where this fact came from (tool name, URL, user statement).
        expires_days: If set, the fact expires this many days from now.
                      Pass None for facts that should persist indefinitely.
    """
    data = load()
    expires_iso: str | None = None
    if expires_days is not None:
        expires_dt = datetime.now(timezone.utc) + timedelta(days=expires_days)
        expires_iso = expires_dt.isoformat()

    # Check for existing entry with the same key and update it
    for existing in data["facts"]:
        if existing.get("key") == key:
            existing["value"]     = value
            existing["source"]    = source
            existing["timestamp"] = _now_iso()
            existing["expires"]   = expires_iso
            save(data)
            logger.info(f"[long_term] Fact updated: key={key!r}")
            return

    # No existing entry — append new
    entry = {
        "id":        str(uuid4()),
        "timestamp": _now_iso(),
        "key":       key,
        "value":     value,
        "source":    source,
        "expires":   expires_iso,
    }
    data["facts"].append(entry)
    save(data)
    logger.info(f"[long_term] Fact stored: key={key!r}")


# ---------------------------------------------------------------------------
# Research logging
# ---------------------------------------------------------------------------

def log_research(
    topic: str,
    findings: str,
    sources: list[str],
    relevance_score: float = 1.0,
) -> None:
    """
    Append a research entry to the research list.

    Args:
        topic:           Short description of what was researched.
        findings:        The key findings / conclusions.
        sources:         List of URLs or tool names that produced the findings.
        relevance_score: 0.0–1.0 quality/relevance signal (default 1.0).
    """
    data = load()
    entry = {
        "id":              str(uuid4()),
        "timestamp":       _now_iso(),
        "topic":           topic,
        "findings":        findings,
        "sources":         list(sources),
        "relevance_score": float(relevance_score),
    }
    data["research"].append(entry)
    # Keep only the most recent _MAX_RESEARCH entries
    if len(data["research"]) > _MAX_RESEARCH:
        data["research"] = data["research"][-_MAX_RESEARCH:]
    save(data)
    logger.info(f"[long_term] Research logged: topic={topic[:60]!r}")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def query_tasks(keyword: str = "", last_n: int = 10) -> list:
    """
    Return up to last_n tasks, optionally filtered by keyword.

    Keyword matching is case-insensitive and checks both goal and summary.
    If keyword is empty, returns the most recent last_n tasks regardless.
    """
    tasks = load()["tasks"]
    if keyword:
        kw = keyword.lower()
        tasks = [
            t for t in tasks
            if kw in t.get("goal", "").lower() or kw in t.get("summary", "").lower()
        ]
    return tasks[-last_n:]


def query_facts(key: str = "") -> list:
    """
    Return all non-expired facts, optionally filtered by key substring.

    Expired facts (where expires < now) are silently excluded.
    """
    now = datetime.now(timezone.utc)
    facts = []
    for f in load()["facts"]:
        expires_str = f.get("expires")
        if expires_str:
            try:
                expires_dt = datetime.fromisoformat(expires_str)
                if expires_dt < now:
                    continue  # expired — skip
            except ValueError:
                pass  # malformed date — keep the fact
        facts.append(f)

    if key:
        k = key.lower()
        facts = [f for f in facts if k in f.get("key", "").lower()]

    return facts


def query_research(topic: str = "", last_n: int = 10) -> list:
    """
    Return up to last_n research entries, optionally filtered by topic substring.
    """
    research = load()["research"]
    if topic:
        t = topic.lower()
        research = [r for r in research if t in r.get("topic", "").lower()]
    return research[-last_n:]


# ---------------------------------------------------------------------------
# Context summary (called before every task run)
# ---------------------------------------------------------------------------

def get_context_summary(current_goal: str) -> str:
    """
    Build a short summary (≤ 500 chars) of relevant past context for the
    given goal.  This is injected into the system prompt so Claude is
    aware of what has already been tried or discovered — without needing
    an extra tool call.

    Returns an empty string if nothing relevant is found.
    """
    parts: list[str] = []

    # ── Recent tasks with keyword overlap ─────────────────────────────
    # Extract rough keywords from the goal (words > 4 chars)
    goal_words = {w.lower() for w in current_goal.split() if len(w) > 4}
    if goal_words:
        # Try each keyword; collect unique matching tasks
        seen_ids: set = set()
        matching_tasks: list = []
        for word in list(goal_words)[:5]:  # cap at 5 keywords
            for t in query_tasks(keyword=word, last_n=5):
                if t["id"] not in seen_ids:
                    seen_ids.add(t["id"])
                    matching_tasks.append(t)
        if matching_tasks:
            task_lines = []
            for t in matching_tasks[-3:]:  # at most 3 task summaries
                task_lines.append(
                    f"[{t['outcome']}] {t['goal'][:60]}: {t['summary'][:80]}"
                )
            parts.append("Past tasks: " + " | ".join(task_lines))

    # ── Relevant facts (all non-expired, filtered by goal keywords) ────
    relevant_facts: list = []
    for word in list(goal_words)[:5]:
        for f in query_facts(key=word):
            if f not in relevant_facts:
                relevant_facts.append(f)
    if relevant_facts:
        fact_lines = [
            f"{f['key']}={f['value'][:60]}" for f in relevant_facts[:3]
        ]
        parts.append("Facts: " + ", ".join(fact_lines))

    # ── Recent research on related topics ──────────────────────────────
    if goal_words:
        seen_research_ids: set = set()
        matching_research: list = []
        for word in list(goal_words)[:3]:
            for r in query_research(topic=word, last_n=3):
                if r["id"] not in seen_research_ids:
                    seen_research_ids.add(r["id"])
                    matching_research.append(r)
        if matching_research:
            research_lines = []
            for r in matching_research[-2:]:  # at most 2 research snippets
                research_lines.append(
                    f"{r['topic'][:40]}: {r['findings'][:80]}"
                )
            parts.append("Research: " + " | ".join(research_lines))

    if not parts:
        return ""

    summary = "Past context: " + " | ".join(parts)
    # Hard cap at 500 chars to keep the system prompt lean
    if len(summary) > 500:
        summary = summary[:497] + "…"
    return summary


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
