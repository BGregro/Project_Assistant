"""
long_term.py  —  Phase 3f: Long-Term Task Memory
             —  Improvement 1: Semantic research retrieval via ChromaDB

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

Improvement 1 additions:
  - _get_research_collection(): returns a "long_term_research" ChromaDB collection
    backed by the same memory/vectors/ directory as embeddings.py.
  - log_research() now also upserts into ChromaDB after saving to JSON.
  - semantic_query_research(query, n_results): finds research entries by semantic
    similarity using nomic-embed-text embeddings via Ollama. Falls back to
    substring query_research() if ChromaDB is unavailable or empty.
  - get_context_summary() now calls semantic_query_research() instead of
    query_research() for the research section, so "passive income" can match
    "make money online" and similar paraphrases.
  - load() now contains a migration step: on first call, if the ChromaDB
    collection is empty but JSON has research entries, all entries are re-indexed.
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import chromadb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# This file lives at  backend/memory/long_term.py
# The data file lives at  memory/long_term.json  (project root / memory /)
_STORE_FILE = Path(__file__).parent.parent.parent / "memory" / "long_term.json"

# ChromaDB vector directory — same root as embeddings.py uses
_VECTOR_DIR = Path(__file__).parent.parent.parent / "memory" / "vectors"

# Maximum number of entries kept per collection.
_MAX_TASKS    = 100
_MAX_RESEARCH = 200

# Ollama embedding constants (same as embeddings.py)
_EMBED_MODEL   = "nomic-embed-text"
_EMBED_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Config helper — read ollama base URL from config.json lazily
# ---------------------------------------------------------------------------

def _get_ollama_base_url() -> str:
    """
    Read ollama_base_url from config.json (project root).
    Falls back to the default localhost address if the file can't be read.
    """
    config_path = Path(__file__).parent.parent.parent / "config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("ollama_base_url", "http://localhost:11434")
    except Exception:
        return "http://localhost:11434"


# ---------------------------------------------------------------------------
# ChromaDB — lazy singleton for the research collection
# ---------------------------------------------------------------------------

_chroma_client: chromadb.PersistentClient | None = None
_research_collection: chromadb.Collection | None = None


def _get_research_collection() -> chromadb.Collection:
    """
    Return (or create on first call) a ChromaDB collection named
    "long_term_research".  Uses the same PersistentClient path as
    embeddings.py  (memory/vectors/).

    The collection uses cosine distance — standard for text embeddings.
    Raises on failure so callers can catch and fall back gracefully.
    """
    global _chroma_client, _research_collection
    if _research_collection is None:
        _VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        # Re-use the existing client if it was already created, otherwise create
        # a new one.  We don't share the global from embeddings.py to keep the
        # modules independent.
        _chroma_client = chromadb.PersistentClient(path=str(_VECTOR_DIR))
        _research_collection = _chroma_client.get_or_create_collection(
            name="long_term_research",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"[long_term] ChromaDB 'long_term_research' ready. "
            f"Stored entries: {_research_collection.count()}"
        )
    return _research_collection


# ---------------------------------------------------------------------------
# Ollama embedding — synchronous (used from both sync and async contexts)
# ---------------------------------------------------------------------------

def _embed_sync(text: str) -> list[float] | None:
    """
    POST to Ollama /api/embeddings (old endpoint, wider compat) and return the
    embedding vector.

    Uses httpx.Client (synchronous) so this can be called from both sync code
    (load-time migration) and async code without needing an event loop.

    Returns None on any failure — callers should fall back to substring search.

    Note: We use /api/embeddings (not /api/embed) for maximum Ollama version
    compatibility.  Both endpoints exist in recent Ollama but the older one
    is more widely available.
    """
    base_url = _get_ollama_base_url()
    try:
        with httpx.Client(timeout=_EMBED_TIMEOUT) as client:
            resp = client.post(
                f"{base_url}/api/embeddings",
                json={"model": _EMBED_MODEL, "prompt": text},
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding")
            if isinstance(embedding, list) and len(embedding) > 0:
                return embedding
            logger.warning("[long_term] Ollama embed returned unexpected shape.")
            return None
    except httpx.ConnectError:
        logger.warning("[long_term] Ollama offline — skipping embed.")
        return None
    except httpx.TimeoutException:
        logger.warning(f"[long_term] Embed timed out after {_EMBED_TIMEOUT}s.")
        return None
    except Exception as e:
        logger.warning(f"[long_term] Embed failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Core I/O
# ---------------------------------------------------------------------------

def load() -> dict:
    """
    Read long_term.json and return its contents as a dict.

    Returns the canonical empty structure if the file does not exist, is empty,
    or is corrupt so callers never need to handle missing keys.

    Migration step (Improvement 1):
    On first call, if the ChromaDB research collection is empty but the JSON
    research array is non-empty, all existing entries are re-indexed into
    ChromaDB so they become semantically searchable without manual action.
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
    except Exception as e:
        logger.warning(f"[long_term] Could not load {_STORE_FILE}: {e} — returning empty store.")
        return empty

    # ── ChromaDB migration (runs at most once per cold start) ──────────
    # We only attempt migration if the JSON has research data. Guarding with
    # a try/except means a broken ChromaDB never prevents the agent from loading.
    if data["research"]:
        try:
            col = _get_research_collection()
            if col.count() == 0:
                n = len(data["research"])
                logger.info(f"[long_term] Migrating {n} research entries to semantic index…")
                _bulk_upsert_research(data["research"], col)
        except Exception as e:
            logger.warning(f"[long_term] Migration skipped (ChromaDB unavailable): {e}")

    return data


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
# Internal ChromaDB helpers
# ---------------------------------------------------------------------------

def _upsert_research_entry(entry: dict, col: chromadb.Collection) -> None:
    """
    Embed a research entry and upsert it into the given ChromaDB collection.
    Combines topic + first 1000 chars of findings as the document text.
    Raises on ChromaDB errors — callers are responsible for handling.
    """
    topic    = entry.get("topic", "")
    findings = entry.get("findings", "")
    sources  = entry.get("sources", [])
    ts       = entry.get("timestamp", "")
    uid      = entry["id"]

    document  = f"{topic}\n\n{findings[:1000]}"
    embedding = _embed_sync(document)
    if embedding is None:
        # Ollama offline — skip this entry silently
        return

    col.upsert(
        ids=[uid],
        embeddings=[embedding],
        documents=[document],
        metadatas=[{
            "topic":     topic,
            "timestamp": ts,
            "sources":   json.dumps(sources),
        }],
    )


def _bulk_upsert_research(entries: list[dict], col: chromadb.Collection) -> None:
    """
    Upsert multiple research entries into the ChromaDB collection.
    Individual embed failures are logged and skipped so one bad entry
    doesn't abort the whole migration.
    """
    success = 0
    for entry in entries:
        try:
            _upsert_research_entry(entry, col)
            success += 1
        except Exception as e:
            logger.warning(f"[long_term] Failed to index entry {entry.get('id', '?')}: {e}")
    logger.info(f"[long_term] Migration complete: {success}/{len(entries)} entries indexed.")


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

    Also upserts into ChromaDB so the entry becomes semantically searchable.
    If the ChromaDB upsert fails, a warning is logged and the function
    continues — JSON is the source of truth.

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

    # ── Semantic index upsert (Improvement 1) ──────────────────────────
    # Do this AFTER the JSON save so a ChromaDB failure never loses the entry.
    try:
        col = _get_research_collection()
        _upsert_research_entry(entry, col)
        logger.debug(f"[long_term] Research entry {entry['id'][:8]} indexed in ChromaDB.")
    except Exception as e:
        logger.warning(
            f"[long_term] ChromaDB upsert failed for research entry '{topic[:40]}': {e} "
            f"(entry saved to JSON — will be re-indexed on next startup)"
        )


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

    This is the substring fallback — prefer semantic_query_research() for
    better recall across paraphrased topics.
    """
    research = load()["research"]
    if topic:
        t = topic.lower()
        research = [r for r in research if t in r.get("topic", "").lower()]
    return research[-last_n:]


def semantic_query_research(query: str, n_results: int = 3) -> list[dict]:
    """
    Find research entries most semantically similar to `query` using
    ChromaDB + nomic-embed-text embeddings.

    Flow:
      1. Embed `query` via Ollama (synchronous httpx call).
      2. Query the "long_term_research" ChromaDB collection.
      3. For each result id, look up the full entry in the JSON store
         (ChromaDB metadata is truncated; JSON has the full findings).
      4. Return the full entry dicts sorted by similarity (best first).

    Falls back to query_research(query, n_results) silently if:
      - Ollama is offline (embedding fails)
      - ChromaDB collection is empty
      - Any other error occurs

    This is a synchronous function — httpx.Client (not async) is used so it
    can be called from both sync functions (get_context_summary) and from
    async handlers without needing await.

    Args:
        query:     Natural-language query string.
        n_results: Maximum number of entries to return.

    Returns:
        List of full research entry dicts from JSON (or fallback list).
    """
    try:
        col = _get_research_collection()
        count = col.count()
        if count == 0:
            logger.debug("[long_term] semantic_query_research: collection empty, using fallback.")
            return query_research(query, n_results)

        embedding = _embed_sync(query)
        if embedding is None:
            # Ollama offline — fall back to substring search silently
            return query_research(query, n_results)

        actual_n = min(n_results, count)
        results = col.query(
            query_embeddings=[embedding],
            n_results=actual_n,
            include=["metadatas", "distances"],
        )

        # Build a lookup map from the JSON store for fast id → full entry access
        all_research = {entry["id"]: entry for entry in load()["research"]}

        output: list[dict] = []
        ids       = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for uid, dist in zip(ids, distances):
            full_entry = all_research.get(uid)
            if full_entry is None:
                # Entry in ChromaDB but pruned from JSON (shouldn't normally happen)
                logger.debug(f"[long_term] Semantic result {uid} not found in JSON store.")
                continue
            # Attach the distance so callers can filter by similarity if desired
            output.append({**full_entry, "_distance": dist})

        return output

    except Exception as e:
        logger.warning(f"[long_term] semantic_query_research failed: {e} — using substring fallback.")
        return query_research(query, n_results)


# ---------------------------------------------------------------------------
# Context summary (called before every task run)
# ---------------------------------------------------------------------------

def get_context_summary(current_goal: str) -> str:
    """
    Build a short summary (≤ 500 chars) of relevant past context for the
    given goal.  This is injected into the system prompt so Claude is
    aware of what has already been tried or discovered — without needing
    an extra tool call.

    Improvement 1: the research section now calls semantic_query_research()
    instead of query_research(), enabling paraphrase-aware retrieval
    (e.g. "passive income" matches "make money online").

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

    # ── Semantically relevant research (Improvement 1) ─────────────────
    # Use the full goal as the query so the embedding captures the whole
    # intent rather than individual keywords — this gives much better recall
    # for paraphrased or conceptually related topics.
    matching_research = semantic_query_research(current_goal, n_results=3)
    if matching_research:
        research_lines = []
        for r in matching_research[-2:]:  # at most 2 research snippets in summary
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
