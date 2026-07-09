"""
agent_tools/memory_maintenance.py  —  Phase 16a: Background Memory Maintenance

Runs nightly via APScheduler (3:00 AM, wired in main.py). Never auto-deletes —
flags only. Uses category-specific retention periods so short-lived facts
(current exchange rates, versions) are flagged sooner than durable ones
(learned skills, project history).

Flow:
    1. Classify each fact into a category using qwen3:14b, batched (20 facts
       per call) to keep prompts small and fast on the iGPU.
    2. Apply the category's retention threshold; flag facts older than it.
    3. Flag likely-duplicate facts via cheap string similarity (no LLM call).
    4. Flag research entries older than research_cache_days (config-overridable).
    5. Summarize reflection coverage across all tasks.
    6. Save a JSON report to outputs/maintenance/maintenance_{YYYYMMDD}.json.

Registers two tools via register_maintenance_tools():
    run_memory_maintenance() — runs the full sweep synchronously, returns the report.
    get_maintenance_report() — reads back the most recent saved report.

Never raises outward — a failed maintenance run is logged and reported as
success=False rather than crashing the nightly scheduler job.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# backend/agent_tools/memory_maintenance.py -> parents: agent_tools, backend, <project root>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MAINTENANCE_DIR = _PROJECT_ROOT / "outputs" / "maintenance"

# Retention thresholds in days. None means "never flagged stale".
# "research" is a fallback — the live value is read from config.json's
# top-level research_cache_days when available (matches the rest of the
# codebase, e.g. long_term.get_context_summary()'s research staleness logic).
RETENTION_DAYS: dict[str, int | None] = {
    "current_facts":   90,
    "preferences":      180,
    "skills":           730,
    "project_history":  None,  # never stale
    "research":         7,     # overridden by config research_cache_days
}

_CLASSIFY_BATCH_SIZE = 20
_DUPLICATE_SIMILARITY_THRESHOLD = 0.82


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_ollama_url() -> str:
    """Read ollama_base_url from config.json, fall back to default. Never raises."""
    config_path = _PROJECT_ROOT / "config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f).get("ollama_base_url", "http://localhost:11434")
    except Exception:
        return "http://localhost:11434"


def _load_config() -> dict:
    """Read the full config.json. Returns {} on any failure."""
    config_path = _PROJECT_ROOT / "config.json"
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _age_days(ts_str: str, now: datetime) -> int | None:
    """Whole days elapsed since an ISO timestamp. Returns None if unparsable."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0, (now - dt).days)
    except Exception:
        return None


def _keys_similar(a: str, b: str) -> bool:
    """Cheap string-similarity check for duplicate-fact detection. No LLM call."""
    if not a or not b or a == b:
        return a == b and bool(a)
    return SequenceMatcher(None, a, b).ratio() >= _DUPLICATE_SIMILARITY_THRESHOLD


async def _classify_facts(facts: list[dict], ollama_url: str) -> dict[str, str]:
    """
    Classify each fact's key into a category using qwen3:14b, in batches of
    _CLASSIFY_BATCH_SIZE. Returns {fact_key: category}. Facts the model fails
    to classify (Ollama offline, bad JSON, unrecognized category) are simply
    absent from the returned map — callers should default to "current_facts".
    """
    from agent_tools.local_llm import local_llm_call, strip_think_tags

    valid_categories = {"current_facts", "preferences", "skills", "project_history"}
    categories: dict[str, str] = {}

    for i in range(0, len(facts), _CLASSIFY_BATCH_SIZE):
        batch = facts[i:i + _CLASSIFY_BATCH_SIZE]
        batch_map = {f.get("key", f"unnamed_{i}"): (f.get("value", "") or "")[:80] for f in batch}

        prompt = (
            "Classify each fact into one category: current_facts, preferences, skills, project_history.\n"
            f"Facts: {json.dumps(batch_map, ensure_ascii=False)}\n\n"
            'Reply ONLY with JSON: {"fact_key": "category", ...}'
        )
        try:
            response = await local_llm_call(prompt, model="qwen3:14b", base_url=ollama_url)
            if not response:
                continue
            cleaned = strip_think_tags(response).strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if "\n" in cleaned:
                    cleaned = cleaned.split("\n", 1)[1]
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                for key, category in parsed.items():
                    category = str(category).strip()
                    if category in valid_categories:
                        categories[str(key)] = category
        except Exception as e:
            logger.debug(f"[memory_maintenance] Batch classification failed (non-fatal): {e}")
            continue

    return categories


def _find_duplicate_facts(facts: list[dict]) -> list[dict]:
    """
    Group facts by key similarity (no LLM — simple string comparison).
    Returns a list of {"key1": ..., "key2": ...} pairs flagged as likely
    duplicates. O(n^2) but fact counts are capped at _MAX_FACTS (500), so
    this stays fast.
    """
    duplicates: list[dict] = []
    keys = [(f.get("key", ""), f.get("id", "")) for f in facts]
    for i in range(len(keys)):
        key_i, id_i = keys[i]
        if not key_i:
            continue
        for j in range(i + 1, len(keys)):
            key_j, id_j = keys[j]
            if not key_j:
                continue
            if _keys_similar(key_i.lower().strip(), key_j.lower().strip()):
                duplicates.append({"key1": key_i, "key2": key_j, "id1": id_i, "id2": id_j})
    return duplicates


# ---------------------------------------------------------------------------
# Main maintenance sweep
# ---------------------------------------------------------------------------

async def run_maintenance(ollama_url: str, config: dict) -> dict:
    """
    Run the full nightly maintenance sweep and return (and persist) the report.

    Args:
        ollama_url: Base URL for the local Ollama instance.
        config:     The full config.json contents (for research_cache_days override).

    Returns:
        The maintenance report dict (also saved to
        outputs/maintenance/maintenance_{YYYYMMDD}.json).
    """
    try:
        from memory.long_term import load as load_long_term
        data = load_long_term()
        facts = data.get("facts", [])
        tasks = data.get("tasks", [])
        research = data.get("research", [])
        now = datetime.now(timezone.utc)

        # ── Step 1: classify facts and apply retention thresholds ──────────
        categories = await _classify_facts(facts, ollama_url)
        facts_flagged_stale = 0
        for f in facts:
            key = f.get("key", "")
            category = categories.get(key, "current_facts")
            threshold = RETENTION_DAYS.get(category, 90)
            if threshold is None:
                continue
            age = _age_days(f.get("timestamp", ""), now)
            if age is not None and age > threshold:
                facts_flagged_stale += 1

        # ── Step 2: duplicate/redundant facts ───────────────────────────────
        duplicates = _find_duplicate_facts(facts)
        facts_flagged_duplicate = len(duplicates)

        # ── Step 3: stale research ──────────────────────────────────────────
        research_cache_days = config.get("research_cache_days", RETENTION_DAYS["research"])
        research_flagged_stale = 0
        for r in research:
            age = _age_days(r.get("timestamp", ""), now)
            if age is not None and age > research_cache_days:
                research_flagged_stale += 1

        # ── Step 4: reflection coverage ─────────────────────────────────────
        tasks_total = len(tasks)
        tasks_with_reflection = sum(1 for t in tasks if t.get("reflection"))
        tasks_without_reflection = tasks_total - tasks_with_reflection

        total_flagged = facts_flagged_stale + facts_flagged_duplicate + research_flagged_stale
        report = {
            "run_date": _now_iso(),
            "facts_flagged_stale": facts_flagged_stale,
            "facts_flagged_duplicate": facts_flagged_duplicate,
            "research_flagged_stale": research_flagged_stale,
            "tasks_total": tasks_total,
            "tasks_with_reflection": tasks_with_reflection,
            "tasks_without_reflection": tasks_without_reflection,
            "recommendation": (
                f"{total_flagged} item(s) flagged for review. No automatic deletions."
            ),
            "duplicate_pairs": duplicates,
        }

        # ── Step 5 (Phase 16b): summarize old verbose research entries ─────
        try:
            from memory.long_term import summarize_old_research
            compressed_count = await summarize_old_research(
                ollama_url=ollama_url,
                local_agent_model=config.get("llm", {}).get("local_agent", "qwen3:14b"),
            )
        except Exception as e:
            logger.debug(f"[memory_maintenance] Research summarization step failed (non-fatal): {e}")
            compressed_count = 0
        report["research_entries_compressed"] = compressed_count

        # ── Step 6 (Phase 16d): monthly memory health score ─────────────────
        failed = sum(1 for t in tasks if t.get("outcome") != "success")
        health_score = round(
            (tasks_with_reflection / max(tasks_total, 1)) * 40 +           # reflection coverage (max 40)
            max(0, 30 - (failed / max(tasks_total, 1) * 100)) +            # low failure rate (max 30)
            min(30, len(data.get("research", [])) * 2),                    # research depth (max 30)
            1,
        )
        report["health_score"] = health_score
        report["health_label"] = (
            "excellent" if health_score >= 80 else
            "good" if health_score >= 60 else
            "fair" if health_score >= 40 else "needs attention"
        )

        _MAINTENANCE_DIR.mkdir(parents=True, exist_ok=True)
        report_path = _MAINTENANCE_DIR / f"maintenance_{now.strftime('%Y%m%d')}.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        # Save a dedicated monthly health snapshot on the 1st of the month.
        if now.day == 1:
            monthly_path = _MAINTENANCE_DIR / f"health_{now.strftime('%Y%m')}.json"
            monthly_path.parent.mkdir(parents=True, exist_ok=True)
            monthly_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(f"[memory_maintenance] Monthly health report saved: {monthly_path.name}")

        logger.info(
            f"[memory_maintenance] Sweep complete: {facts_flagged_stale} stale facts, "
            f"{facts_flagged_duplicate} duplicate pairs, {research_flagged_stale} stale research entries, "
            f"{compressed_count} research entries compressed, health score {health_score} ({report['health_label']})."
        )
        return report

    except Exception as e:
        logger.error(f"[memory_maintenance] run_maintenance failed: {e}")
        return {
            "run_date": _now_iso(),
            "success": False,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register_maintenance_tools() -> None:
    """
    Register the memory-maintenance tools with the global tool registry.

    Tools:
        run_memory_maintenance()
        get_maintenance_report()

    Called once at startup from main.py, after the registry is initialised.
    """
    from agent_tools import register_tool

    # ── run_memory_maintenance ──────────────────────────────────────────────
    async def _run_memory_maintenance() -> dict[str, Any]:
        """
        Tool handler: run the full memory maintenance sweep synchronously and
        return the report. Also runs automatically every night at 3:00 AM.

        Returns:
            The maintenance report dict (see run_maintenance() docstring).
        """
        try:
            config = _load_config()
            report = await run_maintenance(_get_ollama_url(), config)
            return report
        except Exception as e:
            logger.error(f"[memory_maintenance] run_memory_maintenance tool failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="run_memory_maintenance",
        description=(
            "Run a full memory health sweep: flags stale facts (using category-aware "
            "retention — current facts 90d, preferences 180d, skills 730d, project "
            "history never), flags likely-duplicate facts, flags stale research entries, "
            "and reports reflection coverage across all tasks. Never deletes anything — "
            "flags only. Also runs automatically every night at 3:00 AM."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=_run_memory_maintenance,
        is_destructive=False,
    )

    # ── get_maintenance_report ──────────────────────────────────────────────
    async def _get_maintenance_report() -> dict[str, Any]:
        """
        Tool handler: read back the most recently saved maintenance report
        without re-running the sweep.

        Returns:
            The most recent report dict, or {"success": False, "error": ...}
            if no report has been generated yet.
        """
        try:
            if not _MAINTENANCE_DIR.exists():
                return {"success": False, "error": "No maintenance reports found yet."}

            report_files = sorted(_MAINTENANCE_DIR.glob("maintenance_*.json"), reverse=True)
            if not report_files:
                return {"success": False, "error": "No maintenance reports found yet."}

            latest = report_files[0]
            report = json.loads(latest.read_text(encoding="utf-8"))
            report.setdefault("success", True)
            report["report_file"] = latest.name
            return report
        except Exception as e:
            logger.error(f"[memory_maintenance] get_maintenance_report failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="get_maintenance_report",
        description=(
            "Read back the most recently generated memory maintenance report without "
            "re-running the sweep. Use this to check memory health quickly."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=_get_maintenance_report,
        is_destructive=False,
    )

    logger.info(
        "[memory_maintenance] Phase 16a: memory maintenance tools registered "
        "(run_memory_maintenance, get_maintenance_report)"
    )
