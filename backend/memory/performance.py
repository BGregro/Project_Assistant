"""
memory/performance.py  —  Phase 12b: Performance Metrics Database

Tracks per-tool call statistics (count, success rate, average duration) and
per-task-type statistics (classified by a lightweight keyword classifier with
no LLM call required).

Design principles:
  - All writes are atomic: data is written to a .tmp file then renamed via
    Path.replace() so a crash mid-write never produces a corrupt JSON file.
  - Never raises — metrics loss is non-fatal. Every public function wraps its
    body in try/except and logs a debug message on failure.
  - Auto-creates the metrics file on first write if it doesn't exist.
  - Thread/async safe at the single-process level (asyncio single-threaded).

File location: <project_root>/memory/performance_metrics.json
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Resolve relative to this file: backend/memory/performance.py
# → backend/ → project root → memory/performance_metrics.json
METRICS_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "memory"
    / "performance_metrics.json"
)

# ---------------------------------------------------------------------------
# Default structure helpers
# ---------------------------------------------------------------------------


def _default_tool_entry() -> dict:
    """Return a fresh per-tool metrics dict with all counters zeroed."""
    return {
        "call_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "total_duration_ms": 0,
        "last_failure_reason": "",
        "last_used": "",
    }


def _default_task_type_entry() -> dict:
    """Return a fresh per-task-type metrics dict with all counters zeroed."""
    return {
        "count": 0,
        "success_count": 0,
        "total_duration_seconds": 0,
        "avg_tools_per_task": 0.0,
    }


def _default_metrics() -> dict:
    """Return the default (empty) top-level metrics structure."""
    return {
        "tools": {},
        "task_types": {},
        "last_updated": "",
    }


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------


def load() -> dict:
    """
    Load performance_metrics.json from disk.

    Returns the default structure if the file is missing, empty, or contains
    invalid JSON — never raises.
    """
    try:
        if METRICS_FILE.exists():
            text = METRICS_FILE.read_text(encoding="utf-8").strip()
            if text:
                return json.loads(text)
    except Exception as e:
        logger.debug(f"[performance] Could not load metrics file (non-fatal): {e}")
    return _default_metrics()


def save(data: dict) -> None:
    """
    Atomically write `data` to performance_metrics.json.

    Writes to a .tmp sidecar first, then renames via Path.replace() so the
    final file is never in a half-written state.  Works on Windows (Path.replace
    is atomic on the same drive).

    Never raises — on any failure logs a debug message and returns silently.
    """
    try:
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = METRICS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(METRICS_FILE)
    except Exception as e:
        logger.debug(f"[performance] Could not save metrics file (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Task-type keyword classifier
# ---------------------------------------------------------------------------

# Each entry: (task_type_label, list_of_regex_patterns).
# Patterns are matched against the lowercased goal string in order.
# First match wins; falls back to "general" if nothing matches.
_TASK_TYPE_RULES: list[tuple[str, list[str]]] = [
    ("software_development", [
        r"\bbuild\b", r"\bcreate\b", r"\bwrite.*code\b", r"\bimplement\b",
        r"\bdevelop\b", r"\bscaffold\b", r"\bcode\b", r"\bprogram\b",
        r"\brefactor\b", r"\bdebug\b", r"\bfix.*bug\b",
    ]),
    ("research", [
        r"\bresearch\b", r"\bsearch\b", r"\bfind\b", r"\binvestigate\b",
        r"\blook up\b", r"\bsummarise\b", r"\bsummarize\b", r"\banalyse\b",
        r"\banalyze\b", r"\bcompare\b",
    ]),
    ("email_management", [
        r"\bemail\b", r"\binbox\b", r"\bgmail\b", r"\bimap\b", r"\bsmtp\b",
        r"\bmail\b",
    ]),
    ("media_processing", [
        r"\bvideo\b", r"\baudio\b", r"\bffmpeg\b", r"\bconvert\b",
        r"\byoutube\b", r"\bclip\b", r"\btranscode\b", r"\bmp4\b",
        r"\bmp3\b",
    ]),
    ("file_operations", [
        r"\bfiles?\b", r"\bfolder\b", r"\bdirectory\b", r"\bread.*files?\b",
        r"\bwrite.*files?\b", r"\blist.*dir\b", r"\blist.*files?\b",
        r"\bmove\b", r"\bcopy\b",
        r"\bdelete.*files?\b", r"\bpatch\b",
    ]),
    ("scheduling", [
        r"\bschedule\b", r"\bremind\b", r"\brun.*at\b", r"\bevery\b",
        r"\bcron\b", r"\btimed\b", r"\brecurring\b",
    ]),
    ("version_control", [
        r"\bgithub\b", r"\bgit\b", r"\brepo\b", r"\bpush\b",
        r"\bcommit\b", r"\bpull request\b", r"\bpr\b",
    ]),
]


def _classify_task_type(goal: str) -> str:
    """
    Classify a task goal string into a task-type category using keyword patterns.

    Pure Python — no LLM call, no network, no disk I/O.
    Returns one of:
        software_development | research | email_management |
        media_processing | file_operations | scheduling |
        version_control | general
    """
    goal_lower = goal.lower()
    for task_type, patterns in _TASK_TYPE_RULES:
        for pattern in patterns:
            if re.search(pattern, goal_lower):
                return task_type
    return "general"


# ---------------------------------------------------------------------------
# Public recording functions
# ---------------------------------------------------------------------------


def record_tool_call(
    tool_name: str,
    success: bool,
    duration_ms: int,
    failure_reason: str = "",
) -> None:
    """
    Record the result of a single tool call.

    Updates call_count, success_count or failure_count, total_duration_ms,
    last_used (ISO timestamp), and last_failure_reason (if failed).

    Args:
        tool_name:      Registered name of the tool that was called.
        success:        True if the tool returned {"success": true} or equivalent.
        duration_ms:    Wall-clock duration of the tool call in milliseconds.
        failure_reason: Short description of the failure (truncated to 200 chars).
                        Only stored when success=False.

    Never raises — all errors are logged at DEBUG level.
    """
    try:
        data = load()
        tools = data.setdefault("tools", {})

        if tool_name not in tools:
            tools[tool_name] = _default_tool_entry()

        entry = tools[tool_name]
        entry["call_count"] += 1
        entry["total_duration_ms"] += duration_ms
        entry["last_used"] = datetime.now(timezone.utc).isoformat()

        if success:
            entry["success_count"] += 1
        else:
            entry["failure_count"] += 1
            entry["last_failure_reason"] = str(failure_reason)[:200]

        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        save(data)
    except Exception as e:
        logger.debug(f"[performance] record_tool_call failed (non-fatal): {e}")


def record_task(
    goal: str,
    outcome: str,
    duration_seconds: int,
    tools_used: list,
    task_type: str = "",
) -> None:
    """
    Record high-level statistics for a completed task.

    Args:
        goal:             The original user message / task description.
        outcome:          "success", "failure", or "cancelled".
        duration_seconds: Wall-clock seconds the task ran for.
        tools_used:       Unique list of tool names used during the task.
        task_type:        Optional explicit task type label.  If empty, the
                          keyword classifier derives it from `goal`.

    Never raises — all errors are logged at DEBUG level.
    """
    try:
        # Derive task type if not explicitly provided
        classified_type = task_type.strip() if task_type.strip() else _classify_task_type(goal)

        data = load()
        task_types = data.setdefault("task_types", {})

        if classified_type not in task_types:
            task_types[classified_type] = _default_task_type_entry()

        entry = task_types[classified_type]
        prev_count = entry["count"]
        entry["count"] += 1
        entry["total_duration_seconds"] += duration_seconds

        if outcome == "success":
            entry["success_count"] += 1

        # Running average for tools-per-task using Welford-style update
        n = entry["count"]
        prev_avg = entry["avg_tools_per_task"]
        entry["avg_tools_per_task"] = round(
            prev_avg + (len(tools_used) - prev_avg) / n, 2
        )

        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        save(data)
    except Exception as e:
        logger.debug(f"[performance] record_task failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------


def get_tool_metrics(tool_name: str = "") -> dict:
    """
    Return per-tool performance metrics with computed summary fields.

    Args:
        tool_name: If non-empty, return only that tool's metrics dict.
                   If empty, return all tools sorted by call_count descending,
                   plus a summary block.

    Computed fields added to each tool entry:
        success_rate_pct  — percentage of successful calls (0.0–100.0)
        failure_rate_pct  — percentage of failed calls (0.0–100.0)
        avg_duration_ms   — average call duration in milliseconds

    Never raises.
    """
    try:
        data = load()
        tools = data.get("tools", {})

        def _enrich(entry: dict) -> dict:
            enriched = dict(entry)
            calls = enriched.get("call_count", 0)
            enriched["success_rate_pct"] = (
                round(enriched.get("success_count", 0) / calls * 100, 1)
                if calls > 0 else 0.0
            )
            enriched["failure_rate_pct"] = (
                round(enriched.get("failure_count", 0) / calls * 100, 1)
                if calls > 0 else 0.0
            )
            enriched["avg_duration_ms"] = (
                round(enriched.get("total_duration_ms", 0) / calls)
                if calls > 0 else 0
            )
            return enriched

        if tool_name:
            if tool_name not in tools:
                return {"error": f"No metrics found for tool '{tool_name}'"}
            return {"tool": tool_name, **_enrich(tools[tool_name])}

        # All tools — sort by call_count descending
        enriched_tools = [
            {"tool": name, **_enrich(entry)}
            for name, entry in sorted(
                tools.items(), key=lambda x: -x[1].get("call_count", 0)
            )
        ]

        total_calls   = sum(e.get("call_count", 0) for e in enriched_tools)
        total_success = sum(e.get("success_count", 0) for e in enriched_tools)
        return {
            "tools": enriched_tools,
            "summary": {
                "total_tool_calls": total_calls,
                "total_successes":  total_success,
                "total_failures":   total_calls - total_success,
                "overall_success_rate_pct": (
                    round(total_success / total_calls * 100, 1)
                    if total_calls > 0 else 0.0
                ),
                "distinct_tools_tracked": len(enriched_tools),
            },
        }
    except Exception as e:
        logger.debug(f"[performance] get_tool_metrics failed (non-fatal): {e}")
        return {"error": str(e)}


def get_task_type_metrics() -> dict:
    """
    Return per-task-type statistics with computed success rates and averages.

    Computed fields added:
        success_rate_pct     — percentage of tasks that succeeded
        avg_duration_seconds — average task duration in seconds

    Never raises.
    """
    try:
        data = load()
        task_types = data.get("task_types", {})

        enriched: dict[str, dict] = {}
        for task_type, entry in task_types.items():
            count = entry.get("count", 0)
            enriched[task_type] = {
                **entry,
                "success_rate_pct": (
                    round(entry.get("success_count", 0) / count * 100, 1)
                    if count > 0 else 0.0
                ),
                "avg_duration_seconds": (
                    round(entry.get("total_duration_seconds", 0) / count)
                    if count > 0 else 0
                ),
            }
        return enriched
    except Exception as e:
        logger.debug(f"[performance] get_task_type_metrics failed (non-fatal): {e}")
        return {}


def get_top_failing_tools(n: int = 5) -> list:
    """
    Return the `n` tools with the highest failure rates.

    Minimum qualification: at least 3 calls must exist before a tool is
    considered for this list (avoids one-off failures dominating the ranking).

    Each entry in the returned list contains:
        tool             — tool name
        call_count       — total calls
        failure_count    — number of failed calls
        failure_rate_pct — failure rate as a percentage
        last_failure_reason — most recent failure message

    Never raises; returns [] on error.
    """
    try:
        data = load()
        tools = data.get("tools", {})

        candidates = []
        for tool_name, entry in tools.items():
            calls = entry.get("call_count", 0)
            if calls < 3:
                continue  # not enough data to be meaningful
            failures = entry.get("failure_count", 0)
            if failures == 0:
                continue
            failure_rate = round(failures / calls * 100, 1)
            candidates.append({
                "tool":                tool_name,
                "call_count":          calls,
                "failure_count":       failures,
                "failure_rate_pct":    failure_rate,
                "last_failure_reason": entry.get("last_failure_reason", ""),
            })

        # Sort by failure_rate_pct descending, then by failure_count descending
        candidates.sort(key=lambda x: (-x["failure_rate_pct"], -x["failure_count"]))
        return candidates[:n]
    except Exception as e:
        logger.debug(f"[performance] get_top_failing_tools failed (non-fatal): {e}")
        return []
