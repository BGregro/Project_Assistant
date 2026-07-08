"""
agent_tools/reflection_engine.py  —  Phase 14b + 14c: Self-Reflection Engine

Auto-classifies task failures into a fixed set of categories and, once enough
evidence accumulates, generates structured improvement rules ("if X, then Y")
that get injected into every future system prompt.

Phase 14b — Failure classification:
    classify_failure_background(task_id, goal, failed_at_tool, error_summary, ollama_url)
        Fire-and-forget coroutine. Classifies a failed task into one of
        FAILURE_TYPES via the local qwen3:14b model and writes the result
        back onto the task entry via memory.long_term.update_task_failure_type().
    classify_failure(task_id)  — registered tool
        Synchronous (user-invoked) wrapper: loads the task, runs the same
        classification logic, and returns the result immediately instead of
        firing in the background.

Phase 14c — Pattern detection and rule generation:
    generate_rule_if_pattern(ollama_url)
        Runs every 10 completed tasks. Scans the last 30 tasks for a
        failure_type occurring >= 3 times and asks the local model to
        propose a concrete prevention rule, stored in
        memory/improvement_proposals.json with status="proposed".
    get_improvement_proposals(status) / apply_improvement_proposal(rule_id) /
    retire_rule(rule_id, reason)  — registered tools to review, activate, and
        retire proposals. Active rules are written to memory/active_rules.json,
        which agent_core.py injects into the system prompt as [LEARNED RULES].

Storage: memory/improvement_proposals.json — {"proposals": [...], "tasks_since_last_check": N}
Never raises outward — all failures are logged and swallowed, matching the
non-fatal conventions used across memory/long_term.py and goal_tracker.py.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# backend/agent_tools/reflection_engine.py -> parents: agent_tools, backend, <project root>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROPOSALS_FILE = _PROJECT_ROOT / "memory" / "improvement_proposals.json"
ACTIVE_RULES_FILE = _PROJECT_ROOT / "memory" / "active_rules.json"

FAILURE_TYPES = [
    "tool_integration_error",  # tool call failed or returned wrong format
    "logic_error",              # agent reasoned incorrectly
    "knowledge_gap",            # agent lacked needed information
    "resource_constraint",      # rate limit, context overflow, timeout
    "user_communication",       # misunderstood the goal
    "external_failure",         # API down, file missing, network error
]

# Minimum occurrences of the same failure_type (within the last 30 tasks)
# before a rule is proposed.
_PATTERN_MIN_EVIDENCE = 3
# How many completed tasks between pattern-detection sweeps.
_TASKS_BETWEEN_CHECKS = 10
_TASKS_WINDOW = 30


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


def _load_proposals() -> dict:
    """
    Read PROPOSALS_FILE. Returns a fresh, empty structure on missing/corrupt file.
    Never raises.
    """
    try:
        if not PROPOSALS_FILE.exists():
            return {"proposals": [], "tasks_since_last_check": 0}
        with open(PROPOSALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "proposals" not in data:
            return {"proposals": [], "tasks_since_last_check": 0}
        data.setdefault("tasks_since_last_check", 0)
        return data
    except Exception as e:
        logger.warning(f"[reflection_engine] Could not load {PROPOSALS_FILE}: {e} — starting fresh.")
        return {"proposals": [], "tasks_since_last_check": 0}


def _save_proposals(data: dict) -> None:
    """Atomic write: .tmp then os.replace(). Never raises — logs and swallows."""
    try:
        PROPOSALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = PROPOSALS_FILE.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(PROPOSALS_FILE)
    except Exception as e:
        logger.error(f"[reflection_engine] Failed to save {PROPOSALS_FILE}: {e}")


def _save_active_rules(proposals: list[dict]) -> None:
    """
    Write the current status="active" proposals to memory/active_rules.json as
    a simple list of {"if", "then"} — the shape agent_core.py reads directly.
    Non-fatal on failure.
    """
    try:
        active = [
            {"rule_id": p["rule_id"], "if": p["if"], "then": p["then"]}
            for p in proposals
            if p.get("status") == "active"
        ]
        ACTIVE_RULES_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = ACTIVE_RULES_FILE.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(active, f, indent=2, ensure_ascii=False)
        tmp_path.replace(ACTIVE_RULES_FILE)
    except Exception as e:
        logger.error(f"[reflection_engine] Failed to save {ACTIVE_RULES_FILE}: {e}")


def _rules_similar(a: str, b: str) -> bool:
    """
    Cheap string-similarity check to avoid proposing near-duplicate rules.
    No LLM call needed — a simple normalized-word-overlap heuristic is enough
    to catch "same pattern proposed twice" without false-positiving on
    genuinely different rules.
    """
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return False
    overlap = len(wa & wb) / max(1, min(len(wa), len(wb)))
    return overlap > 0.6


# ---------------------------------------------------------------------------
# Phase 14b — Failure classification
# ---------------------------------------------------------------------------

async def classify_failure_background(
    task_id: str,
    goal: str,
    failed_at_tool: str,
    error_summary: str,
    ollama_url: str,
) -> None:
    """
    Fire-and-forget: classify a failed task into one of FAILURE_TYPES via the
    local model and persist the result onto the task entry.

    Never raises — any error is logged and swallowed so a classification
    failure can never surface as a user-visible error.
    """
    try:
        from agent_tools.local_llm import local_llm_call, strip_think_tags

        prompt = (
            "Classify this task failure into exactly one category.\n"
            f"Task goal: {goal}\n"
            f"Failed at tool: {failed_at_tool}\n"
            f"Error: {error_summary[:1000]}\n"
            f"Categories: {FAILURE_TYPES}\n"
            "Reply with ONLY the category name, nothing else."
        )
        response = await local_llm_call(prompt, model="qwen3:14b", base_url=ollama_url)
        if not response:
            logger.debug("[reflection_engine] classify_failure_background: local LLM unavailable, skipping.")
            return

        classified = strip_think_tags(response).strip().lower()
        # Tolerate the model echoing punctuation/quotes around the category name
        classified = classified.strip(" .`'\"")
        if classified not in FAILURE_TYPES:
            logger.debug(
                f"[reflection_engine] Unrecognized classification {classified!r} — defaulting to logic_error."
            )
            classified = "logic_error"

        from memory.long_term import update_task_failure_type
        update_task_failure_type(task_id, classified)

    except Exception as e:
        logger.warning(f"[reflection_engine] classify_failure_background failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Phase 14c — Pattern detection and rule generation
# ---------------------------------------------------------------------------

async def generate_rule_if_pattern(ollama_url: str) -> None:
    """
    Runs every _TASKS_BETWEEN_CHECKS completed tasks (tracked via a counter
    persisted in improvement_proposals.json). Scans the last _TASKS_WINDOW
    tasks for failure_types occurring >= _PATTERN_MIN_EVIDENCE times and asks
    the local model to draft a concrete prevention rule for each pattern.

    Never raises — background job, all errors logged and swallowed.
    """
    try:
        proposals_data = _load_proposals()
        proposals_data["tasks_since_last_check"] = proposals_data.get("tasks_since_last_check", 0) + 1

        if proposals_data["tasks_since_last_check"] < _TASKS_BETWEEN_CHECKS:
            _save_proposals(proposals_data)
            return

        proposals_data["tasks_since_last_check"] = 0

        from memory.long_term import load as load_long_term
        lt = load_long_term()
        recent_tasks = lt.get("tasks", [])[-_TASKS_WINDOW:]

        # Group tasks by failure_type
        by_type: dict[str, list[dict]] = {}
        for t in recent_tasks:
            ft = t.get("failure_type", "")
            if ft:
                by_type.setdefault(ft, []).append(t)

        existing_proposals = proposals_data.get("proposals", [])
        new_rule_count = 0

        for failure_type, tasks in by_type.items():
            if len(tasks) < _PATTERN_MIN_EVIDENCE:
                continue

            task_summaries = "\n".join(
                f"- {t.get('goal', '')[:80]} (failed at: {t.get('failed_at_tool', 'unknown')})"
                for t in tasks[-8:]
            )

            from agent_tools.local_llm import local_llm_call, strip_think_tags
            prompt = (
                f"These tasks all failed with '{failure_type}':\n"
                f"{task_summaries}\n\n"
                "Generate a concrete prevention rule as JSON:\n"
                '{"if": "condition", "then": "action to take", "evidence_count": N}\n'
                "Return ONLY the JSON object."
            )
            response = await local_llm_call(prompt, model="qwen3:14b", base_url=ollama_url)
            if not response:
                continue

            cleaned = strip_think_tags(response).strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if "\n" in cleaned:
                    cleaned = cleaned.split("\n", 1)[1]

            try:
                rule_data = json.loads(cleaned)
                if_condition = str(rule_data.get("if", "")).strip()
                then_action = str(rule_data.get("then", "")).strip()
                if not if_condition or not then_action:
                    raise ValueError("missing if/then")
            except Exception as e:
                logger.debug(f"[reflection_engine] Could not parse rule JSON for {failure_type}: {e}")
                continue

            # Skip if a similar rule already exists (any status)
            if any(
                _rules_similar(if_condition, p.get("if", ""))
                for p in existing_proposals
            ):
                logger.debug(f"[reflection_engine] Similar rule already exists for {failure_type} — skipping.")
                continue

            rule = {
                "rule_id": str(uuid4()),
                "if": if_condition,
                "then": then_action,
                "evidence_count": len(tasks),
                "failure_type": failure_type,
                "status": "proposed",
                "created": _now_iso(),
            }
            existing_proposals.append(rule)
            new_rule_count += 1
            logger.info(f"[reflection_engine] New rule proposed for {failure_type!r}: {if_condition[:60]!r}")

        proposals_data["proposals"] = existing_proposals
        _save_proposals(proposals_data)

        if new_rule_count:
            logger.info(f"[reflection_engine] Pattern sweep complete: {new_rule_count} new rule(s) proposed.")

    except Exception as e:
        logger.warning(f"[reflection_engine] generate_rule_if_pattern failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register_reflection_engine_tools() -> None:
    """
    Register the reflection-engine tools with the global tool registry.

    Tools:
        classify_failure(task_id)
        get_improvement_proposals(status)
        apply_improvement_proposal(rule_id)
        retire_rule(rule_id, reason)

    Called once at startup from main.py, after the registry is initialised.
    """
    from agent_tools import register_tool

    # ── classify_failure ────────────────────────────────────────────────────
    async def _classify_failure(task_id: str) -> dict[str, Any]:
        """
        Tool handler: synchronously classify why a specific task failed.

        Unlike classify_failure_background (fired automatically after every
        failed task), this runs immediately and returns the result — useful
        when the user explicitly asks "why did that fail?".

        Args:
            task_id: UUID of the task to classify.

        Returns:
            {"success": True, "task_id": ..., "failure_type": ..., "goal": ...}
        """
        try:
            from memory.long_term import get_episode, update_task_failure_type
            task = get_episode(task_id)
            if task is None:
                return {"success": False, "error": f"Task '{task_id}' not found."}

            goal = task.get("goal", "")
            failed_at_tool = task.get("failed_at_tool", "")
            error_summary = task.get("summary", "") or str(task.get("reflection", ""))

            from agent_tools.local_llm import local_llm_call, strip_think_tags
            prompt = (
                "Classify this task failure into exactly one category.\n"
                f"Task goal: {goal}\n"
                f"Failed at tool: {failed_at_tool}\n"
                f"Error: {error_summary[:1000]}\n"
                f"Categories: {FAILURE_TYPES}\n"
                "Reply with ONLY the category name, nothing else."
            )
            response = await local_llm_call(prompt, model="qwen3:14b", base_url=_get_ollama_url())
            if not response:
                return {"success": False, "error": "Local LLM unavailable — could not classify."}

            classified = strip_think_tags(response).strip().lower().strip(" .`'\"")
            if classified not in FAILURE_TYPES:
                classified = "logic_error"

            update_task_failure_type(task_id, classified)

            return {
                "success": True,
                "task_id": task_id,
                "failure_type": classified,
                "goal": goal,
            }
        except Exception as e:
            logger.error(f"[reflection_engine] classify_failure failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="classify_failure",
        description=(
            "Analyze why a specific task failed and classify it into one of: "
            "tool_integration_error, logic_error, knowledge_gap, resource_constraint, "
            "user_communication, external_failure. Updates the task's failure_type field. "
            "Use this when the user asks what went wrong with a past task."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "UUID of the task to classify."},
            },
            "required": ["task_id"],
        },
        handler=_classify_failure,
        is_destructive=False,
    )

    # ── get_improvement_proposals ───────────────────────────────────────────
    async def _get_improvement_proposals(status: str = "proposed") -> dict[str, Any]:
        """
        Tool handler: list AI-generated improvement rules filtered by status.

        Args:
            status: "proposed" | "active" | "retired" | "all". Default "proposed".

        Returns:
            {"success": True, "proposals": [...], "count": N}
        """
        try:
            data = _load_proposals()
            all_proposals = data.get("proposals", [])
            status = (status or "proposed").lower().strip()

            if status == "all":
                filtered = list(all_proposals)
            else:
                filtered = [p for p in all_proposals if p.get("status") == status]

            filtered.sort(key=lambda p: p.get("created", ""), reverse=True)
            return {"success": True, "proposals": filtered, "count": len(filtered)}
        except Exception as e:
            logger.error(f"[reflection_engine] get_improvement_proposals failed: {e}")
            return {"success": False, "error": str(e), "proposals": [], "count": 0}

    register_tool(
        name="get_improvement_proposals",
        description=(
            "List AI-generated improvement rules derived from failure patterns. "
            "status: 'proposed' (default, awaiting review), 'active' (currently "
            "injected into the system prompt), 'retired', or 'all'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "proposed | active | retired | all. Default 'proposed'.",
                    "default": "proposed",
                },
            },
            "required": [],
        },
        handler=_get_improvement_proposals,
        is_destructive=False,
    )

    # ── apply_improvement_proposal ──────────────────────────────────────────
    async def _apply_improvement_proposal(rule_id: str) -> dict[str, Any]:
        """
        Tool handler: activate a proposed rule — status "proposed" -> "active".
        Active rules are written to memory/active_rules.json, which
        agent_core.py injects into every future system prompt.

        Args:
            rule_id: UUID of the rule to activate.

        Returns:
            {"success": True, "rule_id": ..., "status": "active"}
        """
        try:
            data = _load_proposals()
            proposals = data.get("proposals", [])
            rule = next((p for p in proposals if p.get("rule_id") == rule_id), None)
            if rule is None:
                return {"success": False, "error": f"Rule '{rule_id}' not found."}

            rule["status"] = "active"
            rule["activated"] = _now_iso()
            data["proposals"] = proposals
            _save_proposals(data)
            _save_active_rules(proposals)

            return {"success": True, "rule_id": rule_id, "status": "active"}
        except Exception as e:
            logger.error(f"[reflection_engine] apply_improvement_proposal failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="apply_improvement_proposal",
        description=(
            "Activate a proposed improvement rule. Active rules are automatically "
            "injected into every future system prompt as [LEARNED RULES], so the "
            "agent applies the lesson going forward without needing to be reminded."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "rule_id": {"type": "string", "description": "UUID of the rule to activate."},
            },
            "required": ["rule_id"],
        },
        handler=_apply_improvement_proposal,
        is_destructive=False,
    )

    # ── retire_rule ──────────────────────────────────────────────────────────
    async def _retire_rule(rule_id: str, reason: str = "") -> dict[str, Any]:
        """
        Tool handler: retire an active or proposed rule — status -> "retired".

        Args:
            rule_id: UUID of the rule to retire.
            reason:  Optional note on why the rule is no longer helpful.

        Returns:
            {"success": True, "rule_id": ..., "status": "retired"}
        """
        try:
            data = _load_proposals()
            proposals = data.get("proposals", [])
            rule = next((p for p in proposals if p.get("rule_id") == rule_id), None)
            if rule is None:
                return {"success": False, "error": f"Rule '{rule_id}' not found."}

            rule["status"] = "retired"
            rule["retired"] = _now_iso()
            rule["retire_reason"] = reason
            data["proposals"] = proposals
            _save_proposals(data)
            _save_active_rules(proposals)

            return {"success": True, "rule_id": rule_id, "status": "retired"}
        except Exception as e:
            logger.error(f"[reflection_engine] retire_rule failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="retire_rule",
        description=(
            "Retire a rule (proposed or active) that is no longer helpful. Retired "
            "rules are removed from the [LEARNED RULES] system prompt injection but "
            "kept in history for reference."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "rule_id": {"type": "string", "description": "UUID of the rule to retire."},
                "reason": {
                    "type": "string",
                    "description": "Optional note on why this rule is being retired.",
                    "default": "",
                },
            },
            "required": ["rule_id"],
        },
        handler=_retire_rule,
        is_destructive=False,
    )

    logger.info(
        "[reflection_engine] Phase 14b-14c: reflection engine tools registered "
        "(classify_failure, get_improvement_proposals, apply_improvement_proposal, retire_rule)"
    )
