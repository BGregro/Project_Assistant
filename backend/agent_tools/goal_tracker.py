"""
goal_tracker.py  —  Phase 13a: Goal Tracking System

Tracks long-term goals with milestones, blockers, and linked tasks so the
agent can maintain direction across sessions instead of only reacting to
whatever the user asks in the moment.

Storage: memory/goals.json — {"goals": [...], "last_updated": ISO}
Created automatically on first write. Never raises outward — goal loss is
treated as non-fatal, same convention as long_term.py and episode_memory.py.

Registers six tools via register_goal_tools():
    create_goal, update_goal, list_goals, get_goal,
    log_goal_progress, add_goal_milestone

Also exposes _link_task_to_goal(goal_id, task_id), a plain helper (not a
tool) called from memory/long_term.py's log_task() when a task is tagged
with a goal_id.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# backend/agent_tools/goal_tracker.py -> parents: agent_tools, backend, <project root>
GOALS_FILE = Path(__file__).resolve().parent.parent.parent / "memory" / "goals.json"

_ALLOWED_STATUSES = {"active", "paused", "complete", "abandoned"}
_UPDATABLE_FIELDS = {"status", "priority", "target_date", "current_strategy", "description"}


# ---------------------------------------------------------------------------
# Internal helpers (not registered as tools)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _load() -> dict:
    """
    Read GOALS_FILE. Returns a fresh, empty structure on missing/corrupt file.
    Never raises.
    """
    try:
        if not GOALS_FILE.exists():
            return {"goals": [], "last_updated": ""}
        with open(GOALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "goals" not in data:
            return {"goals": [], "last_updated": ""}
        return data
    except Exception as e:
        logger.warning(f"[goal_tracker] Could not load {GOALS_FILE}: {e} — starting fresh.")
        return {"goals": [], "last_updated": ""}


def _save(data: dict) -> None:
    """
    Atomic write: write to a .tmp file then os.replace() over the target.
    Never raises — failures are logged and swallowed, matching the
    "goal loss is non-fatal" contract for this module.
    """
    try:
        GOALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data["last_updated"] = _now_iso()
        tmp_path = GOALS_FILE.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(GOALS_FILE)
    except Exception as e:
        logger.error(f"[goal_tracker] Failed to save {GOALS_FILE}: {e}")


def _find_goal(goal_id: str, data: dict | None = None) -> tuple[dict | None, list]:
    """
    Locate a goal by id.

    Returns:
        (goal_dict_or_None, all_goals_list)
        The returned goal dict is the same object living inside all_goals_list,
        so mutating it and then calling _save({"goals": all_goals_list}) persists
        the change.
    """
    if data is None:
        data = _load()
    all_goals = data.get("goals", [])
    for g in all_goals:
        if g.get("goal_id") == goal_id:
            return g, all_goals
    return None, all_goals


def _new_goal(title: str, description: str, priority: int = 3, target_date: str | None = None) -> dict:
    """Build a fresh goal dict with default/empty tracking fields."""
    return {
        "goal_id": str(uuid4()),
        "title": title,
        "description": description,
        "status": "active",          # active|paused|complete|abandoned
        "priority": priority,        # 1 (highest) to 5 (lowest)
        "created_date": _now_iso(),
        "target_date": target_date,
        "milestones": [],            # [{id, title, done, date_completed}]
        "current_strategy": "",
        "blockers": [],
        "related_tasks": [],         # task_ids from long_term.json
        "related_projects": [],
        "progress_notes": [],        # [{timestamp, note}]
        "last_updated": _now_iso(),
    }


def _days_since(iso_timestamp: str) -> int | None:
    """Whole days elapsed since an ISO timestamp. Returns None if unparsable."""
    if not iso_timestamp:
        return None
    try:
        then = datetime.fromisoformat(iso_timestamp)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0, (now - then).days)
    except Exception:
        return None


def _link_task_to_goal(goal_id: str, task_id: str) -> bool:
    """
    Append task_id to a goal's related_tasks list if not already present.

    Called from memory/long_term.py's log_task() when a task is tagged with
    a goal_id. Not a registered tool — plain helper only. Non-fatal: returns
    False (and logs) instead of raising if anything goes wrong.
    """
    try:
        data = _load()
        goal, all_goals = _find_goal(goal_id, data)
        if goal is None:
            logger.warning(f"[goal_tracker] _link_task_to_goal: goal '{goal_id}' not found.")
            return False
        if task_id not in goal["related_tasks"]:
            goal["related_tasks"].append(task_id)
            goal["last_updated"] = _now_iso()
            data["goals"] = all_goals
            _save(data)
        return True
    except Exception as e:
        logger.error(f"[goal_tracker] _link_task_to_goal failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register_goal_tools() -> None:
    """
    Register the six goal-tracking tools with the global tool registry.

    Called once at startup from main.py, after the registry is initialised.
    Safe to call multiple times — duplicate names are silently overwritten.
    """
    from agent_tools import register_tool

    # ── create_goal ─────────────────────────────────────────────────────────
    async def _create_goal(
        title: str, description: str, priority: int = 3, target_date: str = ""
    ) -> dict[str, Any]:
        """
        Tool handler: create a new goal entry.

        Args:
            title:       Short goal title.
            description: What success looks like / why this goal matters.
            priority:    1 (critical) to 5 (someday). Default 3 (normal).
            target_date: Optional ISO date string for a target completion date.

        Returns:
            {"success": True, "goal_id": str, "title": str}
        """
        try:
            priority = max(1, min(int(priority), 5))
        except Exception:
            priority = 3

        try:
            data = _load()
            goal = _new_goal(title, description, priority, target_date or None)
            data.setdefault("goals", []).append(goal)
            _save(data)
            logger.info(f"[goal_tracker] Goal created: {goal['goal_id'][:8]} — {title!r}")
            return {"success": True, "goal_id": goal["goal_id"], "title": title}
        except Exception as e:
            logger.error(f"[goal_tracker] create_goal failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="create_goal",
        description=(
            "Create a new long-term goal to track. Use this whenever the user describes "
            "an ongoing objective, project, or ambition they want the agent to help pursue "
            "over multiple sessions — not a one-off task. priority 1=critical to 5=someday. "
            "target_date is an optional ISO date string. Returns a goal_id you can use with "
            "log_goal_progress, update_goal, and get_goal."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short goal title."},
                "description": {
                    "type": "string",
                    "description": "What success looks like / why this goal matters.",
                },
                "priority": {
                    "type": "integer",
                    "description": "1 (critical) to 5 (someday). Default 3.",
                    "default": 3,
                },
                "target_date": {
                    "type": "string",
                    "description": "Optional ISO date string for a target completion date.",
                    "default": "",
                },
            },
            "required": ["title", "description"],
        },
        handler=_create_goal,
        is_destructive=False,
    )

    # ── update_goal ─────────────────────────────────────────────────────────
    async def _update_goal(goal_id: str, field: str, value: str) -> dict[str, Any]:
        """
        Tool handler: update one field on an existing goal.

        Args:
            goal_id: UUID of the goal to update.
            field:   One of status, priority, target_date, current_strategy, description.
            value:   New value (string; priority is coerced to int).

        Returns:
            {"success": True, "goal_id": ..., "field": ..., "new_value": ...}
        """
        if field not in _UPDATABLE_FIELDS:
            return {
                "success": False,
                "error": f"field must be one of {sorted(_UPDATABLE_FIELDS)}, got {field!r}",
            }
        if field == "status" and value not in _ALLOWED_STATUSES:
            return {
                "success": False,
                "error": f"status must be one of {sorted(_ALLOWED_STATUSES)}, got {value!r}",
            }

        try:
            data = _load()
            goal, all_goals = _find_goal(goal_id, data)
            if goal is None:
                return {"success": False, "error": f"Goal '{goal_id}' not found."}

            new_value: Any = value
            if field == "priority":
                try:
                    new_value = max(1, min(int(value), 5))
                except Exception:
                    return {"success": False, "error": f"priority must be an integer, got {value!r}"}

            goal[field] = new_value
            goal["last_updated"] = _now_iso()
            data["goals"] = all_goals
            _save(data)
            return {"success": True, "goal_id": goal_id, "field": field, "new_value": new_value}
        except Exception as e:
            logger.error(f"[goal_tracker] update_goal failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="update_goal",
        description=(
            "Update a single field on an existing goal: status, priority, target_date, "
            "current_strategy, or description. For status, allowed values are "
            "active, paused, complete, abandoned. Use this to mark a goal complete, "
            "change its priority, or record a change in strategy."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "UUID of the goal to update."},
                "field": {
                    "type": "string",
                    "description": "One of: status, priority, target_date, current_strategy, description.",
                },
                "value": {"type": "string", "description": "New value for the field."},
            },
            "required": ["goal_id", "field", "value"],
        },
        handler=_update_goal,
        is_destructive=False,
    )

    # ── list_goals ──────────────────────────────────────────────────────────
    async def _list_goals(status: str = "active") -> dict[str, Any]:
        """
        Tool handler: list goals filtered by status, sorted by priority then recency.

        Args:
            status: "active", "paused", "complete", "abandoned", or "all".

        Returns:
            {"success": True, "goals": [...], "count": N}
        """
        try:
            data = _load()
            all_goals = data.get("goals", [])
            status = (status or "active").lower().strip()

            if status == "all":
                filtered = list(all_goals)
            else:
                filtered = [g for g in all_goals if g.get("status") == status]

            # priority ascending (1 first), then created_date descending (newest first).
            # Sort by the secondary key first, then the primary key, since Python's
            # sort is stable — this yields the correct combined ordering.
            filtered.sort(key=lambda g: g.get("created_date", ""), reverse=True)
            filtered.sort(key=lambda g: g.get("priority", 3))

            enriched = []
            for g in filtered:
                g2 = dict(g)
                last_note_ts = g["progress_notes"][-1]["timestamp"] if g.get("progress_notes") else g.get("created_date", "")
                g2["days_since_progress"] = _days_since(last_note_ts)
                enriched.append(g2)

            return {"success": True, "goals": enriched, "count": len(enriched)}
        except Exception as e:
            logger.error(f"[goal_tracker] list_goals failed: {e}")
            return {"success": False, "error": str(e), "goals": [], "count": 0}

    register_tool(
        name="list_goals",
        description=(
            "List goals filtered by status (default 'active'; pass 'all' for everything). "
            "Sorted by priority (1=critical first), then most recently created. Each goal "
            "includes days_since_progress. Call this before starting any long or multi-session "
            "task to check whether it relates to an existing goal."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "active | paused | complete | abandoned | all. Default 'active'.",
                    "default": "active",
                },
            },
            "required": [],
        },
        handler=_list_goals,
        is_destructive=False,
    )

    # ── get_goal ────────────────────────────────────────────────────────────
    async def _get_goal(goal_id: str) -> dict[str, Any]:
        """
        Tool handler: retrieve a goal in full, with related_tasks enriched from
        long_term.json (goal, outcome, and a truncated reflection per task).

        Args:
            goal_id: UUID of the goal to retrieve.

        Returns:
            {"success": True, "goal": {...}}
        """
        try:
            data = _load()
            goal, _ = _find_goal(goal_id, data)
            if goal is None:
                return {"success": False, "error": f"Goal '{goal_id}' not found."}

            goal_out = dict(goal)
            enriched_tasks = []
            try:
                from memory.long_term import load as load_long_term
                lt = load_long_term()
                tasks_by_id = {t.get("id"): t for t in lt.get("tasks", [])}
                for task_id in goal.get("related_tasks", []):
                    t = tasks_by_id.get(task_id)
                    if t is None:
                        enriched_tasks.append({"task_id": task_id, "found": False})
                        continue
                    reflection = (t.get("reflection") or "")[:80]
                    enriched_tasks.append(
                        {
                            "task_id": task_id,
                            "found": True,
                            "goal": t.get("goal", ""),
                            "outcome": t.get("outcome", ""),
                            "reflection": reflection,
                        }
                    )
            except Exception as e:
                logger.warning(f"[goal_tracker] get_goal: could not enrich related_tasks: {e}")
                enriched_tasks = [{"task_id": tid, "found": False} for tid in goal.get("related_tasks", [])]

            goal_out["related_tasks"] = enriched_tasks
            return {"success": True, "goal": goal_out}
        except Exception as e:
            logger.error(f"[goal_tracker] get_goal failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="get_goal",
        description=(
            "Retrieve full detail for a single goal: description, milestones, blockers, "
            "progress notes, current strategy, and related tasks (each enriched with its "
            "outcome and a short reflection snippet). Use this before resuming work on a "
            "goal to recall full context."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "UUID of the goal to retrieve."},
            },
            "required": ["goal_id"],
        },
        handler=_get_goal,
        is_destructive=False,
    )

    # ── log_goal_progress ───────────────────────────────────────────────────
    async def _log_goal_progress(
        goal_id: str, note: str, milestone_title: str = ""
    ) -> dict[str, Any]:
        """
        Tool handler: append a timestamped progress note to a goal, optionally
        marking a milestone done (creating it first if it doesn't yet exist).

        Args:
            goal_id:         UUID of the goal.
            note:             Progress note text.
            milestone_title: If provided, find-or-create this milestone and mark it done.

        Returns:
            {"success": True, "goal_id": ..., "milestone_completed": bool}
        """
        try:
            data = _load()
            goal, all_goals = _find_goal(goal_id, data)
            if goal is None:
                return {"success": False, "error": f"Goal '{goal_id}' not found."}

            goal.setdefault("progress_notes", []).append(
                {"timestamp": _now_iso(), "note": note}
            )

            milestone_completed = False
            milestone_title = (milestone_title or "").strip()
            if milestone_title:
                milestones = goal.setdefault("milestones", [])
                target = next(
                    (m for m in milestones if m.get("title") == milestone_title), None
                )
                if target is None:
                    target = {
                        "id": str(uuid4()),
                        "title": milestone_title,
                        "done": False,
                        "date_completed": None,
                    }
                    milestones.append(target)
                if not target.get("done"):
                    target["done"] = True
                    target["date_completed"] = _now_iso()
                    milestone_completed = True

            goal["last_updated"] = _now_iso()
            data["goals"] = all_goals
            _save(data)
            return {
                "success": True,
                "goal_id": goal_id,
                "milestone_completed": milestone_completed,
            }
        except Exception as e:
            logger.error(f"[goal_tracker] log_goal_progress failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="log_goal_progress",
        description=(
            "Append a progress note to a goal after completing work that advances it. "
            "Optionally pass milestone_title to mark (or create-and-mark) a milestone as done. "
            "Call this after finishing any task related to a goal — keeps the goal's history current."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "UUID of the goal."},
                "note": {"type": "string", "description": "Short progress note."},
                "milestone_title": {
                    "type": "string",
                    "description": "Optional: title of a milestone to find-or-create and mark done.",
                    "default": "",
                },
            },
            "required": ["goal_id", "note"],
        },
        handler=_log_goal_progress,
        is_destructive=False,
    )

    # ── add_goal_milestone ──────────────────────────────────────────────────
    async def _add_goal_milestone(goal_id: str, milestone_title: str) -> dict[str, Any]:
        """
        Tool handler: add a new, not-yet-done milestone to a goal.

        Args:
            goal_id:         UUID of the goal.
            milestone_title: Title for the new milestone.

        Returns:
            {"success": True, "milestone_id": str}
        """
        try:
            data = _load()
            goal, all_goals = _find_goal(goal_id, data)
            if goal is None:
                return {"success": False, "error": f"Goal '{goal_id}' not found."}

            milestone_id = str(uuid4())
            goal.setdefault("milestones", []).append(
                {
                    "id": milestone_id,
                    "title": milestone_title,
                    "done": False,
                    "date_completed": None,
                }
            )
            goal["last_updated"] = _now_iso()
            data["goals"] = all_goals
            _save(data)
            return {"success": True, "milestone_id": milestone_id}
        except Exception as e:
            logger.error(f"[goal_tracker] add_goal_milestone failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="add_goal_milestone",
        description=(
            "Add a new incomplete milestone to a goal, useful for planning ahead before "
            "any progress has been made on it. To mark a milestone done, use "
            "log_goal_progress(goal_id, note, milestone_title=...) instead."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "UUID of the goal."},
                "milestone_title": {"type": "string", "description": "Title for the new milestone."},
            },
            "required": ["goal_id", "milestone_title"],
        },
        handler=_add_goal_milestone,
        is_destructive=False,
    )

    logger.info(
        "[goal_tracker] Phase 13a: goal tracking tools registered "
        "(create_goal, update_goal, list_goals, get_goal, log_goal_progress, add_goal_milestone)"
    )
