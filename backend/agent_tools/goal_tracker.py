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


def _get_ollama_url() -> str:
    """Read ollama_base_url from config.json, fall back to default. Never raises."""
    config_path = Path(__file__).resolve().parent.parent.parent / "config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f).get("ollama_base_url", "http://localhost:11434")
    except Exception:
        return "http://localhost:11434"


def log_goal_progress_helper(
    goal_id: str, note: str, milestone_title: str = ""
) -> dict[str, Any]:
    """
    Plain (non-tool) implementation shared by the log_goal_progress tool
    handler and by memory/long_term.py's log_task() (Phase 13b), which
    auto-logs progress on a goal whenever a linked task completes
    successfully. Synchronous — same convention as _link_task_to_goal.

    Returns:
        {"success": True, "goal_id": ..., "milestone_completed": bool, "milestone_title": ...}
        or {"success": False, "error": ...}
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
            "milestone_title": milestone_title,
        }
    except Exception as e:
        logger.error(f"[goal_tracker] log_goal_progress_helper failed: {e}")
        return {"success": False, "error": str(e)}


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
            {"success": True, "goal_id": ..., "milestone_completed": bool, "milestone_title": ...}
        """
        return log_goal_progress_helper(goal_id, note, milestone_title)

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

    # ── get_goal_progress ───────────────────────────────────────────────────
    async def _get_goal_progress(goal_id: str) -> dict[str, Any]:
        """
        Tool handler: return progress notes, milestones, and completion stats
        for a goal without the full enriched detail get_goal() provides.

        Args:
            goal_id: UUID of the goal.

        Returns:
            {"success": True, "goal_id": ..., "title": ..., "progress_notes": [...],
             "milestones": [...], "days_active": int, "completion_pct": float}
        """
        try:
            data = _load()
            goal, _ = _find_goal(goal_id, data)
            if goal is None:
                return {"success": False, "error": f"Goal '{goal_id}' not found."}

            milestones = goal.get("milestones", [])
            done_count = sum(1 for m in milestones if m.get("done"))
            completion_pct = (
                round((done_count / len(milestones)) * 100, 1) if milestones else 0.0
            )
            days_active = _days_since(goal.get("created_date", "")) or 0

            return {
                "success": True,
                "goal_id": goal_id,
                "title": goal.get("title", ""),
                "progress_notes": goal.get("progress_notes", []),
                "milestones": milestones,
                "days_active": days_active,
                "completion_pct": completion_pct,
            }
        except Exception as e:
            logger.error(f"[goal_tracker] get_goal_progress failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="get_goal_progress",
        description=(
            "Get progress notes, milestone list, and completion percentage for a goal. "
            "Lighter-weight than get_goal() — use this for a quick progress check "
            "without the enriched related_tasks detail."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "UUID of the goal."},
            },
            "required": ["goal_id"],
        },
        handler=_get_goal_progress,
        is_destructive=False,
    )

    # ── decompose_goal ──────────────────────────────────────────────────────
    async def _decompose_goal(goal_id: str, max_tasks: int = 6) -> dict[str, Any]:
        """
        Tool handler: use the local agent-tier model to break a goal into
        concrete sub-tasks, added as new (not-yet-done) milestones.

        Args:
            goal_id:   UUID of the goal to decompose.
            max_tasks: Upper bound on how many sub-tasks to generate (1-12, default 6).

        Returns:
            {"success": True, "goal_id": ..., "tasks_generated": N, "tasks": [...]}
        """
        try:
            data = _load()
            goal, all_goals = _find_goal(goal_id, data)
            if goal is None:
                return {"success": False, "error": f"Goal '{goal_id}' not found."}

            try:
                max_tasks = max(1, min(int(max_tasks), 12))
            except Exception:
                max_tasks = 6

            existing_milestones = goal.get("milestones", [])
            existing_titles = [m.get("title", "") for m in existing_milestones]

            prompt = (
                f"Decompose this goal into {max_tasks} concrete, actionable sub-tasks.\n"
                f"Goal: {goal.get('title', '')}\n"
                f"Description: {goal.get('description', '')}\n"
                f"Existing milestones: {existing_titles}\n\n"
                "Return ONLY a JSON array of strings, each a specific task description. "
                "Tasks should be ordered logically. No preamble, no markdown."
            )

            from agent_tools.local_llm import local_llm_call, strip_think_tags
            response = await local_llm_call(prompt, model="qwen3:14b", base_url=_get_ollama_url())
            if not response:
                return {
                    "success": False,
                    "error": "Local LLM unavailable — could not decompose goal.",
                }

            cleaned = strip_think_tags(response).strip()
            # Tolerate accidental markdown code fences from the local model
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if "\n" in cleaned:
                    cleaned = cleaned.split("\n", 1)[1]

            try:
                parsed = json.loads(cleaned)
                if not isinstance(parsed, list):
                    raise ValueError("response was not a JSON array")
                sub_tasks = [str(t).strip() for t in parsed if str(t).strip()]
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Could not parse local LLM response as a JSON array: {e}",
                }

            existing_lower = {t.lower() for t in existing_titles}
            added: list[str] = []
            for task_desc in sub_tasks[:max_tasks]:
                if task_desc.lower() in existing_lower:
                    continue
                existing_milestones.append({
                    "id": str(uuid4()),
                    "title": task_desc,
                    "done": False,
                    "date_completed": None,
                })
                existing_lower.add(task_desc.lower())
                added.append(task_desc)

            goal["milestones"] = existing_milestones
            goal["last_updated"] = _now_iso()
            data["goals"] = all_goals
            _save(data)

            return {
                "success": True,
                "goal_id": goal_id,
                "tasks_generated": len(added),
                "tasks": added,
            }
        except Exception as e:
            logger.error(f"[goal_tracker] decompose_goal failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="decompose_goal",
        description=(
            "Use the local qwen3:14b model to break a large goal into concrete, "
            "ordered sub-tasks, automatically added as new milestones. Use this when "
            "a goal is high-level and needs a plan before work can begin. "
            "max_tasks caps how many sub-tasks are generated (default 6, max 12)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "UUID of the goal to decompose."},
                "max_tasks": {
                    "type": "integer",
                    "description": "Maximum number of sub-tasks to generate. Default 6.",
                    "default": 6,
                },
            },
            "required": ["goal_id"],
        },
        handler=_decompose_goal,
        is_destructive=False,
    )

    # ── detect_goal_blocker ─────────────────────────────────────────────────
    async def _detect_goal_blocker(goal_id: str, stall_days: int = 7) -> dict[str, Any]:
        """
        Tool handler: check whether a goal has stalled (no progress notes for
        more than stall_days) and, if so, ask the local model for one
        specific unblocking suggestion.

        Args:
            goal_id:    UUID of the goal to check.
            stall_days: Days without progress before a goal is considered stalled (default 7).

        Returns:
            {"blocked": True,  "days_stalled": N, "suggestion": str} or
            {"blocked": False, "days_since_progress": N}
        """
        try:
            data = _load()
            goal, _ = _find_goal(goal_id, data)
            if goal is None:
                return {"success": False, "error": f"Goal '{goal_id}' not found."}

            last_note_ts = (
                goal["progress_notes"][-1]["timestamp"]
                if goal.get("progress_notes") else goal.get("created_date", "")
            )
            days_since_progress = _days_since(last_note_ts) or 0

            if days_since_progress > stall_days and goal.get("status") == "active":
                from agent_tools.local_llm import local_llm_call
                prompt = (
                    f"This goal has stalled for {days_since_progress} days. "
                    "Suggest one specific action to unblock it: "
                    f"{goal.get('title', '')} — {goal.get('description', '')}"
                )
                suggestion = await local_llm_call(prompt, model="qwen3:14b", base_url=_get_ollama_url())
                return {
                    "success": True,
                    "blocked": True,
                    "goal_id": goal_id,
                    "days_stalled": days_since_progress,
                    "suggestion": suggestion or "Review this goal and decide on one concrete next step.",
                }

            return {
                "success": True,
                "blocked": False,
                "goal_id": goal_id,
                "days_since_progress": days_since_progress,
            }
        except Exception as e:
            logger.error(f"[goal_tracker] detect_goal_blocker failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="detect_goal_blocker",
        description=(
            "Check whether a goal has stalled — no progress notes logged for more than "
            "stall_days (default 7). If stalled and still active, returns a local-model "
            "suggestion for one concrete unblocking action. Use this periodically on "
            "active goals, or when the user asks 'what's stuck?'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "UUID of the goal to check."},
                "stall_days": {
                    "type": "integer",
                    "description": "Days without progress before flagging as stalled. Default 7.",
                    "default": 7,
                },
            },
            "required": ["goal_id"],
        },
        handler=_detect_goal_blocker,
        is_destructive=False,
    )

    # ── schedule_goal_work ──────────────────────────────────────────────────
    async def _schedule_goal_work(
        goal_id: str, schedule: str = "every monday at 09:00"
    ) -> dict[str, Any]:
        """
        Tool handler: register a recurring scheduled task (via TaskScheduler)
        that periodically checks progress on a goal and works its next
        incomplete milestone.

        Args:
            goal_id:  UUID of the goal.
            schedule: Human-readable schedule string (same format as schedule_task).

        Returns:
            {"success": True, "goal_id": ..., "scheduled": schedule}
        """
        try:
            data = _load()
            goal, _ = _find_goal(goal_id, data)
            if goal is None:
                return {"success": False, "error": f"Goal '{goal_id}' not found."}

            task_message = (
                f"Check progress on goal '{goal.get('title', '')}' and work on the "
                "next incomplete milestone."
            )

            from agent_tools import scheduler_tool
            if scheduler_tool._scheduler is None:
                return {"success": False, "error": "Task scheduler is not available."}

            result = await scheduler_tool._scheduler.schedule_task(
                task_id=f"goal_{goal_id[:8]}",
                message=task_message,
                schedule_str=schedule,
            )
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error", "Failed to schedule goal work."),
                }

            return {"success": True, "goal_id": goal_id, "scheduled": schedule}
        except Exception as e:
            logger.error(f"[goal_tracker] schedule_goal_work failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="schedule_goal_work",
        description=(
            "Register a recurring scheduled task that periodically checks a goal's "
            "progress and works its next incomplete milestone — same schedule string "
            "format as schedule_task (e.g. 'every monday at 09:00'). Confirm the "
            "schedule with the user first, since it runs autonomously."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "UUID of the goal."},
                "schedule": {
                    "type": "string",
                    "description": "Human-readable schedule string. Default 'every monday at 09:00'.",
                    "default": "every monday at 09:00",
                },
            },
            "required": ["goal_id"],
        },
        handler=_schedule_goal_work,
        is_destructive=True,
    )

    # ── generate_goal_report ────────────────────────────────────────────────
    async def _generate_goal_report() -> dict[str, Any]:
        """
        Tool handler: build a concise, local-model-written weekly progress
        report across all active goals, save it to outputs/goal_reports/,
        and email it if email notifications are enabled in config.json.

        Returns:
            {"success": True, "report_path": str, "goals_covered": N}
        """
        try:
            data = _load()
            all_goals = data.get("goals", [])
            active_goals = [g for g in all_goals if g.get("status") == "active"]

            goal_summaries: list[str] = []
            for g in active_goals:
                milestones = g.get("milestones", [])
                done_count = sum(1 for m in milestones if m.get("done"))
                completion_pct = (
                    round((done_count / len(milestones)) * 100, 1) if milestones else 0.0
                )
                days_active = _days_since(g.get("created_date", "")) or 0
                last_note_ts = (
                    g["progress_notes"][-1]["timestamp"]
                    if g.get("progress_notes") else g.get("created_date", "")
                )
                days_since_progress = _days_since(last_note_ts) or 0
                goal_summaries.append(
                    f"- {g.get('title', '')} (priority {g.get('priority', 3)}): "
                    f"{completion_pct}% complete ({done_count}/{len(milestones)} milestones), "
                    f"active {days_active}d, last progress {days_since_progress}d ago"
                )

            if not goal_summaries:
                report_text = "No active goals to report on this week."
            else:
                from agent_tools.local_llm import local_llm_call
                prompt = (
                    "Write a weekly goal progress report. Be direct and actionable.\n"
                    "Active goals:\n" + "\n".join(goal_summaries) + "\n\n"
                    "Focus on: what's on track, what's stalled, recommended next actions."
                )
                report_text = await local_llm_call(prompt, model="qwen3:14b", base_url=_get_ollama_url())
                if not report_text:
                    report_text = (
                        "Local LLM unavailable — raw goal summary:\n" + "\n".join(goal_summaries)
                    )

            report_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "goal_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now(timezone.utc)
            report_path = report_dir / f"report_{today.strftime('%Y%m%d')}.md"
            report_path.write_text(
                f"# Weekly Goal Report — {today.strftime('%Y-%m-%d')}\n\n{report_text}\n",
                encoding="utf-8",
            )

            # Optional email delivery — non-fatal if disabled or unavailable
            try:
                config_path = Path(__file__).resolve().parent.parent.parent / "config.json"
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                if cfg.get("email", {}).get("enabled"):
                    from agent_tools.notification_tool import send_email
                    await send_email(
                        subject="Weekly Goal Progress Report",
                        body=report_text,
                    )
            except Exception as e:
                logger.debug(f"[goal_tracker] generate_goal_report: email delivery skipped ({e})")

            return {
                "success": True,
                "report_path": str(report_path),
                "goals_covered": len(active_goals),
            }
        except Exception as e:
            logger.error(f"[goal_tracker] generate_goal_report failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="generate_goal_report",
        description=(
            "Generate a concise, actionable weekly progress report across all active "
            "goals. Saves to outputs/goal_reports/ as markdown and emails it if email "
            "notifications are enabled. Also runs automatically every Sunday at 20:00."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=_generate_goal_report,
        is_destructive=False,
    )

    # ── delete_goal ─────────────────────────────────────────────────────────
    async def _delete_goal(goal_id: str, reason: str = "") -> dict[str, Any]:
        """
        Permanently delete a goal by ID. Use this to remove duplicate goals,
        abandoned goals, or goals that are no longer relevant.
        Logs a note before deletion so there is an audit trail.

        Args:
            goal_id: The goal_id of the goal to delete
            reason: Why the goal is being deleted (optional but recommended)
        """
        try:
            data = _load()
            goals = data.get("goals", [])
            target = next((g for g in goals if g["goal_id"] == goal_id), None)
            if not target:
                return {"success": False, "error": f"Goal '{goal_id}' not found."}
            title = target.get("title", "unknown")
            data["goals"] = [g for g in goals if g["goal_id"] != goal_id]
            data["last_updated"] = _now_iso()
            _save(data)
            logger.info(f"[goal_tracker] Deleted goal '{title}' ({goal_id}). Reason: {reason or 'not specified'}")
            return {"success": True, "deleted_goal_id": goal_id, "deleted_title": title, "reason": reason}
        except Exception as e:
            logger.error(f"[goal_tracker] delete_goal failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="delete_goal",
        description=(
            "Permanently delete a goal by ID. Use this to remove duplicate goals, "
            "abandoned goals, or goals that are no longer relevant. Always call "
            "list_goals(status='all') first to find the correct goal_id — this "
            "action is irreversible."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "The goal_id of the goal to delete."},
                "reason": {
                    "type": "string",
                    "description": "Why the goal is being deleted (optional but recommended).",
                    "default": "",
                },
            },
            "required": ["goal_id"],
        },
        handler=_delete_goal,
        is_destructive=True,
    )

    logger.info(
        "[goal_tracker] Phase 13a-13d: goal tracking tools registered "
        "(create_goal, update_goal, list_goals, get_goal, log_goal_progress, "
        "add_goal_milestone, get_goal_progress, decompose_goal, detect_goal_blocker, "
        "schedule_goal_work, generate_goal_report, delete_goal)"
    )
