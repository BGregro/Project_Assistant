"""
episode_memory.py  —  Phase 12a: Episode Journal Tools

Exposes three tools that let Claude interact with the episode journal:

  log_reflection(task_id, reflection_text)
    Manually add or update a reflection on a past task entry.
    Normally reflections are generated automatically in the background after
    every task completion (by TaskRunner._generate_reflection using qwen3:14b).
    This tool lets Claude add reflections mid-conversation when it notices
    something worth recording — e.g. "I notice I always search before coding".

  get_episode(task_id)
    Retrieve the full task entry for a given task_id, including its reflection,
    tools_used, outcome, duration, and any other stored fields.
    Useful for Phase 14 (failure classification) and Phase 17 (strategy mining).

  get_recent_episodes(n)
    Return the last N task entries from long_term.json, including reflections.
    Lets Claude review its own recent history at a glance to spot patterns,
    identify what went well, and plan improvements.

All three tools are non-destructive (read or append-only).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_episode_memory_tools() -> None:
    """
    Register episode memory tools with the global tool registry.

    Called once at startup from main.py, after the registry is initialised.
    Safe to call multiple times — duplicate names are silently overwritten.
    """
    from agent_tools import register_tool

    # ── log_reflection ─────────────────────────────────────────────────────
    async def _log_reflection(task_id: str, reflection_text: str) -> dict[str, Any]:
        """
        Tool handler: add or update a reflection on an existing task entry.

        Args:
            task_id:         UUID of the task to annotate (from get_recent_episodes).
            reflection_text: 2-3 sentence reflection. Write in first person.
                             Focus on what worked, what didn't, and what to
                             do differently next time.

        Returns:
            {"success": bool, "task_id": str}
        """
        try:
            from memory.long_term import log_reflection
            found = log_reflection(task_id, reflection_text)
            if found:
                logger.info(f"[episode_memory] Manual reflection stored for task {task_id[:8]}")
                return {"success": True, "task_id": task_id}
            else:
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": f"Task '{task_id}' not found in long_term.json",
                }
        except Exception as e:
            logger.error(f"[episode_memory] log_reflection failed: {e}")
            return {"success": False, "task_id": task_id, "error": str(e)}

    register_tool(
        name="log_reflection",
        description=(
            "Add or update a reflection on a past task entry in the episode journal. "
            "Reflections are normally generated automatically in the background after "
            "every task, but this tool lets you add one manually when you notice "
            "something worth recording. Write 2-3 sentences in first person focusing "
            "on what worked, what didn't, and what you'd do differently. "
            "Get task_id values from get_recent_episodes()."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "UUID of the task entry to annotate.",
                },
                "reflection_text": {
                    "type": "string",
                    "description": (
                        "2-3 sentence reflection in first person. "
                        "Cover: what worked, what didn't, what to do differently next time."
                    ),
                },
            },
            "required": ["task_id", "reflection_text"],
        },
        handler=_log_reflection,
        is_destructive=False,
    )

    # ── get_episode ─────────────────────────────────────────────────────────
    async def _get_episode(task_id: str) -> dict[str, Any]:
        """
        Tool handler: retrieve a complete task episode including its reflection.

        Args:
            task_id: UUID of the task entry to retrieve.

        Returns:
            The full task dict, or {"success": False, "error": ...} if not found.
        """
        try:
            from memory.long_term import get_episode
            episode = get_episode(task_id)
            if episode is not None:
                return {"success": True, "episode": episode}
            else:
                return {
                    "success": False,
                    "error": f"Task '{task_id}' not found in long_term.json",
                }
        except Exception as e:
            logger.error(f"[episode_memory] get_episode failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="get_episode",
        description=(
            "Retrieve the full task episode for a given task_id, including its reflection, "
            "tools used, outcome, duration, and failure metadata. "
            "Use this when you need to examine a specific past task in detail — "
            "for Phase 14 failure classification, Phase 17 strategy extraction, "
            "or when the user asks about a specific past run. "
            "Get task_id values from get_recent_episodes()."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "UUID of the task entry to retrieve.",
                },
            },
            "required": ["task_id"],
        },
        handler=_get_episode,
        is_destructive=False,
    )

    # ── get_recent_episodes ─────────────────────────────────────────────────
    async def _get_recent_episodes(n: int = 10) -> dict[str, Any]:
        """
        Tool handler: return the most recent N task episodes including reflections.

        Args:
            n: Number of episodes to return (default 10, capped at 50).

        Returns:
            {"success": True, "episodes": [...], "count": N}
        """
        try:
            from memory.long_term import load
            n = max(1, min(n, 50))   # clamp to 1–50
            data = load()
            tasks = data.get("tasks", [])
            recent = tasks[-n:] if len(tasks) >= n else tasks
            # Return newest first for easier reading
            recent = list(reversed(recent))
            return {
                "success": True,
                "episodes": recent,
                "count": len(recent),
                "total_stored": len(tasks),
            }
        except Exception as e:
            logger.error(f"[episode_memory] get_recent_episodes failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="get_recent_episodes",
        description=(
            "Return the most recent N task episodes from the episode journal, "
            "including their reflections, tools used, outcomes, and durations. "
            "Use this to review recent history, spot patterns across tasks, "
            "and understand what has been working or failing. "
            "Returns episodes newest-first. "
            "Default n=10; max 50. Each episode includes a 'task_id' field "
            "you can pass to get_episode() for full detail or log_reflection() "
            "to add a manual annotation."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of recent episodes to return (default 10, max 50).",
                    "default": 10,
                },
            },
            "required": [],
        },
        handler=_get_recent_episodes,
        is_destructive=False,
    )

    logger.info("[episode_memory] Phase 12a: episode memory tools registered (log_reflection, get_episode, get_recent_episodes)")
