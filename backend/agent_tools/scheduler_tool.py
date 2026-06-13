"""
scheduler_tool.py  —  Phase 5e: Scheduler Tools

Exposes three agent-callable tools that wrap TaskScheduler:

  schedule_task          — register a new recurring or one-time task
  list_scheduled_tasks   — inspect all active schedules
  cancel_scheduled_task  — remove a scheduled task by ID

The TaskScheduler instance is injected at startup via set_scheduler()
(called from main.py after the scheduler is created) so this module
does not import TaskScheduler at module level, avoiding circular imports.

schedule_task and cancel_scheduled_task are marked destructive because
they autonomously consume Claude API tokens in the background.
"""

import logging
from agent_tools import register_tool

logger = logging.getLogger(__name__)

# Module-level reference to the TaskScheduler — set by main.py at startup.
_scheduler = None


def set_scheduler(scheduler) -> None:
    """
    Store the TaskScheduler instance so the tool handlers can reach it.
    Must be called from main.py before register_scheduler_tools().
    """
    global _scheduler
    _scheduler = scheduler
    logger.info("[scheduler_tool] Scheduler reference set.")


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def _schedule_task(task_id: str, message: str, schedule: str) -> dict:
    """
    Schedule a recurring or one-time agent task.

    Args:
        task_id:  Short unique name for this schedule, e.g. 'daily_trends'.
                  Used to reference the task in cancel_scheduled_task.
        message:  What the agent should do when the schedule fires — same as
                  typing a message in the chat, e.g.
                  'Search YouTube for trending Shorts today and log findings'.
        schedule: Human-readable schedule string.
                  Examples:
                    'every hour'
                    'every 30 minutes'
                    'every day at 09:00'
                    'every monday at 08:00'
                    'once at 2026-07-15 10:00'
    """
    if _scheduler is None:
        return {
            "success": False,
            "error": (
                "Scheduler is not initialised. "
                "Ensure task_scheduler.start() was called at server startup."
            ),
        }
    return await _scheduler.schedule_task(
        task_id=task_id,
        message=message,
        schedule_str=schedule,
    )


async def _list_scheduled_tasks() -> dict:
    """Return all currently active scheduled tasks with their next run times."""
    if _scheduler is None:
        return {"success": False, "error": "Scheduler is not initialised.", "tasks": [], "count": 0}
    tasks = _scheduler.list_scheduled()
    return {"success": True, "tasks": tasks, "count": len(tasks)}


async def _cancel_scheduled_task(task_id: str) -> dict:
    """
    Cancel a scheduled task by its ID.

    Args:
        task_id: The ID used when the task was scheduled.
    """
    if _scheduler is None:
        return {"success": False, "error": "Scheduler is not initialised."}
    return _scheduler.cancel_task(task_id)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_scheduler_tools() -> None:
    """Register Phase 5e scheduler tools into the live tool registry."""

    register_tool(
        name="schedule_task",
        description=(
            "Schedule an agent task to run automatically at a given time or interval. "
            "task_id is a short unique name (e.g. 'weekly_research', 'daily_trends'). "
            "message is what the agent will do when triggered — write it exactly as you "
            "would type it in chat (e.g. 'Search YouTube for trending Shorts and log findings'). "
            "schedule examples: 'every hour', 'every 30 minutes', 'every day at 09:00', "
            "'every monday at 08:00', 'once at 2026-07-15 10:00'. "
            "Scheduled tasks run in the background even when you are not chatting. "
            "Use list_scheduled_tasks() first to check whether a similar task already exists. "
            "Use cancel_scheduled_task(task_id) to stop a recurring task."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": (
                        "Short unique identifier for this schedule, e.g. 'daily_trends'. "
                        "Letters, numbers, and underscores only. Re-using an existing ID "
                        "replaces the previous schedule."
                    ),
                },
                "message": {
                    "type": "string",
                    "description": (
                        "The instruction the agent will execute when the schedule fires. "
                        "Write it as if you were typing it in chat."
                    ),
                },
                "schedule": {
                    "type": "string",
                    "description": (
                        "Human-readable schedule string. "
                        "Examples: 'every hour', 'every 30 minutes', "
                        "'every day at 09:00', 'every monday at 08:00', "
                        "'once at 2026-07-15 10:00'."
                    ),
                },
            },
            "required": ["task_id", "message", "schedule"],
        },
        handler=_schedule_task,
        is_destructive=True,   # creates autonomous background actions
    )

    register_tool(
        name="list_scheduled_tasks",
        description=(
            "List all currently active scheduled tasks. "
            "Returns each task's ID, message, schedule string, and next run time. "
            "Call this before scheduling a new task to check for duplicates."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=_list_scheduled_tasks,
        is_destructive=False,
    )

    register_tool(
        name="cancel_scheduled_task",
        description=(
            "Cancel a scheduled task by its ID, stopping any future runs. "
            "Use list_scheduled_tasks() to find the correct task_id."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The ID of the scheduled task to cancel.",
                },
            },
            "required": ["task_id"],
        },
        handler=_cancel_scheduled_task,
        is_destructive=True,   # permanently removes the schedule
    )
