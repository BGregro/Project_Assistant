"""
task_scheduler.py  —  Phase 5e: Scheduled / Recurring Tasks

Uses APScheduler's AsyncIOScheduler so all jobs execute inside the existing
FastAPI async event loop without spawning threads.

Schedule string format (human-readable, parsed by _parse_schedule):
  "every 30 minutes"          → IntervalTrigger(minutes=30)
  "every 2 hours"             → IntervalTrigger(hours=2)
  "every hour"                → IntervalTrigger(hours=1)
  "every day at 09:00"        → CronTrigger(hour=9, minute=0)
  "every monday at 08:30"     → CronTrigger(day_of_week='mon', hour=8, minute=30)
  "once at 2026-07-01 12:00"  → DateTrigger(run_date='2026-07-01 12:00')

Schedules are persisted to memory/scheduled_tasks.json and restored on
every server start so tasks survive restarts.

Security note: scheduled tasks run autonomously and call the agent's full
tool chain including destructive tools. The schedule_task tool is therefore
marked destructive so the user sees a confirmation dialog before any
recurring task is registered.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

# Persisted schedule file (gitignored)
_SCHEDULE_FILE = Path(__file__).resolve().parent.parent / "memory" / "scheduled_tasks.json"


def _parse_schedule(schedule_str: str) -> dict:
    """
    Parse a human-readable schedule string into an APScheduler trigger spec.

    Returns a dict with:
        trigger  — "interval" | "cron" | "date"
        **kwargs — trigger-specific keyword arguments

    Raises ValueError if the string cannot be parsed.
    """
    s = schedule_str.strip().lower()

    # ── "every N minutes / hours / seconds" ──────────────────────────────────
    m = re.match(r"every\s+(\d+)\s+(minute|minutes|hour|hours|second|seconds)", s)
    if m:
        amount = int(m.group(1))
        unit   = m.group(2).rstrip("s")   # normalise to singular
        return {"trigger": "interval", unit + "s": amount}

    # ── "every hour" (no number) ─────────────────────────────────────────────
    if re.match(r"every\s+hour$", s):
        return {"trigger": "interval", "hours": 1}

    # ── "every minute" (no number) ───────────────────────────────────────────
    if re.match(r"every\s+minute$", s):
        return {"trigger": "interval", "minutes": 1}

    # ── "every day at HH:MM" ─────────────────────────────────────────────────
    m = re.match(r"every\s+day\s+at\s+(\d{1,2}):(\d{2})", s)
    if m:
        return {"trigger": "cron", "hour": int(m.group(1)), "minute": int(m.group(2))}

    # ── "every {weekday} at HH:MM" ───────────────────────────────────────────
    days = {
        "monday": "mon", "tuesday": "tue", "wednesday": "wed", "thursday": "thu",
        "friday": "fri", "saturday": "sat", "sunday": "sun",
    }
    for day_name, day_abbr in days.items():
        m = re.match(rf"every\s+{day_name}\s+at\s+(\d{{1,2}}):(\d{{2}})", s)
        if m:
            return {
                "trigger":      "cron",
                "day_of_week":  day_abbr,
                "hour":         int(m.group(1)),
                "minute":       int(m.group(2)),
            }

    # ── "once at YYYY-MM-DD HH:MM" ───────────────────────────────────────────
    m = re.match(r"once\s+at\s+(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2})", s)
    if m:
        return {"trigger": "date", "run_date": m.group(1)}

    raise ValueError(
        f"Cannot parse schedule string: {schedule_str!r}. "
        "Examples: 'every hour', 'every 30 minutes', 'every day at 09:00', "
        "'every monday at 08:00', 'once at 2026-07-01 12:00'"
    )


def _build_trigger(spec: dict):
    """Build an APScheduler trigger object from a parsed spec dict."""
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.cron    import CronTrigger
    from apscheduler.triggers.date    import DateTrigger

    trigger_type = spec.pop("trigger")
    if trigger_type == "interval":
        return IntervalTrigger(**spec)
    elif trigger_type == "cron":
        return CronTrigger(**spec)
    elif trigger_type == "date":
        return DateTrigger(**spec)
    else:
        raise ValueError(f"Unknown trigger type: {trigger_type!r}")


class TaskScheduler:
    """
    Owns all scheduled / recurring agent tasks.

    Lifecycle (called from main.py):
        scheduler = TaskScheduler()
        set_scheduler(scheduler)          # gives scheduler_tool.py a reference
        register_scheduler_tools()        # registers schedule_task et al. with the registry
        scheduler.set_refs(...)           # inject agent + runner references
        scheduler.start()                 # start APScheduler + restore persisted schedules
        ...
        scheduler.shutdown()              # on FastAPI shutdown event
    """

    SCHEDULE_FILE: Path = _SCHEDULE_FILE

    def __init__(self) -> None:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        self._scheduler = AsyncIOScheduler()

        # task_id → metadata dict persisted to disk
        self._jobs: dict[str, dict] = {}

        # References injected by main.py after all objects are created
        self._agent_ref                  = None
        self._runner_ref                 = None
        self._send_event_ref: Callable | None = None   # broadcast function
        self._pending_confirmations_ref: dict | None = None

    # ------------------------------------------------------------------
    # Dependency injection
    # ------------------------------------------------------------------

    def set_refs(
        self,
        agent,
        task_runner,
        send_event: Callable[[str, dict], Awaitable[None]],
        pending_confirmations: dict,
    ) -> None:
        """
        Inject the agent, runner, and WebSocket broadcast callback.
        Called from main.py after all top-level objects exist.

        send_event must be a broadcast function — it must send to ALL active
        WebSocket connections, not just the one that created the schedule, since
        the user may have disconnected and reconnected by the time a job fires.
        """
        self._agent_ref                  = agent
        self._runner_ref                 = task_runner
        self._send_event_ref             = send_event
        self._pending_confirmations_ref  = pending_confirmations

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the APScheduler and restore persisted schedules."""
        self._scheduler.start()
        logger.info("[scheduler] APScheduler started.")
        self._load_and_reschedule()

    def shutdown(self) -> None:
        """Gracefully stop the scheduler (called on FastAPI shutdown)."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("[scheduler] APScheduler shut down.")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def schedule_task(
        self,
        task_id: str,
        message: str,
        schedule_str: str,
    ) -> dict:
        """
        Parse schedule_str, register the job with APScheduler, and persist.

        Returns a result dict with next_run_time on success,
        or success=False + error on failure.
        """
        if not task_id or not task_id.strip():
            return {"success": False, "error": "task_id must not be empty."}
        if not message or not message.strip():
            return {"success": False, "error": "message must not be empty."}

        # Parse the human-readable schedule string
        try:
            spec = _parse_schedule(schedule_str)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        # Build the APScheduler trigger (spec dict is mutated by _build_trigger)
        try:
            trigger = _build_trigger(spec)
        except Exception as e:
            return {"success": False, "error": f"Failed to build trigger: {e}"}

        # Register with APScheduler; replace_existing=True allows re-scheduling
        try:
            job = self._scheduler.add_job(
                self._run_job,
                trigger=trigger,
                args=[task_id, message],
                id=task_id,
                replace_existing=True,
                name=f"agent_task:{task_id}",
            )
        except Exception as e:
            return {"success": False, "error": f"APScheduler error: {e}"}

        next_run = str(job.next_run_time) if job.next_run_time else "unknown"

        # Persist metadata
        self._jobs[task_id] = {
            "task_id":  task_id,
            "message":  message,
            "schedule": schedule_str,
            "created":  datetime.utcnow().isoformat(),
        }
        self._save()

        logger.info(
            f"[scheduler] Scheduled '{task_id}' — {schedule_str!r} "
            f"(next run: {next_run})"
        )
        return {
            "success":       True,
            "task_id":       task_id,
            "schedule":      schedule_str,
            "next_run_time": next_run,
        }

    def cancel_task(self, task_id: str) -> dict:
        """Remove a scheduled task by ID."""
        if task_id not in self._jobs:
            return {"success": False, "error": f"No scheduled task with id '{task_id}'."}

        try:
            self._scheduler.remove_job(task_id)
        except Exception as e:
            logger.warning(f"[scheduler] remove_job('{task_id}') error (non-fatal): {e}")

        self._jobs.pop(task_id, None)
        self._save()
        logger.info(f"[scheduler] Cancelled scheduled task '{task_id}'.")
        return {"success": True, "task_id": task_id}

    def list_scheduled(self) -> list[dict]:
        """Return metadata for all currently scheduled tasks."""
        result = []
        for task_id, meta in self._jobs.items():
            job      = self._scheduler.get_job(task_id)
            next_run = str(job.next_run_time) if (job and job.next_run_time) else "unknown"
            result.append({
                "task_id":       task_id,
                "message":       meta["message"],
                "schedule":      meta["schedule"],
                "next_run_time": next_run,
                "created":       meta.get("created", ""),
            })
        return result

    # ------------------------------------------------------------------
    # Job execution
    # ------------------------------------------------------------------

    async def _run_job(self, task_id: str, message: str) -> None:
        """
        Called by APScheduler when a job fires.

        Sends a status broadcast so the frontend shows activity, then runs
        the agent's full task-runner loop with the scheduled message.
        """
        logger.info(f"[scheduler] Firing job '{task_id}': {message[:80]!r}")

        if self._send_event_ref:
            try:
                await self._send_event_ref(
                    "status",
                    {"text": f"⏰ Scheduled task '{task_id}' starting…"},
                )
            except Exception as e:
                logger.warning(f"[scheduler] Status broadcast failed (non-fatal): {e}")

        if self._agent_ref is None or self._runner_ref is None:
            logger.error(
                f"[scheduler] Job '{task_id}' fired but agent/runner refs are not set. "
                "Call set_refs() before start()."
            )
            return

        # Reuse an empty history — scheduled tasks always start fresh
        history: list[dict] = []

        try:
            reply = await self._agent_ref.run_with_task_runner(
                task_runner=self._runner_ref,
                user_message=message,
                history=history,
                send_event=self._send_event_ref or _noop_send_event,
                pending_confirmations=self._pending_confirmations_ref or {},
                context_summary=f"[Scheduled task: {task_id}]",
            )
            logger.info(
                f"[scheduler] Job '{task_id}' complete. "
                f"Reply: {str(reply)[:120]!r}"
            )
            if self._send_event_ref:
                await self._send_event_ref(
                    "status",
                    {"text": f"✅ Scheduled task '{task_id}' finished."},
                )
        except Exception as e:
            logger.exception(f"[scheduler] Job '{task_id}' raised an exception")
            if self._send_event_ref:
                try:
                    await self._send_event_ref(
                        "error",
                        {"text": f"Scheduled task '{task_id}' failed: {e}"},
                    )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Write current job metadata to disk."""
        try:
            self.SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.SCHEDULE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._jobs, f, ensure_ascii=False, indent=2)
            logger.debug(f"[scheduler] Saved {len(self._jobs)} schedule(s) to disk.")
        except Exception as e:
            logger.warning(f"[scheduler] Could not save schedule file: {e}")

    def _load_and_reschedule(self) -> None:
        """Read persisted schedules and re-register them with APScheduler."""
        if not self.SCHEDULE_FILE.exists():
            logger.info("[scheduler] No persisted schedules found.")
            return

        try:
            with open(self.SCHEDULE_FILE, encoding="utf-8") as f:
                saved = json.load(f)
        except Exception as e:
            logger.warning(f"[scheduler] Could not read schedule file: {e}")
            return

        if not isinstance(saved, dict):
            logger.warning("[scheduler] Schedule file has unexpected format — skipping.")
            return

        restored = 0
        for task_id, meta in saved.items():
            schedule_str = meta.get("schedule", "")
            message      = meta.get("message", "")
            if not schedule_str or not message:
                continue
            try:
                spec    = _parse_schedule(schedule_str)
                trigger = _build_trigger(spec)
                self._scheduler.add_job(
                    self._run_job,
                    trigger=trigger,
                    args=[task_id, message],
                    id=task_id,
                    replace_existing=True,
                    name=f"agent_task:{task_id}",
                )
                self._jobs[task_id] = meta
                restored += 1
                logger.info(f"[scheduler] Restored schedule '{task_id}' ({schedule_str!r})")
            except Exception as e:
                logger.warning(
                    f"[scheduler] Could not restore schedule '{task_id}': {e}"
                )

        logger.info(f"[scheduler] Restored {restored} schedule(s) from disk.")


# ---------------------------------------------------------------------------
# Noop send_event fallback (used when refs are not yet set)
# ---------------------------------------------------------------------------

async def _noop_send_event(event_type: str, data: dict) -> None:
    """Silently discard events when no WebSocket connection is active."""
    pass
