"""
batch_tools.py  —  Phase 11.5c: Reflection Backfill + Batch Job Tools

Exposes two agent-callable tools built on top of batch_processor.py:

  backfill_reflections()
    Finds every task in long_term.json missing a reflection, submits one
    batch request per task asking qwen-quality-equivalent Claude for a
    2-3 sentence reflection, and schedules a recurring APScheduler job
    that polls the batch and writes results back via log_reflection()
    once it completes. Runs at 50% of the normal per-token cost since
    it uses the Anthropic Message Batches API.

  list_batch_jobs()
    Lists all batches that haven't yet been retrieved, so Claude (or the
    user) can check progress on backfills / other background batch work.

The TaskScheduler instance is injected at startup via set_scheduler()
(called from main.py, same pattern as agent_tools/scheduler_tool.py) so
this module doesn't import TaskScheduler at module level.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level reference to the TaskScheduler — set by main.py at startup.
_scheduler = None

# Name of the recurring APScheduler job that polls in-flight reflection
# backfill batches. A single shared job name means repeated calls to
# backfill_reflections() simply keep the same polling job alive rather
# than stacking up duplicate jobs.
_POLL_JOB_ID = "batch_reflection_poller"


def set_scheduler(scheduler) -> None:
    """
    Store the TaskScheduler instance so backfill_reflections() can register
    the background polling job. Must be called from main.py before
    register_batch_tools() is exercised (safe to call in either order,
    since the reference is only read at call time).
    """
    global _scheduler
    _scheduler = scheduler
    logger.info("[batch_tools] Scheduler reference set.")


REFLECTION_PROMPT_TEMPLATE = (
    "You are reviewing a completed AI agent task for its episode journal. "
    "Write a 2-3 sentence reflection in first person, covering what worked, "
    "what didn't, and what to do differently next time. Be concise and concrete.\n\n"
    "Goal: {goal}\n"
    "Outcome: {outcome}\n"
    "Tools used: {tools_used}\n"
    "Duration: {duration} seconds\n\n"
    "Reflection:"
)


async def _poll_and_process_batches() -> None:
    """
    APScheduler job body (runs every 30 minutes).

    Iterates all pending (not-yet-retrieved) batches, polls each one, and
    for any that have ended, retrieves results and writes each reflection
    via log_reflection(), then marks the batch retrieved.

    Never raises — all failures are logged and swallowed so a bad batch
    can't take down the scheduler's job loop.
    """
    import batch_processor
    from memory.long_term import log_reflection

    try:
        pending = batch_processor.list_pending_batches()
    except Exception as e:
        logger.error(f"[batch_tools] Failed to list pending batches (non-fatal): {e}")
        return

    if not pending:
        logger.debug("[batch_tools] No pending batches to poll.")
        return

    for batch_meta in pending:
        batch_id = batch_meta.get("batch_id", "")
        job_name = batch_meta.get("job_name", "unknown")
        if not batch_id:
            continue

        try:
            status = await batch_processor.poll_batch(batch_id)
        except Exception as e:
            logger.warning(f"[batch_tools] poll_batch failed for {batch_id[:12]} (non-fatal): {e}")
            continue

        if status.get("status") != "ended":
            logger.debug(
                f"[batch_tools] Batch {batch_id[:12]} ({job_name}) still in progress: {status}"
            )
            continue

        try:
            results = await batch_processor.get_results(batch_id)
        except Exception as e:
            logger.error(f"[batch_tools] get_results failed for {batch_id[:12]} (non-fatal): {e}")
            continue

        written = 0
        for result in results:
            if result.get("type") != "success":
                logger.warning(
                    f"[batch_tools] Reflection request {result.get('custom_id')} "
                    f"errored: {result.get('error')}"
                )
                continue
            task_id = result.get("custom_id", "")
            reflection_text = (result.get("content") or "").strip()
            if not task_id or not reflection_text:
                continue
            try:
                found = log_reflection(task_id, reflection_text)
                if found:
                    written += 1
            except Exception as e:
                logger.warning(f"[batch_tools] log_reflection failed for {task_id} (non-fatal): {e}")

        try:
            batch_processor._mark_retrieved(batch_id)
        except Exception as e:
            logger.warning(f"[batch_tools] _mark_retrieved failed for {batch_id[:12]} (non-fatal): {e}")

        logger.info(
            f"[batch_tools] Batch '{job_name}' ({batch_id[:12]}...) processed: "
            f"{written}/{len(results)} reflections written."
        )


def _ensure_poller_scheduled() -> None:
    """
    Register the recurring 30-minute polling job with the raw APScheduler
    instance owned by TaskScheduler, if not already registered.

    Uses replace_existing=True so calling this multiple times is safe and
    idempotent — it just refreshes the same job.
    """
    if _scheduler is None:
        logger.warning("[batch_tools] Cannot schedule poller — scheduler not set.")
        return
    try:
        raw_scheduler = _scheduler._scheduler  # AsyncIOScheduler instance
        raw_scheduler.add_job(
            _poll_and_process_batches,
            trigger="interval",
            minutes=30,
            id=_POLL_JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        logger.info("[batch_tools] Batch result poller scheduled (every 30 minutes).")
    except Exception as e:
        logger.error(f"[batch_tools] Failed to schedule batch poller (non-fatal): {e}")


def register_batch_tools() -> None:
    """
    Register Phase 11.5b/11.5c batch tools with the global tool registry.

    Called once at startup from main.py, after the registry and scheduler
    are both initialised.
    """
    from agent_tools import register_tool

    # ── backfill_reflections ────────────────────────────────────────────
    async def _backfill_reflections() -> dict[str, Any]:
        """
        Tool handler: submit a batch job to generate reflections for every
        task in long_term.json that's missing one.

        Returns:
            {"success": bool, "tasks_queued": int, "batch_id": str,
             "estimated_cost_savings": "50%"}
        """
        try:
            import batch_processor
            from memory.long_term import load
        except Exception as e:
            return {"success": False, "error": f"Import failed: {e}"}

        try:
            data = load()
            tasks = data.get("tasks", [])
            missing = [
                t for t in tasks
                if not (t.get("reflection") or "").strip()
            ]

            if not missing:
                return {
                    "success": True,
                    "tasks_queued": 0,
                    "batch_id": "",
                    "estimated_cost_savings": "50%",
                    "message": "No tasks are missing reflections — nothing to backfill.",
                }

            requests = []
            for task in missing:
                task_id = task.get("id", "")
                if not task_id:
                    continue
                prompt = REFLECTION_PROMPT_TEMPLATE.format(
                    goal=task.get("goal", "")[:500],
                    outcome=task.get("outcome", "unknown"),
                    tools_used=", ".join(task.get("tools_used", [])) or "none",
                    duration=task.get("duration_seconds", 0),
                )
                requests.append({
                    "custom_id": task_id,
                    "params": {
                        "model": "claude-haiku-4-5",
                        "max_tokens": 300,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                })

            if not requests:
                return {
                    "success": True,
                    "tasks_queued": 0,
                    "batch_id": "",
                    "estimated_cost_savings": "50%",
                    "message": "No valid tasks with IDs to backfill.",
                }

            batch_id = await batch_processor.submit_batch(requests, job_name="reflection_backfill")

            # Schedule (or refresh) the recurring poller that will pick up
            # results and write them back via log_reflection().
            _ensure_poller_scheduled()

            logger.info(
                f"[batch_tools] backfill_reflections: queued {len(requests)} tasks, "
                f"batch_id={batch_id}"
            )
            return {
                "success": True,
                "tasks_queued": len(requests),
                "batch_id": batch_id,
                "estimated_cost_savings": "50%",
            }

        except Exception as e:
            logger.error(f"[batch_tools] backfill_reflections failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="backfill_reflections",
        description=(
            "Submit a batch job to generate missing reflections for past tasks in the "
            "episode journal, at 50% of normal API cost via the Anthropic Message "
            "Batches API. Finds every task in long_term.json without a reflection, "
            "submits one batch request per task, and schedules a background job that "
            "polls every 30 minutes and writes each reflection back automatically once "
            "the batch completes. This can take anywhere from minutes to a few hours "
            "depending on batch queue load — use list_batch_jobs() to check progress. "
            "Safe to call multiple times; tasks that already have reflections are skipped."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=_backfill_reflections,
        is_destructive=False,
    )

    # ── list_batch_jobs ──────────────────────────────────────────────────
    async def _list_batch_jobs() -> dict[str, Any]:
        """
        Tool handler: list all batches that haven't yet been retrieved.

        Returns:
            {"success": True, "pending_batches": [...], "count": N}
        """
        try:
            import batch_processor
            pending = batch_processor.list_pending_batches()
            return {"success": True, "pending_batches": pending, "count": len(pending)}
        except Exception as e:
            logger.error(f"[batch_tools] list_batch_jobs failed: {e}")
            return {"success": False, "error": str(e), "pending_batches": [], "count": 0}

    register_tool(
        name="list_batch_jobs",
        description=(
            "List all background batch jobs (e.g. reflection backfills) that have been "
            "submitted but not yet fully retrieved. Shows batch_id, job_name, status, "
            "and submitted_at for each. Use this to check on the progress of a call to "
            "backfill_reflections() or any other batch-processing job."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=_list_batch_jobs,
        is_destructive=False,
    )

    logger.info("[batch_tools] Phase 11.5c: batch tools registered (backfill_reflections, list_batch_jobs)")
