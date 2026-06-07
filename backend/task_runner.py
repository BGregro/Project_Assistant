"""
task_runner.py  —  Phase 3b: Long-Running Task Loop

Wraps the agentic loop in a resumable, cancellable task runner that:

  1. Persists task state to disk after every tool dispatch (checkpointing).
  2. Checks for cancellation between every tool call via asyncio.Event.
  3. Accepts mid-task user messages via asyncio.Queue without interrupting
     the running loop — they are injected as user turns at the next safe
     checkpoint boundary.
  4. Emits task_progress and task_stopped WebSocket events so the frontend
     can drive a live step timeline and Stop button.

Concurrency model (see main.py for the other half):
  - The WebSocket handler launches run_task() via asyncio.create_task().
    This schedules the coroutine on the event loop but yields control back
    to the caller immediately.
  - While run_task() is running, the WebSocket receive pump in main.py
    continues to dispatch incoming messages.  If the message type is
    "stop_task" it calls cancel(); if it's a regular "message" it calls
    inject_message() which puts the text into the queue.
  - Inside run_task(), _cancel_event.is_set() is checked at the top of
    every iteration (before any await), so cancellation is honoured at
    the earliest safe point — never mid-tool.
  - _message_queue.get_nowait() drains all queued user messages at the
    top of every iteration as well, appending them to the messages list
    before the next Claude API call.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable
from uuid import uuid4

import anthropic

logger = logging.getLogger(__name__)

# Path to the persisted task state file.
_TASK_FILE = Path(__file__).parent.parent / "memory" / "current_task.json"


class TaskRunner:
    """
    Owns one agentic task at a time.  A new task can only start after the
    previous one has completed, been cancelled, or failed.
    """

    def __init__(self) -> None:
        self._cancel_event = asyncio.Event()
        self._message_queue: asyncio.Queue[str] = asyncio.Queue()
        self._current_task: dict | None = None
        self._task_file: Path = _TASK_FILE
        self._is_running: bool = False

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Signal the running task to stop at the next checkpoint."""
        logger.info("[task_runner] Cancel requested.")
        self._cancel_event.set()

    async def inject_message(self, message: str) -> None:
        """Queue a user message to be picked up between tool calls."""
        await self._message_queue.put(message)
        logger.info(f"[task_runner] Message injected into queue: {message[:60]!r}")

    def is_running(self) -> bool:
        """Return True if a task is currently executing."""
        return self._is_running

    def load_last_task(self) -> dict | None:
        """
        Read the last persisted task state from disk.
        Returns None if no file exists or it cannot be parsed.
        """
        try:
            if self._task_file.exists():
                with open(self._task_file, encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"[task_runner] Could not load task file: {e}")
        return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run_task(
        self,
        initial_message: str,
        messages: list[dict],
        system: str,
        send_event: Callable[[str, dict], Awaitable[None]],
        pending_confirmations: dict,
        agent,               # AgentCore instance — duck-typed to avoid circular import
    ) -> str:
        """
        Run the agentic loop indefinitely (no hard iteration cap) until:
          - Claude returns stop_reason == "end_turn"  → task complete
          - _cancel_event is set                       → task cancelled
          - An unhandled exception is raised           → task failed

        This coroutine is meant to be launched with asyncio.create_task() so
        the WebSocket handler remains responsive while it executes.

        Args:
            initial_message:        The raw user text that triggered this task.
            messages:               The assembled message history for Claude.
            system:                 The system prompt string.
            send_event:             Async callback sending typed events to the browser.
            pending_confirmations:  Shared dict for the permission layer.
            agent:                  AgentCore instance (provides _run_claude_once
                                    and _execute_tool).

        Returns:
            The final text reply, or a short status string on cancellation/error.
        """
        # ── Initialise task state ──────────────────────────────────────
        task_id = str(uuid4())
        start_ts = datetime.now(timezone.utc).isoformat()
        self._current_task = {
            "id":              task_id,
            "started_at":      start_ts,
            "initial_message": initial_message,
            "steps":           [],
            "status":          "running",
        }
        self._save_task()
        self._is_running = True
        self._cancel_event.clear()         # reset from any previous run
        # Drain any leftover messages from a previous session
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        step_counter = 0
        run_start_ms = _now_ms()

        await send_event("task_started", {"task_id": task_id})
        await send_event("task_progress", {
            "step":       step_counter,
            "label":      "Starting…",
            "status":     "running",
            "elapsed_ms": 0,
        })

        # ── Agentic loop ───────────────────────────────────────────────
        try:
            while True:

                # ── 1. Check for cancellation ──────────────────────────
                if self._cancel_event.is_set():
                    logger.info("[task_runner] Task cancelled by user.")
                    self._current_task["status"] = "cancelled"
                    self._save_task()
                    await send_event("task_stopped", {"reason": "user_cancelled"})
                    return "Task cancelled."

                # ── 2. Drain injected user messages ────────────────────
                injected_count = 0
                while True:
                    try:
                        queued_msg = self._message_queue.get_nowait()
                        messages.append({"role": "user", "content": queued_msg})
                        step_counter += 1
                        await send_event("task_progress", {
                            "step":       step_counter,
                            "label":      "User instruction received",
                            "status":     "done",
                            "elapsed_ms": _elapsed_ms(run_start_ms),
                        })
                        logger.info(
                            f"[task_runner] Injected user message: {queued_msg[:60]!r}"
                        )
                        injected_count += 1
                    except asyncio.QueueEmpty:
                        break

                # ── 3. Call Claude (single API round-trip) ─────────────
                step_counter += 1
                step_start_ms = _now_ms()

                await send_event("status", {"text": "Thinking…"})
                await send_event("task_progress", {
                    "step":       step_counter,
                    "label":      "Thinking…",
                    "status":     "running",
                    "elapsed_ms": _elapsed_ms(run_start_ms),
                })

                try:
                    response = await agent._run_claude_once(messages, system)
                except anthropic.APIConnectionError:
                    # Let agent handle fallback — re-raise so the except below sees it
                    raise
                except anthropic.AuthenticationError:
                    await send_event("error", {
                        "text": "Invalid or missing ANTHROPIC_API_KEY."
                    })
                    raise
                except anthropic.RateLimitError:
                    await send_event("error", {
                        "text": "Claude API rate limit hit. Try again shortly."
                    })
                    raise
                except anthropic.APIError as e:
                    await send_event("error", {"text": f"Claude API error: {e}"})
                    raise

                # ── 4a. end_turn — task complete ───────────────────────
                if response.stop_reason == "end_turn":
                    final_text = _extract_text(response)
                    await send_event("message", {"text": final_text, "source": "claude"})

                    await send_event("task_progress", {
                        "step":       step_counter,
                        "label":      "Done",
                        "status":     "done",
                        "elapsed_ms": _elapsed_ms(step_start_ms),
                    })

                    self._current_task["status"] = "complete"
                    self._save_task()
                    await send_event("task_stopped", {"reason": "complete"})
                    logger.info(
                        f"[task_runner] Task {task_id} complete in "
                        f"{_elapsed_ms(run_start_ms)}ms."
                    )
                    return final_text

                # ── 4b. tool_use — dispatch tools then loop ────────────
                elif response.stop_reason == "tool_use":
                    tool_use_blocks = [
                        b for b in response.content if b.type == "tool_use"
                    ]
                    # Append assistant turn before dispatching tools
                    messages.append({"role": "assistant", "content": response.content})

                    tool_results = []
                    for block in tool_use_blocks:
                        # Mark thinking step done
                        await send_event("task_progress", {
                            "step":       step_counter,
                            "label":      "Thinking…",
                            "status":     "done",
                            "elapsed_ms": _elapsed_ms(step_start_ms),
                        })

                        # New step for this tool
                        step_counter += 1
                        tool_step_start = _now_ms()
                        await send_event("task_progress", {
                            "step":       step_counter,
                            "label":      block.name,
                            "status":     "running",
                            "elapsed_ms": 0,
                        })

                        # Dispatch the tool (permission layer lives inside _execute_tool)
                        result_msg = await agent._execute_tool(
                            tool_name=block.name,
                            tool_input=block.input,
                            tool_use_id=block.id,
                            send_event=send_event,
                            pending_confirmations=pending_confirmations,
                        )
                        tool_results.append(result_msg)

                        # Parse success flag from the serialised result
                        success = _result_success(result_msg)
                        tool_status = "done" if success else "failed"

                        await send_event("task_progress", {
                            "step":       step_counter,
                            "label":      block.name,
                            "status":     tool_status,
                            "elapsed_ms": _elapsed_ms(tool_step_start),
                        })

                        # Checkpoint: persist step record to disk
                        self._current_task["steps"].append({
                            "step":       step_counter,
                            "tool":       block.name,
                            "status":     tool_status,
                            "elapsed_ms": _elapsed_ms(tool_step_start),
                        })
                        self._save_task()

                    # Feed tool results back into messages for the next iteration
                    messages.append({"role": "user", "content": tool_results})

                # ── 4c. unexpected stop_reason ─────────────────────────
                else:
                    logger.warning(
                        f"[task_runner] Unexpected stop_reason: "
                        f"{response.stop_reason!r} — stopping task."
                    )
                    partial = _extract_text(response)
                    if partial:
                        await send_event("message", {"text": partial, "source": "claude"})
                    break

        except Exception as e:
            logger.exception("[task_runner] Task raised an unhandled exception")
            self._current_task["status"] = "failed"
            self._current_task["error"] = str(e)
            self._save_task()
            await send_event("task_stopped", {"reason": "error", "error": str(e)})
            return f"Task failed: {e}"

        finally:
            # Always clear the running flag, even on unexpected exits
            self._is_running = False

        # Reached if an unknown stop_reason broke the loop without exception
        self._current_task["status"] = "failed"
        self._save_task()
        await send_event("task_stopped", {
            "reason": "error",
            "error":  "Unexpected stop_reason from Claude.",
        })
        return "Task ended unexpectedly."

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_task(self) -> None:
        """Write the current task state to disk as JSON."""
        if self._current_task is None:
            return
        try:
            self._task_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._task_file, "w", encoding="utf-8") as f:
                json.dump(self._current_task, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[task_runner] Could not save task file: {e}")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    """Current time in milliseconds."""
    return int(time.monotonic() * 1000)


def _elapsed_ms(since_ms: int) -> int:
    """Milliseconds elapsed since `since_ms`."""
    return _now_ms() - since_ms


def _extract_text(response: anthropic.types.Message) -> str:
    """Pull all TextBlock content from a Claude response into one string."""
    parts = [block.text for block in response.content if hasattr(block, "text")]
    return "\n".join(parts).strip()


def _result_success(tool_result_msg: dict) -> bool:
    """
    Parse the `success` flag out of a tool result message.

    tool_result_msg looks like:
        { "type": "tool_result", "tool_use_id": "...", "content": "<json>" }
    """
    try:
        payload = json.loads(tool_result_msg.get("content", "{}"))
        return bool(payload.get("success", True))
    except (json.JSONDecodeError, AttributeError):
        return True   # assume success if we can't parse
