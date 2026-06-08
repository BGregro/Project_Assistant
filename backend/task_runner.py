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

Phase 3d additions:
  - Token estimation: _token_estimate tracks approximate context growth in
    characters / 4 (rough token proxy) after every tool dispatch.
  - Mid-task context compression: when _token_estimate > compression_threshold
    (config: context.compression_threshold, default 6000), completed steps
    are summarised by the local LLM, old tool_result messages are replaced
    with the summary, and the estimate is reset.
  - model_override parameter: passed down from AgentCore intent routing so
    the first Claude call uses claude-sonnet-4-6 for COMPLEX intents.

Concurrency model (see main.py for the other half):
  - The WebSocket handler launches run_task() via asyncio.create_task().
  - While run_task() is running, the WebSocket receive pump continues to
    dispatch incoming messages.
  - Inside run_task(), _cancel_event.is_set() is checked at the top of
    every iteration so cancellation is honoured at the earliest safe point.
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

# Number of recent tool exchanges to keep verbatim when compressing context.
# Everything older than this window is replaced by the summary.
_KEEP_RECENT_TOOL_EXCHANGES = 3


class TaskRunner:
    """
    Owns one agentic task at a time.  A new task can only start after the
    previous one has completed, been cancelled, or failed.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._cancel_event = asyncio.Event()
        self._message_queue: asyncio.Queue[str] = asyncio.Queue()
        self._current_task: dict | None = None
        self._task_file: Path = _TASK_FILE
        self._is_running: bool = False

        # Phase 3d: context compression state
        cfg = config or {}
        self._token_estimate: int = 0
        self._compression_threshold: int = (
            cfg.get("context", {}).get("compression_threshold", 6000)
        )

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
        model_override: str | None = None,  # Phase 3d: COMPLEX intent → sonnet
    ) -> str:
        """
        Run the agentic loop indefinitely (no hard iteration cap) until:
          - Claude returns stop_reason == "end_turn"  → task complete
          - _cancel_event is set                       → task cancelled
          - An unhandled exception is raised           → task failed

        Phase 3d: model_override is used only for the first Claude call (COMPLEX
        intent routing). Subsequent iterations fall back to agent.primary_model.

        This coroutine is meant to be launched with asyncio.create_task() so
        the WebSocket handler remains responsive while it executes.
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
        # Reset Phase 3d token estimate for this task
        self._token_estimate = 0
        # Drain any leftover messages from a previous session
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        step_counter = 0
        run_start_ms = _now_ms()

        # model_override is consumed on the first Claude call only
        _current_model_override = model_override

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
                    response = await agent._run_claude_once(
                        messages, system, model_override=_current_model_override
                    )
                    # Model override is only for the first call
                    _current_model_override = None
                except anthropic.APIConnectionError:
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

                        # Dispatch the tool (permission layer + compression live inside
                        # _execute_tool in agent_core)
                        result_msg = await agent._execute_tool(
                            tool_name=block.name,
                            tool_input=block.input,
                            tool_use_id=block.id,
                            send_event=send_event,
                            pending_confirmations=pending_confirmations,
                        )
                        tool_results.append(result_msg)

                        # ── Phase 3d: update token estimate ───────────
                        # result_msg["content"] is a JSON string of the (possibly
                        # compressed) result that Claude will see.
                        result_content = result_msg.get("content", "")
                        self._token_estimate += len(result_content) // 4

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

                    # Feed (compressed) tool results back into messages
                    messages.append({"role": "user", "content": tool_results})

                    # ── Phase 3d: context compression ─────────────────
                    # When the running token estimate crosses the threshold,
                    # summarise completed steps and prune old tool exchanges
                    # from the messages list to keep Claude's context lean.
                    if self._token_estimate > self._compression_threshold:
                        messages = await self._compress_context(
                            messages=messages,
                            agent=agent,
                            send_event=send_event,
                        )

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
    # Phase 3d: context compression helper
    # ------------------------------------------------------------------

    async def _compress_context(
        self,
        messages: list[dict],
        agent,
        send_event: Callable[[str, dict], Awaitable[None]],
    ) -> list[dict]:
        """
        Summarise completed task steps and prune old tool_result messages to
        prevent Claude's context from overflowing during long autonomous runs.

        Strategy:
          1. Collect completed step labels and tool names from self._current_task.
          2. Build a plain-text steps summary and call summarize_completed_steps().
          3. Find all tool_result messages in the messages list.
          4. Keep the most recent _KEEP_RECENT_TOOL_EXCHANGES tool exchanges verbatim.
          5. Remove the older tool_result messages and insert a synthetic user
             message containing the summary in their place.
          6. Reset self._token_estimate.

        Returns the (possibly pruned) messages list.
        """
        from agent_tools.local_llm import summarize_completed_steps

        steps = self._current_task.get("steps", [])
        if not steps:
            return messages

        # Build a readable steps summary for the local LLM
        step_lines = []
        for s in steps:
            status_marker = "✓" if s["status"] == "done" else "✗"
            step_lines.append(
                f"{status_marker} Step {s['step']}: {s['tool']} "
                f"({s['status']}, {s['elapsed_ms']}ms)"
            )
        steps_text = "\n".join(step_lines)

        summary = await summarize_completed_steps(
            steps_text=steps_text,
            model=agent.local_model,
            base_url=agent.ollama_url,
        )

        # Identify positions of tool_result messages in the list
        # A tool_result message looks like: {"role": "user", "content": [{"type": "tool_result", ...}]}
        # or the individual {"type": "tool_result", ...} dicts inside a user content list.
        # We target the outer user-role messages that contain only tool_result content blocks.
        tool_result_indices = []
        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list) and all(
                isinstance(c, dict) and c.get("type") == "tool_result"
                for c in content
            ):
                tool_result_indices.append(i)

        # Keep the most recent N tool exchanges; summarise everything older
        keep_count = _KEEP_RECENT_TOOL_EXCHANGES
        if len(tool_result_indices) <= keep_count:
            # Not enough old exchanges to make compression worthwhile yet
            self._token_estimate = 0
            return messages

        prune_indices = set(tool_result_indices[:-keep_count])
        insertion_point = min(prune_indices)

        # Build new messages list: everything before the first pruned index,
        # then the summary injection, then skip pruned messages, then keep the rest.
        new_messages = []
        summary_inserted = False
        for i, msg in enumerate(messages):
            if i in prune_indices:
                if not summary_inserted:
                    new_messages.append({
                        "role":    "user",
                        "content": (
                            f"[Context compressed — summary of completed steps:\n{summary}]"
                        ),
                    })
                    summary_inserted = True
                # Skip the pruned tool_result message
                continue
            new_messages.append(msg)

        removed_count = len(prune_indices)
        self._token_estimate = 0
        logger.info(
            f"[task_runner] Compressed context. "
            f"Removed {removed_count} tool_result message(s), added summary."
        )
        await send_event("status", {"text": "Context compressed — continuing task…"})

        return new_messages

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
