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

Phase 4a additions:
  - _project_manifest: dict | None = None tracks the active scaffold when a
    scaffold_project tool call succeeds mid-task.  A compact summary is
    appended to the live system prompt so Claude retains project structure
    without re-reading the full scaffold JSON every iteration.
  - _project_manifest is reset to None at the start of every new task so
    stale project context never bleeds into unrelated runs.
  - The WebSocket handler launches run_task() via asyncio.create_task().
  - While run_task() is running, the WebSocket receive pump continues to
    dispatch incoming messages.
  - Inside run_task(), _cancel_event.is_set() is checked at the top of
    every iteration so cancellation is honoured at the earliest safe point.

Parallel tool dispatch (improvement):
  - When Claude returns multiple tool_use blocks in one response, all
    approved tool calls now run concurrently via asyncio.gather instead
    of sequentially.
  - Per-block exceptions are caught and converted to synthetic error
    results so one failing tool never cancels its siblings.
  - tool_results list order matches the original tool_use_blocks order
    (asyncio.gather preserves insertion order).
  - step_counter increments are pre-assigned before gather so each block
    gets a unique step number without races.
  - Step records are appended to self._current_task["steps"] after all
    tools complete (still in original order).
  - Consecutive failure detection and scaffold detection run after gather,
    once all results are known.
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

# Phase 12b: performance metrics — imported lazily inside functions to avoid
# circular imports at module load time.  See record_tool_call / record_task calls.

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

        # Phase 4a: active project manifest — set when scaffold_project succeeds
        # during a task run.  Keeps project context alive across iterations
        # without re-sending the full scaffold every turn.
        self._project_manifest: dict | None = None

        # Phase 12a: reference to AgentCore — used by _generate_reflection to
        # read ollama_url and local_agent_model without a circular import.
        # Set by main.py after both objects are instantiated.
        self._agent_ref = None

        # Phase 6a: pending mid-task questions.
        # Keyed by question_id; each value holds the asyncio.Event and the
        # answer string populated by answer_question() when the user responds.
        self._pending_questions: dict[str, dict] = {}

        # Improvement 5: timestamp of the last rate-limit error.
        # Used to pace requests for 2 minutes after any 429 to avoid
        # hitting the limit again immediately after a retry.
        self._last_rate_limit_ts: float = 0.0

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

    async def ask_user(
        self,
        question: str,
        question_id: str,
        send_event,
    ) -> str:
        """
        Phase 6a: Pause the task loop and ask the user a question.

        Emits a 'user_question' WebSocket event so the frontend can show the
        question modal.  Suspends until the user submits an answer or the
        10-minute timeout expires.

        Args:
            question:    The question text to display.
            question_id: Unique ID generated by the ask_user tool.
            send_event:  The active WebSocket send_event coroutine.

        Returns:
            The user's answer as a string, or "" on timeout.
        """
        event = asyncio.Event()
        self._pending_questions[question_id] = {"event": event, "answer": ""}

        await send_event("user_question", {
            "question_id": question_id,
            "question":    question,
        })
        logger.info(
            f"[task_runner] Waiting for user answer to question '{question_id[:8]}'..."
        )

        try:
            await asyncio.wait_for(event.wait(), timeout=600.0)
            answer = self._pending_questions.pop(question_id, {}).get("answer", "")
            logger.info(
                f"[task_runner] Got answer for '{question_id[:8]}': {answer[:60]!r}"
            )
            return answer
        except asyncio.TimeoutError:
            self._pending_questions.pop(question_id, None)
            logger.warning(
                f"[task_runner] Question '{question_id[:8]}' timed out after 10 minutes."
            )
            return ""

    def answer_question(self, question_id: str, answer: str) -> None:
        """
        Phase 6a: Called by the main.py WebSocket handler when the user
        submits an answer via the question modal.

        Sets the answer and releases the asyncio.Event so ask_user() resumes.
        """
        if question_id in self._pending_questions:
            self._pending_questions[question_id]["answer"] = answer
            self._pending_questions[question_id]["event"].set()
            logger.info(
                f"[task_runner] Answer received for question '{question_id[:8]}': "
                f"{answer[:60]!r}"
            )
        else:
            logger.warning(
                f"[task_runner] answer_question: unknown question_id '{question_id[:8]}'"
            )

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
        _start_time = time.time()   # Phase 3f: wall-clock start for duration logging
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
        # Reset Phase 4a project manifest for this task
        self._project_manifest = None
        # Reset consecutive failure tracker for this task
        self._consecutive_failures: dict[str, int] = {}
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
            _iteration = 0
            _max_iterations = getattr(agent, "max_iterations", 30)
            while True:
                _iteration += 1
                if _iteration > _max_iterations:
                    msg = (
                        f"Agent reached the maximum number of iterations "
                        f"({_max_iterations}) without completing the task."
                    )
                    logger.warning(f"[task_runner] {msg}")
                    await send_event("error", {"text": msg})
                    self._current_task["status"] = "failed"
                    self._save_task()
                    await send_event("task_stopped", {"reason": "error", "error": msg})
                    return msg

                # Improvement 5: post-rate-limit pacing.
                # After any 429, slow down for 2 minutes to avoid immediately
                # hitting the rate limit again on the very next request.
                if self._last_rate_limit_ts and (time.time() - self._last_rate_limit_ts) < 120:
                    await asyncio.sleep(2.0)

                # ── 0. Sanitize message history ────────────────────────
                # Delegate to agent._sanitize_messages() — the canonical
                # implementation lives on AgentCore (Fix 1) so the repair
                # logic is defined in exactly one place.  The local copy
                # below is kept as a fallback only for the edge case where
                # agent is not yet wired (shouldn't happen in practice).
                messages = agent._sanitize_messages(messages)

                # ── 1. Check for cancellation ──────────────────────────
                if self._cancel_event.is_set():
                    logger.info("[task_runner] Task cancelled by user.")
                    self._current_task["status"] = "cancelled"
                    self._save_task()
                    await send_event("task_stopped", {"reason": "user_cancelled"})
                    self._is_running = False
                    # Phase 3f: log cancellation to long-term memory
                    try:
                        from memory.long_term import log_task as _log_task
                        _log_task(
                            goal=initial_message,
                            outcome="failure",
                            summary=f"Cancelled by user after {len(self._current_task.get('steps', []))} steps.",
                            tools_used=list({s["tool"] for s in self._current_task.get("steps", [])}),
                            duration_seconds=int(time.time() - _start_time),
                        )
                    except Exception as _lt_e:
                        logger.warning(f"[task_runner] Long-term log (cancel) failed (non-fatal): {_lt_e}")
                    # Phase 12b: record cancelled task metrics
                    try:
                        from memory import performance as _perf
                        _perf.record_task(
                            goal=initial_message,
                            outcome="cancelled",
                            duration_seconds=int(time.time() - _start_time),
                            tools_used=list({s["tool"] for s in self._current_task.get("steps", [])}),
                        )
                    except Exception as _perf_e:
                        logger.debug(f"[task_runner] Metrics record_task (cancel) failed (non-fatal): {_perf_e}")
                    return "Task cancelled."

                # ── 2. Drain injected user messages ────────────────────
                while True:
                    try:
                        queued_msg = self._message_queue.get_nowait()
                        logger.info(
                            f"[task_runner] Injected user message: {queued_msg[:60]!r}"
                        )

                        # Notify the UI which queued message is now being processed
                        await send_event("queued_message_active", {
                            "content": queued_msg,
                            "preview": queued_msg[:80],
                        })

                        messages.append({"role": "user", "content": queued_msg})
                        step_counter += 1
                        await send_event("task_progress", {
                            "step":       step_counter,
                            "label":      "User instruction received",
                            "status":     "done",
                            "elapsed_ms": _elapsed_ms(run_start_ms),
                        })
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
                    self._last_rate_limit_ts = time.time()
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
                    # Mark not-running before cleanup so any race-window messages can be re-queued
                    self._is_running = False
                    # Phase 3f: persist outcome to long-term memory
                    _logged_task_id: str = ""
                    try:
                        from memory.long_term import log_task as _log_task
                        _logged_task_id = _log_task(
                            goal=initial_message,
                            outcome="success",
                            summary=f"Completed in {len(self._current_task['steps'])} steps.",
                            tools_used=list({s["tool"] for s in self._current_task["steps"]}),
                            duration_seconds=int(time.time() - _start_time),
                        )
                    except Exception as _lt_e:
                        logger.warning(f"[task_runner] Long-term log failed (non-fatal): {_lt_e}")

                    # Phase 12b: record task-level metrics
                    try:
                        from memory import performance as _perf
                        _perf.record_task(
                            goal=initial_message,
                            outcome="success",
                            duration_seconds=int(time.time() - _start_time),
                            tools_used=list({s["tool"] for s in self._current_task["steps"]}),
                        )
                    except Exception as _perf_e:
                        logger.debug(f"[task_runner] Metrics record_task failed (non-fatal): {_perf_e}")

                    # Phase 12a: fire background reflection — never blocks the user
                    if _logged_task_id:
                        asyncio.get_event_loop().create_task(
                            self._generate_reflection(
                                task_id=_logged_task_id,
                                goal=initial_message,
                                tools_used=list({s["tool"] for s in self._current_task["steps"]}),
                                outcome="success",
                                duration_seconds=int(time.time() - _start_time),
                                send_event=send_event,
                            )
                        )

                    # Phase 9b: fire-and-forget email notification on completion
                    try:
                        import json as _json
                        import pathlib as _pathlib
                        _cfg_path = _pathlib.Path(__file__).parent.parent / "config.json"
                        _cfg = _json.loads(_cfg_path.read_text()) if _cfg_path.exists() else {}
                        if _cfg.get("email", {}).get("enabled", False):
                            from agent_tools.notification_tool import send_email as _send_email
                            _subject = f"Task complete: {initial_message[:50]}"
                            _body = (
                                f"Your agent finished a task.\n\n"
                                f"Goal: {initial_message}\n"
                                f"Steps: {len(self._current_task['steps'])}\n"
                                f"Duration: {int(time.time() - _start_time)}s\n\n"
                                f"Summary: {final_text[:300]}"
                            )
                            asyncio.ensure_future(_send_email(_subject, _body))
                    except Exception as _notif_e:
                        logger.debug(f"[task_runner] Email notification skipped (non-fatal): {_notif_e}")

                    # Drain any messages that arrived during the race window between
                    # task completion and _is_running = False being visible to main.py
                    leftover = []
                    while not self._message_queue.empty():
                        try:
                            leftover.append(self._message_queue.get_nowait())
                        except asyncio.QueueEmpty:
                            break
                    if leftover and send_event:
                        await send_event("requeue_message", {"content": leftover[0]})
                        logger.info(
                            f"[task_runner] Re-queued {len(leftover)} race-condition message(s)"
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
                    if not hasattr(self, '_consecutive_failures'):
                        self._consecutive_failures: dict[str, int] = {}

                    # ── Pre-assign step numbers ───────────────────────────
                    # Each block needs a unique step number.  We allocate them
                    # all up-front so concurrent coroutines never race on
                    # step_counter.  The "thinking…" step was already counted
                    # above; we increment here for each tool block.
                    block_step_numbers: list[int] = []
                    for _ in tool_use_blocks:
                        step_counter += 1
                        block_step_numbers.append(step_counter)

                    # ── Mark thinking step done (once, before tools fire) ─
                    await send_event("task_progress", {
                        "step":       step_counter - len(tool_use_blocks),
                        "label":      "Thinking…",
                        "status":     "done",
                        "elapsed_ms": _elapsed_ms(step_start_ms),
                    })

                    # ── Parallel dispatch ─────────────────────────────────
                    # Each coroutine handles one tool_use block independently:
                    # sends its own "running" / "done"/"failed" progress events
                    # and returns (result_msg, step_record).
                    # asyncio.gather preserves the order of return values so
                    # tool_results always matches the original block order.

                    async def _dispatch_one(
                        block,
                        assigned_step: int,
                    ) -> tuple[dict, dict]:
                        """
                        Dispatch a single tool call and return
                        (result_msg, step_record).

                        Exceptions are caught here so that one failing tool
                        never cancels its siblings via gather's default
                        exception propagation.
                        """
                        tool_step_start = _now_ms()
                        await send_event("task_progress", {
                            "step":       assigned_step,
                            "label":      block.name,
                            "status":     "running",
                            "elapsed_ms": 0,
                        })

                        try:
                            result_msg = await agent._execute_tool(
                                tool_name=block.name,
                                tool_input=block.input,
                                tool_use_id=block.id,
                                send_event=send_event,
                                pending_confirmations=pending_confirmations,
                            )
                        except Exception as _tool_exc:
                            # Unexpected exception from the tool itself — wrap it
                            # in a synthetic error result so the sibling tools and
                            # the agent loop continue normally.
                            logger.exception(
                                f"[task_runner] Unexpected exception from tool "
                                f"'{block.name}' during parallel dispatch"
                            )
                            import json as _j
                            result_msg = {
                                "type":        "tool_result",
                                "tool_use_id": block.id,
                                "content":     _j.dumps({
                                    "success": False,
                                    "error":   (
                                        f"Tool raised an unexpected exception: "
                                        f"{_tool_exc}"
                                    ),
                                }),
                            }

                        elapsed = _elapsed_ms(tool_step_start)
                        success = _result_success(result_msg)
                        tool_status = "done" if success else "failed"

                        # Phase 12b: record per-tool metric (fire-and-forget, non-fatal)
                        try:
                            from memory import performance as _perf
                            _failure_reason = ""
                            if not success:
                                try:
                                    _payload = json.loads(result_msg.get("content", "{}"))
                                    _failure_reason = str(_payload.get("error", ""))[:200]
                                except Exception:
                                    pass
                            _perf.record_tool_call(
                                tool_name=block.name,
                                success=success,
                                duration_ms=elapsed,
                                failure_reason=_failure_reason,
                            )
                        except Exception as _perf_e:
                            logger.debug(f"[task_runner] Metrics record_tool_call failed (non-fatal): {_perf_e}")

                        await send_event("task_progress", {
                            "step":       assigned_step,
                            "label":      block.name,
                            "status":     tool_status,
                            "elapsed_ms": elapsed,
                        })

                        step_record = {
                            "step":       assigned_step,
                            "tool":       block.name,
                            "status":     tool_status,
                            "elapsed_ms": elapsed,
                        }
                        return result_msg, step_record

                    # Fire all tool dispatches concurrently; results arrive in
                    # the same order as tool_use_blocks (gather guarantees this).
                    dispatch_results: list[tuple[dict, dict]] = await asyncio.gather(
                        *[
                            _dispatch_one(block, step_num)
                            for block, step_num in zip(tool_use_blocks, block_step_numbers)
                        ]
                    )

                    # ── Unpack gather results ─────────────────────────────
                    step_records: list[dict] = []
                    for (result_msg, step_record), block in zip(
                        dispatch_results, tool_use_blocks
                    ):
                        tool_results.append(result_msg)
                        step_records.append(step_record)

                        # ── Phase 4a: detect successful scaffold_project call ──
                        # When scaffold_project succeeds, store the manifest and
                        # inject a compact system-prompt addon so Claude knows
                        # the active project without re-reading the full scaffold.
                        if block.name == "scaffold_project":
                            try:
                                result_payload = json.loads(result_msg.get("content", "{}"))
                                if result_payload.get("success") and result_payload.get("scaffold"):
                                    scaffold = result_payload["scaffold"]
                                    self._project_manifest = scaffold
                                    proj_name  = scaffold.get("name", "unknown")
                                    file_count = result_payload.get("file_count", 0)
                                    system += (
                                        f"\n\nActive project: {proj_name} — "
                                        f"{file_count} files to implement. "
                                        f"Remaining: see outputs/{proj_name}/scaffold.json"
                                    )
                                    logger.info(
                                        f"[task_runner] Project manifest set: '{proj_name}' "
                                        f"({file_count} files)"
                                    )
                            except Exception as _pm_e:
                                logger.debug(
                                    f"[task_runner] Could not parse scaffold result (non-fatal): {_pm_e}"
                                )

                        # Token estimate update
                        result_content = result_msg.get("content", "")
                        self._token_estimate += len(result_content) // 4

                        # ── Consecutive failure guard ──────────────────────
                        # Runs after gather so we see the definitive success/fail
                        # status of every block before deciding whether to inject
                        # a stop hint.
                        success = _result_success(result_msg)
                        if not success:
                            self._consecutive_failures[block.name] = (
                                self._consecutive_failures.get(block.name, 0) + 1
                            )
                            if self._consecutive_failures[block.name] >= 3:
                                logger.warning(
                                    f"[task_runner] Tool '{block.name}' failed "
                                    f"{self._consecutive_failures[block.name]} times "
                                    f"in a row — injecting stop hint."
                                )
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        f"The tool '{block.name}' has failed "
                                        f"{self._consecutive_failures[block.name]} times in a row. "
                                        "Please stop retrying it. Either use a different approach "
                                        "or end the task with an explanation of what went wrong."
                                    ),
                                })
                                self._consecutive_failures[block.name] = 0
                        else:
                            self._consecutive_failures[block.name] = 0

                    # ── Persist all step records (in order) ───────────────
                    self._current_task["steps"].extend(step_records)
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

                # ── 4c. max_tokens — Claude was cut off mid-response ─────────
                elif response.stop_reason == "max_tokens":
                    # Not fatal. Append whatever was generated as a partial
                    # assistant turn and inject a continuation prompt so the
                    # loop resumes from where it left off.
                    partial = _extract_text(response)
                    logger.warning(
                        f"[task_runner] max_tokens at step {step_counter}. "
                        f"Partial text: {len(partial)} chars. Continuing…"
                    )
                    await send_event("status", {
                        "text": "Response limit reached — continuing…"
                    })
                    # Append partial output as a complete assistant turn
                    if partial:
                        messages.append({"role": "assistant", "content": partial})
                    else:
                        messages.append({"role": "assistant", "content": response.content})
                    # Inject continuation prompt
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was cut off. "
                            "Continue exactly where you stopped without repeating yourself."
                        ),
                    })
                    await send_event("task_progress", {
                        "step":       step_counter,
                        "label":      "Continuing… (response limit)",
                        "status":     "running",
                        "elapsed_ms": _elapsed_ms(step_start_ms),
                    })
                    # Do NOT break — loop continues

                # ── 4d. unexpected stop_reason ─────────────────────
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
            # Set running flag before cleanup so new messages aren't silently queued
            self._is_running = False
            # Phase 3f: log failure to long-term memory
            try:
                from memory.long_term import log_task as _log_task
                _log_task(
                    goal=initial_message,
                    outcome="failure",
                    summary=f"Failed after {len(self._current_task.get('steps', []))} steps: {str(e)[:120]}",
                    tools_used=list({s["tool"] for s in self._current_task.get("steps", [])}),
                    duration_seconds=int(time.time() - _start_time),
                )
            except Exception as _lt_e:
                logger.warning(f"[task_runner] Long-term log failed (non-fatal): {_lt_e}")
            # Phase 12b: record failed task metrics
            try:
                from memory import performance as _perf
                _perf.record_task(
                    goal=initial_message,
                    outcome="failure",
                    duration_seconds=int(time.time() - _start_time),
                    tools_used=list({s["tool"] for s in self._current_task.get("steps", [])}),
                )
            except Exception as _perf_e:
                logger.debug(f"[task_runner] Metrics record_task (failure) failed (non-fatal): {_perf_e}")
            return f"Task failed: {e}"

        finally:
            # Guard: ensure _is_running is always cleared even on unexpected exits.
            # In normal (complete/cancelled/failed) paths it's already set above;
            # this catches any edge case we haven't accounted for.
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
    # Pre-flight history sanitizer
    # ------------------------------------------------------------------

    def _sanitize_messages(self, messages: list) -> list:
        """
        Pre-flight check: ensure no orphaned tool_use blocks exist in the message
        history before sending to the Anthropic API.

        When a rate-limit (429) or network error interrupts a task mid-response,
        the last assistant message may contain tool_use blocks without a matching
        user tool_result message. Every subsequent API call will fail with 400
        until this is corrected.

        Fix: if the last assistant message has tool_use blocks and is NOT followed
        by a user tool_result message, inject synthetic tool_result messages so the
        history is valid again.
        """
        if not messages:
            return messages

        last = messages[-1]
        if last.get("role") != "assistant":
            return messages

        content = last.get("content", [])
        if not isinstance(content, list):
            return messages

        tool_use_blocks = [
            b for b in content
            if (isinstance(b, dict) and b.get("type") == "tool_use")
            or (hasattr(b, "type") and b.type == "tool_use")
        ]

        if not tool_use_blocks:
            return messages

        # Orphaned tool_use detected — inject synthetic error results
        logger.warning(
            f"[task_runner] Detected {len(tool_use_blocks)} orphaned tool_use block(s) "
            "from interrupted request — injecting synthetic tool_results to repair history."
        )
        synthetic_results = []
        for b in tool_use_blocks:
            tool_id = b["id"] if isinstance(b, dict) else b.id
            synthetic_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": "Tool call was interrupted by a network or rate-limit error. Please retry the operation.",
            })

        repaired = list(messages) + [{
            "role": "user",
            "content": synthetic_results,
        }]
        logger.info(f"[task_runner] History repaired — injected {len(synthetic_results)} synthetic tool_result(s).")
        return repaired

    # ------------------------------------------------------------------
    # Phase 12a: background reflection generator
    # ------------------------------------------------------------------

    async def _generate_reflection(
        self,
        task_id: str,
        goal: str,
        tools_used: list,
        outcome: str,
        duration_seconds: int,
        send_event,
    ) -> None:
        """
        Generate a structured reflection on the completed task using qwen3:14b.

        Runs as a background fire-and-forget coroutine — any failure is logged
        and silently swallowed. Never blocks the user or the main task loop.

        The reflection is stored in the task entry's 'reflection' field via
        long_term.log_reflection(). It becomes the raw material for Phase 14
        (pattern detection) and Phase 17 (strategy evolution).

        Args:
            task_id:          UUID of the just-logged task entry.
            goal:             The original user message / task description.
            tools_used:       Unique set of tool names used during the task.
            outcome:          "success", "failure", or "partial".
            duration_seconds: Wall-clock seconds the task ran for.
            send_event:       Active WebSocket send_event (used for optional debug status).
        """
        try:
            from agent_tools.local_llm import local_llm_call
            from memory.long_term import log_reflection

            prompt = (
                f"A task just completed. Generate a brief 2-3 sentence reflection.\n\n"
                f"Goal: {goal}\n"
                f"Outcome: {outcome}\n"
                f"Tools used: {', '.join(tools_used) if tools_used else 'none'}\n"
                f"Duration: {duration_seconds}s\n\n"
                "Reflect on: what worked well, what could have been done better, "
                "and what you would do differently next time. "
                "Be specific and honest. Write in first person ('I used...', 'I should have...'). "
                "2-3 sentences maximum."
            )

            # Read model config from agent reference (set by main.py); fall back to defaults.
            local_model = (
                getattr(self._agent_ref, "local_agent_model", "qwen3:14b")
                if self._agent_ref else "qwen3:14b"
            )
            ollama_url = (
                getattr(self._agent_ref, "ollama_url", "http://localhost:11434")
                if self._agent_ref else "http://localhost:11434"
            )

            reflection = await local_llm_call(prompt, local_model, ollama_url)
            reflection = reflection.strip() if reflection else ""

            if reflection and len(reflection) > 10:
                success = log_reflection(task_id, reflection)
                if success:
                    logger.info(
                        f"[task_runner] Phase 12a: reflection stored for task {task_id[:8]}..."
                    )
                else:
                    logger.warning(
                        f"[task_runner] Phase 12a: log_reflection returned False for {task_id[:8]}"
                    )
            else:
                logger.debug("[task_runner] Phase 12a: reflection was empty — skipping")

        except Exception as e:
            # Never propagate — background task failure must not affect the user
            logger.debug(f"[task_runner] Phase 12a: background reflection failed (non-fatal): {e}")

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
        Summarise completed task steps and prune old tool exchanges to prevent
        Claude's context window from overflowing during long autonomous runs.

        CRITICAL — Anthropic API constraint:
        Every assistant turn that contains tool_use blocks MUST be immediately
        followed by a user turn containing the matching tool_result blocks.
        Pruning only the tool_result half orphans the tool_use blocks and causes
        a 400 Bad Request on the very next API call.  This method therefore
        always locates and removes COMPLETE PAIRS (assistant + user) together.

        CRITICAL — timeout/failure guard:
        summarize_completed_steps() returns the raw steps_text unchanged when
        the local LLM is offline or times out.  If we pruned messages without
        a real summary we would silently corrupt the history.  We detect this
        by comparing summary == steps_text and bail out (reset estimate only)
        rather than corrupt the message sequence.
        """
        from agent_tools.local_llm import summarize_completed_steps

        steps = self._current_task.get("steps", [])
        if not steps:
            return messages

        # ── Build steps text for summarization ───────────────────────────────
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

        # ── Guard: bail if the summarizer failed ──────────────────────────────
        # summarize_completed_steps() returns steps_text unchanged on timeout,
        # Ollama offline, or any other error.  Proceeding would prune the
        # conversation without any useful summary, corrupting message history.
        if not summary or summary == steps_text:
            logger.warning(
                "[task_runner] Skipping context compression — summarizer returned "
                "no usable output (local LLM offline or timed out). "
                "Resetting token estimate to avoid immediate re-trigger."
            )
            self._token_estimate = 0
            return messages

        # ── Identify complete (assistant, user) tool exchange pairs ───────────
        # A valid pair is two adjacent messages where:
        #   messages[i]   role="assistant"  content contains ≥1 tool_use block
        #   messages[i+1] role="user"       content contains only tool_result blocks
        # Both halves MUST be pruned together — never one without the other.
        exchange_pairs: list[tuple[int, int]] = []
        i = 0
        while i < len(messages) - 1:
            msg      = messages[i]
            next_msg = messages[i + 1]

            # Detect tool_use assistant turn.
            # Handles both live SDK objects (block.type) and dict history entries.
            is_tool_use_turn = (
                msg.get("role") == "assistant"
                and isinstance(msg.get("content"), list)
                and any(
                    (isinstance(b, dict) and b.get("type") == "tool_use")
                    or (hasattr(b, "type") and b.type == "tool_use")
                    for b in msg["content"]
                )
            )

            # Detect tool_result user turn immediately following.
            next_content = next_msg.get("content")
            is_tool_result_turn = (
                next_msg.get("role") == "user"
                and isinstance(next_content, list)
                and len(next_content) > 0
                and all(
                    isinstance(c, dict) and c.get("type") == "tool_result"
                    for c in next_content
                )
            )

            if is_tool_use_turn and is_tool_result_turn:
                exchange_pairs.append((i, i + 1))
                i += 2   # both halves consumed — skip past the complete pair
            else:
                i += 1

        # ── Keep the most recent N pairs; prune everything older ──────────────
        keep_count = _KEEP_RECENT_TOOL_EXCHANGES
        if len(exchange_pairs) <= keep_count:
            # Not enough completed exchanges to compress yet
            self._token_estimate = 0
            return messages

        pairs_to_prune = exchange_pairs[:-keep_count]
        prune_indices: set[int] = set()
        for assistant_idx, result_idx in pairs_to_prune:
            prune_indices.add(assistant_idx)
            prune_indices.add(result_idx)

        # ── Rebuild message list ──────────────────────────────────────────────
        # Insert the summary placeholder as role="assistant" at the position of
        # the first pruned assistant turn.  Using role="assistant" prevents two
        # consecutive user messages when compression fires multiple times, which
        # would also cause a 400.
        first_pruned = min(prune_indices)
        new_messages: list[dict] = []
        summary_inserted = False

        for idx, msg in enumerate(messages):
            if idx in prune_indices:
                if not summary_inserted and idx == first_pruned:
                    new_messages.append({
                        "role":    "assistant",
                        "content": (
                            f"[Context compressed — summary of completed steps:\n{summary}]"
                        ),
                    })
                    summary_inserted = True
                # Drop this half of the pruned pair — never add it
                continue
            new_messages.append(msg)

        removed_pairs = len(pairs_to_prune)
        self._token_estimate = 0
        logger.info(
            f"[task_runner] Compressed context: removed {removed_pairs} tool exchange "
            f"pair(s) ({len(prune_indices)} messages), inserted summary."
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
