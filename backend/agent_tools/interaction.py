"""
interaction.py  —  Phase 6a: Mid-Task User Questions

Lets the agent pause mid-task and request clarification from the user.
The task runner suspends the agentic loop until the user responds (or a
10-minute timeout elapses), then continues with the answer injected into
the conversation context.

This is much better than the agent guessing:
  - Ambiguous requirements ("which approach do you prefer?")
  - Destructive decisions ("should I overwrite this file?")
  - Choosing between valid alternatives ("React or Vue?")

Usage from the agent:
    ask_user(question="Which database should I use — SQLite or PostgreSQL?")

The UI shows a question modal with a textarea.  The user's typed answer is
sent back via WebSocket as a question_answer message, which main.py routes
to task_runner.answer_question().

Module-level refs (set by main.py at startup):
    _task_runner_ref   — the live TaskRunner instance
    _send_event_ref    — the active WebSocket send_event coroutine
"""

from __future__ import annotations

import logging
from typing import Callable, Awaitable
from uuid import uuid4

from agent_tools import register_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level references — set by main.py after construction
# ---------------------------------------------------------------------------

_task_runner_ref = None   # TaskRunner instance
_send_event_ref  = None   # send_event(event_type, data) coroutine callable


def set_task_runner(tr) -> None:
    """Called by main.py once the TaskRunner instance is ready."""
    global _task_runner_ref
    _task_runner_ref = tr
    logger.info("[interaction] TaskRunner reference set.")


def set_send_event(fn: Callable) -> None:
    """Called by main.py whenever the active WebSocket send_event changes."""
    global _send_event_ref
    _send_event_ref = fn
    logger.debug("[interaction] send_event reference updated.")


# ---------------------------------------------------------------------------
# Tool: ask_user
# ---------------------------------------------------------------------------

async def ask_user(question: str) -> dict:
    """
    Pause the running task and ask the user a question.

    The agent loop suspends until:
      a) The user submits an answer via the question modal → answer is returned.
      b) 10 minutes pass with no response → returns with timed_out=True.

    The answer is injected back into the agent's context as part of the tool
    result, so Claude can continue the task with the user's input.

    Args:
        question: The question to display to the user.

    Returns:
        {
            "success":   True,
            "question":  str,    # the question that was asked
            "answer":    str,    # the user's response (empty string on timeout)
            "timed_out": bool,   # True if the user didn't respond in 10 minutes
            "note":      str,    # advisory message on timeout (omitted on success)
        }
    """
    if _task_runner_ref is None:
        logger.error("[interaction] ask_user called but TaskRunner is not set.")
        return {
            "success": False,
            "error":   "TaskRunner reference not initialised — cannot pause for user input.",
        }

    if _send_event_ref is None:
        logger.error("[interaction] ask_user called but send_event is not set.")
        return {
            "success": False,
            "error":   "WebSocket send_event not available — no active frontend connection.",
        }

    question_id = str(uuid4())
    logger.info(f"[interaction] Asking user (id={question_id[:8]}): {question[:80]!r}")

    # Delegate to TaskRunner — it owns the asyncio.Event and the timeout
    answer = await _task_runner_ref.ask_user(
        question=question,
        question_id=question_id,
        send_event=_send_event_ref,
    )

    timed_out = answer == ""

    if timed_out:
        logger.warning(
            f"[interaction] Question '{question_id[:8]}' timed out — "
            "no user response within 10 minutes."
        )
        return {
            "success":   True,
            "question":  question,
            "answer":    "",
            "timed_out": True,
            "note": (
                "User did not respond within 10 minutes — "
                "continuing with best guess based on available context."
            ),
        }

    logger.info(
        f"[interaction] Question '{question_id[:8]}' answered: {answer[:60]!r}"
    )
    return {
        "success":   True,
        "question":  question,
        "answer":    answer,
        "timed_out": False,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_interaction_tools() -> None:
    """Register the ask_user tool with the live tool registry."""

    register_tool(
        name="ask_user",
        description=(
            "Pause the current task and ask the user a question. "
            "Use this when you are unsure about a key decision mid-task: "
            "which approach to take, which API to use, whether to overwrite "
            "an existing file, ambiguous requirements, or choosing between "
            "multiple valid implementations. "
            "The task loop suspends until the user answers or 10 minutes pass. "
            "Always prefer asking over guessing when the decision is "
            "irreversible or significantly affects the outcome. "
            "Returns the user's answer as a string in the 'answer' field."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {
                    "type":        "string",
                    "description": (
                        "The question to ask the user. Be specific — include "
                        "the options you are considering so the user can give "
                        "a decisive answer. Example: "
                        "'Should I use SQLite (simpler, no server) or "
                        "PostgreSQL (more scalable)? SQLite is fine for a "
                        "personal tool; PostgreSQL is better if you expect "
                        "multiple users.'"
                    ),
                },
            },
            "required": ["question"],
        },
        handler=ask_user,
        is_destructive=False,   # reads user input, modifies nothing
    )

    logger.info("[startup] Registered tool: ask_user")
