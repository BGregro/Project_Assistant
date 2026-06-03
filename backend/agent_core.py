"""
agent_core.py  —  Agent Core

The "brain" of the agent. Owns the full request lifecycle:

  1. Optionally optimise the user's raw message via local LLM (prompt optimizer).
  2. Send the message to Claude with the tool definitions.
  3. If Claude requests tools, dispatch them (with permission checks for destructive ones).
  4. Feed tool results back to Claude and repeat until Claude gives a final answer.
  5. Stream events to the frontend at every step so the UI stays live.

The core knows NOTHING about how tools work internally — it only calls handlers
from the registry. This keeps the architecture modular (adding a Phase 2 tool
requires zero changes here).
"""

import asyncio
import json
import logging
from typing import Any, Callable, Awaitable

import anthropic

from tools import get_all_definitions, get_handler, is_destructive
from tools.local_llm import optimize_prompt, local_llm_call

logger = logging.getLogger(__name__)


class AgentCore:
    def __init__(self, config: dict) -> None:
        self.config = config

        llm_cfg = config.get("llm", {})
        self.primary_model: str = llm_cfg.get("primary", "claude-haiku-4-5-20251001")
        self.complex_model: str = llm_cfg.get("complex", "claude-sonnet-4-6")
        self.local_model: str = llm_cfg.get("local", "qwen2.5:7b")

        self.use_prompt_optimizer: bool = config.get("use_prompt_optimizer", True)
        self.local_fallback: bool = config.get("local_fallback", True)
        self.ollama_url: str = config.get("ollama_base_url", "http://localhost:11434")

        ctx_cfg = config.get("context", {})
        self.max_iterations: int = ctx_cfg.get("max_iterations_per_turn", 10)

        # AsyncAnthropic reads ANTHROPIC_API_KEY from the environment automatically.
        # Do NOT hardcode keys here; use a .env file or set the env var manually.
        self.client = anthropic.AsyncAnthropic()

        # Short system prompt on purpose: every token in the system prompt is paid
        # for on every API call. Keep it informative but not verbose.
        self.system_prompt = (
            "You are a capable personal AI agent with access to filesystem tools. "
            "Use tools whenever they would help complete the user's request. "
            "Before using a tool, briefly state what you're about to do. "
            "After getting tool results, synthesise them into a clear final answer. "
            "Be concise. Avoid unnecessary preamble."
        )

    # ------------------------------------------------------------------
    # Public entry point — called from main.py WebSocket handler
    # ------------------------------------------------------------------

    async def run(
        self,
        user_message: str,
        history: list[dict],
        send_event: Callable[[str, dict], Awaitable[None]],
        pending_confirmations: dict,
    ) -> str:
        """
        Process a single user turn end-to-end.

        Args:
            user_message:           Raw text from the user.
            history:                Previous turns as [{"role":..., "content":...}].
            send_event:             Async callback — sends a typed event to the frontend.
            pending_confirmations:  Shared dict for the permission layer (confirmation events).

        Returns:
            The final text reply from the agent (also sent via send_event).
        """

        # ----------------------------------------------------------------
        # Step 1: Prompt optimisation (local LLM tier)
        # ----------------------------------------------------------------
        optimized_message = user_message
        if self.use_prompt_optimizer:
            await send_event("status", {"text": "Optimizing prompt…"})
            optimized_message = await optimize_prompt(
                raw_message=user_message,
                model=self.local_model,
                base_url=self.ollama_url,
            )
            # Tell the frontend whether the optimizer actually changed anything
            if optimized_message != user_message:
                await send_event("prompt_optimized", {
                    "original": user_message,
                    "optimized": optimized_message,
                })
            else:
                await send_event("prompt_optimized", None)   # No change; UI can hide indicator

        # ----------------------------------------------------------------
        # Step 2: Build message list for Claude
        # ----------------------------------------------------------------
        # We use the OPTIMIZED message when talking to Claude (better prompts → better answers)
        # but store the ORIGINAL in history (more readable for the user).
        messages = history + [{"role": "user", "content": optimized_message}]

        # ----------------------------------------------------------------
        # Step 3: Agentic loop — run until final answer or iteration limit
        # ----------------------------------------------------------------
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            await send_event("status", {"text": "Thinking…"})

            # --- Call Claude API ---
            try:
                response = await self.client.messages.create(
                    model=self.primary_model,
                    max_tokens=4096,
                    system=self.system_prompt,
                    tools=get_all_definitions(),   # tells Claude what tools exist
                    messages=messages,
                )
            except anthropic.APIConnectionError:
                return await self._handle_api_unreachable(
                    optimized_message, send_event
                )
            except anthropic.AuthenticationError:
                await send_event("error", {"text": "Invalid or missing ANTHROPIC_API_KEY."})
                return "Error: authentication failed. Check your API key."
            except anthropic.RateLimitError:
                await send_event("error", {"text": "Claude API rate limit hit. Try again shortly."})
                return "Error: rate limit exceeded."
            except anthropic.APIError as e:
                await send_event("error", {"text": f"Claude API error: {e}"})
                return f"Error: {e}"

            # --- Handle stop reason ---

            if response.stop_reason == "end_turn":
                # Claude finished — extract and return the final text
                final_text = _extract_text(response)
                await send_event("message", {"text": final_text, "source": "claude"})
                logger.info(f"[agent] Finished in {iteration} iteration(s).")
                return final_text

            elif response.stop_reason == "tool_use":
                # Claude wants to call one or more tools.
                # The response may include both text blocks and tool_use blocks.
                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

                # Append Claude's full response (tool_use blocks included) to history.
                # This is required by the Anthropic API — the assistant turn must precede
                # the tool_result turn.
                messages.append({"role": "assistant", "content": response.content})

                # Execute each requested tool and collect results
                tool_results = []
                for block in tool_use_blocks:
                    result = await self._execute_tool(
                        tool_name=block.name,
                        tool_input=block.input,
                        tool_use_id=block.id,
                        send_event=send_event,
                        pending_confirmations=pending_confirmations,
                    )
                    tool_results.append(result)

                # Feed all tool results back to Claude in a single user turn
                messages.append({"role": "user", "content": tool_results})

            else:
                # max_tokens or other unexpected stop — try to extract any text
                logger.warning(f"[agent] Unexpected stop_reason: {response.stop_reason!r}")
                partial = _extract_text(response)
                if partial:
                    await send_event("message", {"text": partial, "source": "claude"})
                    return partial
                break

        # Iteration limit reached
        msg = "Agent reached the maximum number of tool-use iterations without a final answer."
        await send_event("error", {"text": msg})
        return msg

    # ------------------------------------------------------------------
    # Tool execution (with permission check)
    # ------------------------------------------------------------------

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict,
        tool_use_id: str,
        send_event: Callable,
        pending_confirmations: dict,
    ) -> dict:
        """
        Run a single tool call:
          - Notify the frontend the tool is being called.
          - If destructive, pause and ask user for confirmation.
          - Call the handler.
          - Notify the frontend of the result.
          - Return a tool_result block for the next Claude message.
        """
        # Notify frontend a tool call is happening (shows inline in the chat)
        await send_event("tool_call", {"tool": tool_name, "input": tool_input})

        # --- Permission gate for destructive tools ---
        if is_destructive(tool_name):
            approved = await self._request_confirmation(
                tool_name=tool_name,
                tool_input=tool_input,
                send_event=send_event,
                pending_confirmations=pending_confirmations,
            )
            if not approved:
                result = {"success": False, "error": "Operation cancelled by user."}
                await send_event("tool_denied", {"tool": tool_name})
                return _make_tool_result(tool_use_id, result)

        # --- Dispatch to the registered handler ---
        handler = get_handler(tool_name)
        if handler is None:
            result = {"success": False, "error": f"No handler registered for tool '{tool_name}'."}
            logger.error(f"[agent] Unregistered tool called: {tool_name!r}")
        else:
            try:
                result = await handler(**tool_input)
            except TypeError as e:
                # Argument mismatch between Claude's call and the handler signature
                result = {"success": False, "error": f"Tool argument error: {e}"}
                logger.exception(f"[agent] Tool '{tool_name}' argument error")
            except Exception as e:
                result = {"success": False, "error": str(e)}
                logger.exception(f"[agent] Tool '{tool_name}' raised an exception")

        await send_event("tool_result", {
            "tool": tool_name,
            "success": result.get("success", False),
            "result": result,
        })

        return _make_tool_result(tool_use_id, result)

    # ------------------------------------------------------------------
    # Permission layer — pause and wait for frontend confirmation
    # ------------------------------------------------------------------

    async def _request_confirmation(
        self,
        tool_name: str,
        tool_input: dict,
        send_event: Callable,
        pending_confirmations: dict,
        timeout: float = 60.0,
    ) -> bool:
        """
        Pause execution and ask the user to approve a destructive operation.

        Flow:
          1. Generate a unique confirmation_id.
          2. Register an asyncio.Event in pending_confirmations.
          3. Send "confirm_required" to the frontend (UI shows Approve/Deny).
          4. Wait for the frontend to POST back a "confirm" message that sets the Event.
          5. Return True (approved) or False (denied/timed out).

        The main.py WebSocket handler is responsible for calling
        pending_confirmations[id]["event"].set() when the user responds.
        """
        # Use object id to make confirmation_id unique even for parallel calls
        confirmation_id = f"{tool_name}_{id(tool_input)}"
        event = asyncio.Event()
        pending_confirmations[confirmation_id] = {"event": event, "result": None}

        await send_event("confirm_required", {
            "confirmation_id": confirmation_id,
            "tool": tool_name,
            "input": tool_input,
            "message": (
                f"The agent wants to run '{tool_name}' "
                f"with: {json.dumps(tool_input, ensure_ascii=False)}"
            ),
        })

        try:
            # Wait for the user to click Approve or Deny in the UI
            await asyncio.wait_for(event.wait(), timeout=timeout)
            approved = pending_confirmations[confirmation_id]["result"] is True
            logger.info(f"[agent] Confirmation '{confirmation_id}': {'approved' if approved else 'denied'}")
            return approved
        except asyncio.TimeoutError:
            logger.warning(f"[agent] Confirmation '{confirmation_id}' timed out after {timeout}s.")
            return False
        finally:
            pending_confirmations.pop(confirmation_id, None)

    # ------------------------------------------------------------------
    # Local fallback when Claude API is unreachable
    # ------------------------------------------------------------------

    async def _handle_api_unreachable(
        self,
        message: str,
        send_event: Callable,
    ) -> str:
        """
        Claude API can't be reached. If local_fallback is enabled and Ollama is
        running, answer directly from the local model (no tools, no agentic loop).
        """
        if self.local_fallback:
            await send_event("status", {"text": "Claude API unreachable — trying local fallback…"})
            fallback = await local_llm_call(
                prompt=message,
                model=self.local_model,
                base_url=self.ollama_url,
            )
            if fallback:
                note = "\n\n*(Answered by local model — Claude API was unreachable)*"
                full = fallback + note
                await send_event("message", {"text": full, "source": "local"})
                return full

        msg = "Could not reach Claude API and local fallback is unavailable."
        await send_event("error", {"text": msg})
        return msg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(response: anthropic.types.Message) -> str:
    """Pull all TextBlock content from a Claude response into a single string."""
    parts = [block.text for block in response.content if hasattr(block, "text")]
    return "\n".join(parts).strip()


def _make_tool_result(tool_use_id: str, result: Any) -> dict:
    """
    Wrap a tool result in the format the Anthropic API expects:
    { type: "tool_result", tool_use_id: "...", content: "<json string>" }

    The content field is a JSON string, not a nested dict — Claude parses it.
    """
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(result, ensure_ascii=False),
    }
