"""
agent_core.py  —  Agent Core

The "brain" of the agent. Owns the full request lifecycle:

  1. Optionally optimise the user's raw message via local LLM (prompt optimizer).
  2. If local_mode is enabled, delegate entirely to local_agent_call() (no Claude API).
     Otherwise, send the message to Claude with the tool definitions.
  3. If Claude requests tools, dispatch them (with permission checks for destructive ones).
  4. Feed tool results back to Claude and repeat until Claude gives a final answer.
  5. Stream events to the frontend at every step so the UI stays live.
  6. After any tool result that contains a "tree" key, broadcast a tree_update event
     so the frontend sidebar stays in sync.

The core knows NOTHING about how tools work internally — it only calls handlers
from the registry. This keeps the architecture modular.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Awaitable

import anthropic

from agent_tools import get_all_definitions, get_handler, is_destructive as tool_is_destructive
from agent_tools.local_llm import optimize_prompt, local_llm_call, local_agent_call

logger = logging.getLogger(__name__)


class AgentCore:
    def __init__(self, config: dict) -> None:
        self.config = config

        llm_cfg = config.get("llm", {})
        self.primary_model:     str  = llm_cfg.get("primary",     "claude-haiku-4-5")
        self.complex_model:     str  = llm_cfg.get("complex",     "claude-sonnet-4-6")
        self.local_model:       str  = llm_cfg.get("local",       "qwen2.5:7b")
        self.local_agent_model: str  = llm_cfg.get("local_agent", "qwen2.5:14b")

        self.use_prompt_optimizer: bool  = config.get("use_prompt_optimizer", True)
        self.local_fallback:       bool  = config.get("local_fallback", True)
        self.local_mode:           bool  = config.get("local_mode", False)
        self.ollama_url:           str   = config.get("ollama_base_url", "http://localhost:11434")

        # local_agent_timeout is the per-HTTP-request timeout (seconds) for the
        # agentic Ollama loop. Large models on CPU can take several minutes per
        # response, so we surface this in config rather than hard-coding it.
        self.local_agent_timeout: float = float(config.get("local_agent_timeout", 300))

        ctx_cfg = config.get("context", {})
        self.max_iterations: int = ctx_cfg.get("max_iterations_per_turn", 10)

        # AsyncAnthropic reads ANTHROPIC_API_KEY from the environment automatically.
        # Do NOT hardcode keys here; use a .env file or set the env var manually.
        self.client = anthropic.AsyncAnthropic()

        self._base_system_prompt = (
            "You are a capable personal AI agent with access to filesystem tools. "
            "Use tools whenever they would help complete the user's request. "
            "Before using a tool, briefly state what you're about to do. "
            "After getting tool results, synthesise them into a clear final answer. "
            "Be concise. Avoid unnecessary preamble.\n\n"
            "You have access to an execute_code tool that runs Python or Bash on the "
            "host machine. Prefer writing and running code over guessing at results "
            "for anything computational, data-related, or verifiable by execution. "
            "Always inspect stdout and stderr from the result before concluding success "
            "or failure — a zero exit code with empty stdout may still mean the output "
            "went to a file or was suppressed. "
            "If the first run fails, read the error, fix the code, and try again rather "
            "than giving up or answering from assumptions."
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
        context_summary: str = "",
    ) -> str:
        """
        Process a single user turn end-to-end.

        Args:
            user_message:           Raw text from the user.
            history:                Recent turns as [{role, content}].
            send_event:             Async callback — sends a typed event to the frontend.
            pending_confirmations:  Shared dict for the permission layer.
            context_summary:        Optional context note injected into the system prompt.

        Returns:
            The final text reply from the agent.
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
            if optimized_message != user_message:
                await send_event("prompt_optimized", {
                    "original":  user_message,
                    "optimized": optimized_message,
                })
            else:
                await send_event("prompt_optimized", None)

        # ----------------------------------------------------------------
        # Step 2: Build system prompt — append context summary if present
        # ----------------------------------------------------------------
        system = self._base_system_prompt
        if context_summary:
            system = f"{self._base_system_prompt}\n\n{context_summary}"

        # ----------------------------------------------------------------
        # Step 3: Build message list
        # ----------------------------------------------------------------
        messages = history + [{"role": "user", "content": optimized_message}]

        # ----------------------------------------------------------------
        # Step 4: Route to local mode or Claude API
        # ----------------------------------------------------------------
        if self.local_mode:
            return await self._run_local(messages, send_event, pending_confirmations)
        else:
            return await self._run_claude(
                messages, system, send_event, pending_confirmations, optimized_message
            )

    # ------------------------------------------------------------------
    # Local-only agentic loop
    # ------------------------------------------------------------------

    async def _run_local(
        self,
        messages: list[dict],
        send_event: Callable,
        pending_confirmations: dict,
    ) -> str:
        """
        Run the full agentic loop through Ollama — no Claude API calls.
        Passes self.local_agent_timeout (read from config) to local_agent_call()
        so large models have enough time to respond.
        """
        await send_event("status", {"text": "Running locally (Ollama)…"})

        async def _dispatch(tool_name: str, tool_input: dict) -> dict:
            fake_id = f"local_{tool_name}_{id(tool_input)}"
            tool_result_msg = await self._execute_tool(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_use_id=fake_id,
                send_event=send_event,
                pending_confirmations=pending_confirmations,
            )
            try:
                return json.loads(tool_result_msg.get("content", "{}"))
            except (json.JSONDecodeError, AttributeError):
                return {"success": False, "error": "Could not parse tool result."}

        final_text = await local_agent_call(
            prompt=messages[-1]["content"] if messages else "",
            tools=get_all_definitions(),
            messages=messages,
            model=self.local_agent_model,
            base_url=self.ollama_url,
            max_iterations=self.max_iterations,
            tool_dispatcher=_dispatch,
            timeout=self.local_agent_timeout,   # ← config-driven, no longer hardcoded
        )

        await send_event("message", {"text": final_text, "source": "local"})
        return final_text

    # ------------------------------------------------------------------
    # Claude agentic loop
    # ------------------------------------------------------------------

    async def _run_claude(
        self,
        messages: list[dict],
        system: str,
        send_event: Callable,
        pending_confirmations: dict,
        optimized_message: str,
    ) -> str:
        """The original Claude API agentic loop."""
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            await send_event("status", {"text": "Thinking…"})

            try:
                response = await self.client.messages.create(
                    model=self.primary_model,
                    max_tokens=4096,
                    system=system,
                    tools=get_all_definitions(),
                    messages=messages,
                )
            except anthropic.APIConnectionError:
                return await self._handle_api_unreachable(optimized_message, send_event)
            except anthropic.AuthenticationError:
                await send_event("error", {"text": "Invalid or missing ANTHROPIC_API_KEY."})
                return "Error: authentication failed. Check your API key."
            except anthropic.RateLimitError:
                await send_event("error", {"text": "Claude API rate limit hit. Try again shortly."})
                return "Error: rate limit exceeded."
            except anthropic.APIError as e:
                await send_event("error", {"text": f"Claude API error: {e}"})
                return f"Error: {e}"

            if response.stop_reason == "end_turn":
                final_text = _extract_text(response)
                await send_event("message", {"text": final_text, "source": "claude"})
                logger.info(f"[agent] Finished in {iteration} iteration(s).")
                return final_text

            elif response.stop_reason == "tool_use":
                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                messages.append({"role": "assistant", "content": response.content})

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

                messages.append({"role": "user", "content": tool_results})

            else:
                logger.warning(f"[agent] Unexpected stop_reason: {response.stop_reason!r}")
                partial = _extract_text(response)
                if partial:
                    await send_event("message", {"text": partial, "source": "claude"})
                    return partial
                break

        msg = "Agent reached the maximum number of tool-use iterations without a final answer."
        await send_event("error", {"text": msg})
        return msg

    # ------------------------------------------------------------------
    # Tool execution (with permission check + tree broadcast)
    # ------------------------------------------------------------------

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict,
        tool_use_id: str,
        send_event: Callable,
        pending_confirmations: dict,
    ) -> dict:
        await send_event("tool_call", {"tool": tool_name, "input": tool_input, "tool_use_id": tool_use_id})

        if tool_is_destructive(tool_name):
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

        handler = get_handler(tool_name)
        if handler is None:
            result = {"success": False, "error": f"No handler registered for tool '{tool_name}'."}
            logger.error(f"[agent] Unregistered tool called: {tool_name!r}")
        else:
            try:
                result = await handler(**tool_input)
            except TypeError as e:
                result = {"success": False, "error": f"Tool argument error: {e}"}
                logger.exception(f"[agent] Tool '{tool_name}' argument error")
            except Exception as e:
                result = {"success": False, "error": str(e)}
                logger.exception(f"[agent] Tool '{tool_name}' raised an exception")

        await send_event("tool_result", {
            "tool":    tool_name,
            "success": result.get("success", False),
            "result":  result,
        })

        # Tree broadcast: if the tool result contains a "tree" key (filesystem tools),
        # forward it as a separate WebSocket event for the sidebar.
        if isinstance(result, dict) and "tree" in result:
            await send_event("tree_update", {"tree": result["tree"]})

        return _make_tool_result(tool_use_id, result)

    # ------------------------------------------------------------------
    # Permission layer
    # ------------------------------------------------------------------

    async def _request_confirmation(
        self,
        tool_name: str,
        tool_input: dict,
        send_event: Callable,
        pending_confirmations: dict,
        timeout: float = 60.0,
    ) -> bool:
        confirmation_id = f"{tool_name}_{id(tool_input)}"
        event = asyncio.Event()
        pending_confirmations[confirmation_id] = {"event": event, "result": None}

        await send_event("confirm_required", {
            "confirmation_id": confirmation_id,
            "tool":    tool_name,
            "input":   tool_input,
            "message": (
                f"The agent wants to run '{tool_name}' "
                f"with: {json.dumps(tool_input, ensure_ascii=False)}"
            ),
        })

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            approved = pending_confirmations[confirmation_id]["result"] is True
            logger.info(
                f"[agent] Confirmation '{confirmation_id}': "
                f"{'approved' if approved else 'denied'}"
            )
            return approved
        except asyncio.TimeoutError:
            logger.warning(
                f"[agent] Confirmation '{confirmation_id}' timed out after {timeout}s."
            )
            return False
        finally:
            pending_confirmations.pop(confirmation_id, None)

    # ------------------------------------------------------------------
    # Local fallback (Claude API unreachable)
    # ------------------------------------------------------------------

    async def _handle_api_unreachable(self, message: str, send_event: Callable) -> str:
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
    """
    return {
        "type":        "tool_result",
        "tool_use_id": tool_use_id,
        "content":     json.dumps(result, ensure_ascii=False),
    }
