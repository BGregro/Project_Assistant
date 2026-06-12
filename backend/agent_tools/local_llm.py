"""
agent_tools/local_llm.py  —  Local LLM Tier (Ollama)

This module is the interface to the local Ollama instance.
It is NOT a registered tool (Claude doesn't call it directly). Instead, agent_core.py
and main.py call it for preprocessing tasks before/after Claude API calls:

  optimize_prompt()     — Rewrite raw user input into a clean, precise prompt.
  summarize_history()   — Compress old conversation turns into a compact summary.
  local_llm_call()      — Generic single-turn completion via /api/generate.
  local_agent_call()    — Full agentic loop via /api/chat with tool use (Feature 3).
  is_ollama_available() — Quick health check used for status display in the UI.

All functions handle Ollama being offline gracefully: they log a warning and return
None / the original input, never raising exceptions to the caller.
"""

import json
import logging
import httpx
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Ollama's default address. Can be overridden via config.json → "ollama_base_url"
DEFAULT_BASE_URL    = "http://localhost:11434"
DEFAULT_MODEL       = "qwen2.5:7b"
DEFAULT_AGENT_MODEL = "qwen2.5:14b"

# Generous timeouts: local inference can be slow on CPU
GENERATE_TIMEOUT = 60.0
# NOTE: local_agent_call() no longer uses CHAT_TIMEOUT — it accepts a `timeout`
# parameter so the caller (agent_core.py) can pass config-driven values.
# CHAT_TIMEOUT is kept here only for reference / backward compatibility.
CHAT_TIMEOUT     = 300.0
HEALTH_TIMEOUT   = 3.0


async def local_llm_call(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
) -> Optional[str]:
    """
    Send a single prompt to the Ollama /api/generate endpoint and return the response.

    Uses stream=False so we get the full response in one HTTP response body,
    which is simpler to handle than streaming for short preprocessing tasks.

    Returns None if Ollama is unreachable or returns an error — never raises.
    """
    payload: dict = {
        "model":      model,
        "prompt":     prompt,
        "stream":     False,
        "keep_alive": -1,   # Keep model loaded for the life of the Ollama process
    }
    if system:
        payload["system"] = system

    try:
        async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client:
            response = await client.post(f"{base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            return text if text else None

    except httpx.ConnectError:
        logger.warning("[local_llm] Ollama is offline (connection refused).")
        return None
    except httpx.TimeoutException:
        logger.warning(f"[local_llm] Ollama request timed out after {GENERATE_TIMEOUT}s.")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"[local_llm] Ollama returned HTTP {e.response.status_code}.")
        return None
    except Exception as e:
        logger.warning(f"[local_llm] Unexpected error calling Ollama: {e}")
        return None


async def optimize_prompt(
    raw_message: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """
    Use the local LLM to rewrite a raw user message into a clean, precise prompt
    for Claude.

    Falls back to the ORIGINAL message if Ollama is offline or returns garbage.
    This means the system still works 100% without a local model.
    """
    system = (
        "You are a prompt optimizer for an AI assistant. "
        "Rewrite the user's message into a clear, concise, and unambiguous instruction. "
        "Preserve the original intent exactly. Fix grammar and remove filler words. "
        "Output ONLY the rewritten prompt — no explanation, no quotes, no preamble."
    )

    result = await local_llm_call(
        prompt=raw_message,
        model=model,
        system=system,
        base_url=base_url,
    )

    if result is None:
        logger.info("[local_llm] optimize_prompt: fallback to original (Ollama unavailable).")
        return raw_message

    # Sanity check: if the result is wildly longer than the input, something went wrong
    if len(result) > len(raw_message) * 3:
        logger.warning("[local_llm] optimize_prompt: result suspiciously long, using original.")
        return raw_message

    if result == raw_message:
        logger.debug("[local_llm] optimize_prompt: no change needed.")
    else:
        logger.info(
            f"[local_llm] Prompt optimized:\n"
            f"  ORIGINAL : {raw_message[:80]!r}\n"
            f"  OPTIMIZED: {result[:80]!r}"
        )

    return result


async def summarize_history(
    turns: list[dict],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    max_chars_per_turn: int = 300,
) -> Optional[str]:
    """
    Compress a list of old conversation turns into a compact natural-language summary.

    Called by build_context() in main.py when the conversation history grows beyond
    the recent-turns window. The summary is injected into Claude's system prompt.
    """
    if not turns:
        return None

    transcript_lines = []
    for entry in turns:
        role    = entry.get("role", "?")
        content = entry.get("content", "")[:max_chars_per_turn]
        if len(entry.get("content", "")) > max_chars_per_turn:
            content += "…"
        transcript_lines.append(f"{role.upper()}: {content}")

    transcript = "\n".join(transcript_lines)

    system = (
        "You are a conversation summarizer for an AI agent system. "
        "Your summary will be injected into a system prompt, so it must be dense and factual. "
        "Output ONLY the summary — no preamble, no labels, no markdown."
    )

    prompt = (
        f"Summarize this conversation history in 3-5 sentences. "
        f"Focus on: what the user is building or trying to accomplish, "
        f"key decisions or conclusions reached, any errors that were resolved, "
        f"and stated preferences or constraints.\n\n"
        f"CONVERSATION:\n{transcript}"
    )

    result = await local_llm_call(
        prompt=prompt,
        model=model,
        system=system,
        base_url=base_url,
    )

    if result:
        logger.info(
            f"[local_llm] summarize_history: compressed {len(turns)} entries "
            f"→ {len(result)} chars."
        )

    return result


async def local_agent_call(
    prompt: str,
    tools: list[dict],
    messages: list[dict],
    model: str = DEFAULT_AGENT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    max_iterations: int = 10,
    tool_dispatcher: Any = None,
    timeout: float = 300.0,
) -> str:
    """
    Run a full agentic loop entirely through Ollama — no Claude API required.

    Uses Ollama's /api/chat endpoint with OpenAI-compatible tool calling
    (supported by qwen2.5:14b and similar function-calling capable models).

    The loop mirrors agent_core.py's Claude loop:
      1. Send current messages + tool definitions to Ollama.
      2. If the model emits tool calls, dispatch them and feed results back.
      3. Repeat until the model returns a plain text response.
      4. Return the final text.

    Args:
        prompt:          The current user message (already appended to messages).
        tools:           List of tool definitions in Anthropic format (name, description,
                         input_schema). Converted to Ollama/OpenAI format internally.
        messages:        Full message history in [{role, content}] format.
        model:           Ollama model to use (default: qwen2.5:14b).
        base_url:        Ollama base URL.
        max_iterations:  Safety cap on tool call rounds to prevent infinite loops.
        tool_dispatcher: Async callable(tool_name, tool_input) → dict result.
                         Passed in from agent_core so we don't import it here
                         (avoids circular imports). If None, tool calls return errors.
        timeout:         Per-request HTTP timeout in seconds. Configurable via
                         config.json → "local_agent_timeout" (default 300s).
                         Larger models on CPU may need several minutes per response.

    Returns:
        Final text response string, or an error message if Ollama is unreachable.
    """

    # --- Convert Anthropic-format tool definitions to Ollama/OpenAI format ---
    # Anthropic: {"name": ..., "description": ..., "input_schema": {...}}
    # OpenAI:    {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
    ollama_tools = []
    for t in tools:
        ollama_tools.append({
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t.get("description", ""),
                "parameters":  t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })

    # Build the working message list (deep copy so we don't mutate the caller's list)
    working_messages = list(messages)

    for iteration in range(max_iterations):
        logger.info(f"[local_llm] local_agent_call iteration {iteration + 1} (timeout={timeout}s)")

        payload = {
            "model":      model,
            "messages":   working_messages,
            "tools":      ollama_tools,
            "stream":     False,
            "keep_alive": -1,   # Keep model loaded for the life of the Ollama process
        }

        try:
            # Use the caller-supplied timeout so large models get enough time
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError:
            logger.warning("[local_llm] Ollama offline during local_agent_call.")
            return "Error: local Ollama is not running. Start Ollama and try again."
        except httpx.TimeoutException:
            logger.warning(f"[local_llm] local_agent_call timed out after {timeout}s.")
            return (
                f"Error: local model timed out after {timeout}s. "
                "Try increasing local_agent_timeout in config, or use a smaller model."
            )
        except httpx.HTTPStatusError as e:
            logger.warning(f"[local_llm] Ollama HTTP {e.response.status_code} in local_agent_call.")
            return f"Error: Ollama returned HTTP {e.response.status_code}."
        except Exception as e:
            logger.warning(f"[local_llm] Unexpected error in local_agent_call: {e}")
            return f"Error: {e}"

        # Extract the assistant message from the response
        assistant_msg = data.get("message", {})
        content       = assistant_msg.get("content", "")
        tool_calls    = assistant_msg.get("tool_calls", [])

        # Append the assistant's turn to the working message list
        working_messages.append({
            "role":       "assistant",
            "content":    content,
            "tool_calls": tool_calls,
        })

        # --- No tool calls: final answer ---
        if not tool_calls:
            logger.info(f"[local_llm] local_agent_call finished in {iteration + 1} iteration(s).")
            return content.strip() if content else "No response generated."

        # --- Tool calls: dispatch each one and feed results back ---
        tool_result_messages = []
        for tc in tool_calls:
            fn   = tc.get("function", {})
            name = fn.get("name", "")
            # Arguments may be a JSON string or already a dict
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = raw_args

            logger.info(f"[local_llm] Tool call: {name}({args})")

            if tool_dispatcher is not None:
                try:
                    result = await tool_dispatcher(name, args)
                except Exception as e:
                    result = {"success": False, "error": str(e)}
            else:
                result = {"success": False, "error": "No tool dispatcher available in local mode."}

            # Ollama expects tool results as role="tool" messages
            tool_result_messages.append({
                "role":    "tool",
                "content": json.dumps(result, ensure_ascii=False),
                "name":    name,
            })

        working_messages.extend(tool_result_messages)

    # Reached max iterations without a final answer
    return "Error: local agent reached the maximum number of tool-use iterations without a final answer."


async def is_ollama_available(base_url: str = DEFAULT_BASE_URL) -> bool:
    """
    Quick health check: return True if Ollama is running and responsive.
    Used by main.py on startup and by the frontend status endpoint.
    Hits /api/tags which lists available models — a lightweight endpoint.
    """
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            resp = await client.get(f"{base_url}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Phase 3d — Efficiency layer functions
# ---------------------------------------------------------------------------

async def classify_intent(
    message: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """
    Classify a user message into exactly one routing category using the local LLM.

    Categories:
        SIMPLE  — greetings, trivial questions answerable in one sentence, no tools needed
        TOOL    — needs a tool call but is straightforward (file ops, web search, system info)
        COMPLEX — multi-step reasoning, planning, code architecture, research synthesis,
                  self-modification, app development

    Returns one of the three uppercase category strings.
    Falls back to "TOOL" (safe default) if the call fails or returns an unexpected value.
    """
    system = (
        "Classify the user message into exactly one category. "
        "Reply with only the category word, nothing else. "
        "Categories: "
        "SIMPLE (greetings, trivial questions answerable in one sentence, no tools needed), "
        "TOOL (needs a tool call but straightforward — file ops, web search, system info, single-step tasks), "
        "COMPLEX (multi-step reasoning, planning, code architecture, research synthesis, "
        "self-modification, app development)."
    )

    payload: dict = {
        "model":      model,
        "prompt":     message,
        "system":     system,
        "stream":     False,
        "keep_alive": -1,
    }

    try:
        async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client:
            response = await client.post(f"{base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "").strip().upper()
            if raw in ("SIMPLE", "TOOL", "COMPLEX"):
                logger.info(f"[local_llm] classify_intent → {raw}")
                return raw
            # If the model returned something unexpected, log and default
            logger.warning(
                f"[local_llm] classify_intent returned unexpected value {raw!r}, "
                "defaulting to TOOL."
            )
            return "TOOL"

    except httpx.ConnectError:
        logger.warning("[local_llm] classify_intent: Ollama offline, defaulting to TOOL.")
    except httpx.TimeoutException:
        logger.warning("[local_llm] classify_intent: timed out, defaulting to TOOL.")
    except httpx.HTTPStatusError as e:
        logger.warning(f"[local_llm] classify_intent: HTTP {e.response.status_code}, defaulting to TOOL.")
    except Exception as e:
        logger.warning(f"[local_llm] classify_intent: unexpected error ({e}), defaulting to TOOL.")

    return "TOOL"


async def compress_tool_result(
    tool_name: str,
    result: dict,
    user_goal: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> dict:
    """
    Compress a verbose tool result down to the information relevant to the user's goal.

    Skip compression entirely if the serialised result is under 500 characters —
    the overhead isn't worth it.

    Returns either:
        {"compressed": True,  "content": "<compressed text>"}   — on success
        original result dict unchanged                          — on failure or tiny result
    """
    serialised = str(result)
    if len(serialised) < 500:
        # Not worth compressing
        return result

    system = (
        "You are a result compressor. Given a tool result and the user's current goal, "
        "extract ONLY the information relevant to the goal. "
        "Discard boilerplate, irrelevant fields, and verbose formatting. "
        "Return compressed plain text, maximum 300 words."
    )

    prompt = f"Goal: {user_goal}\n\nTool result:\n{serialised}"

    payload: dict = {
        "model":      model,
        "prompt":     prompt,
        "system":     system,
        "stream":     False,
        "keep_alive": -1,
    }

    try:
        async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client:
            response = await client.post(f"{base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            compressed_text = data.get("response", "").strip()

        if compressed_text:
            ratio = len(compressed_text) / len(serialised)
            logger.info(
                f"[local_llm] compress_tool_result ({tool_name}): "
                f"{len(serialised)} → {len(compressed_text)} chars "
                f"({ratio:.0%})"
            )
            return {"compressed": True, "content": compressed_text}

    except httpx.ConnectError:
        logger.warning("[local_llm] compress_tool_result: Ollama offline, using original.")
    except httpx.TimeoutException:
        logger.warning("[local_llm] compress_tool_result: timed out, using original.")
    except httpx.HTTPStatusError as e:
        logger.warning(f"[local_llm] compress_tool_result: HTTP {e.response.status_code}, using original.")
    except Exception as e:
        logger.warning(f"[local_llm] compress_tool_result: unexpected error ({e}), using original.")

    # Any failure: return the original so Claude always gets something
    return result


async def summarize_completed_steps(
    steps_text: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """
    Compress a multi-step task history into 3-5 dense bullet points.

    Called by TaskRunner when the running token estimate exceeds the
    compression_threshold config value.

    Returns the summary string, or steps_text unchanged on failure so
    the task can always continue.
    """
    system = (
        "Summarize the following completed task steps into 3-5 bullet points. "
        "Focus on what was accomplished and what key data was found. "
        "Be dense and specific. Start each bullet with •"
    )

    payload: dict = {
        "model":      model,
        "prompt":     steps_text,
        "system":     system,
        "stream":     False,
        "keep_alive": -1,
    }

    try:
        async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client:
            response = await client.post(f"{base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            summary = data.get("response", "").strip()

        if summary:
            logger.info(
                f"[local_llm] summarize_completed_steps: "
                f"{len(steps_text)} → {len(summary)} chars"
            )
            return summary

    except httpx.ConnectError:
        logger.warning("[local_llm] summarize_completed_steps: Ollama offline, using original.")
    except httpx.TimeoutException:
        logger.warning("[local_llm] summarize_completed_steps: timed out, using original.")
    except httpx.HTTPStatusError as e:
        logger.warning(f"[local_llm] summarize_completed_steps: HTTP {e.response.status_code}, using original.")
    except Exception as e:
        logger.warning(f"[local_llm] summarize_completed_steps: unexpected error ({e}), using original.")

    return steps_text


async def prevalidate_code(
    code: str,
    language: str,
    intent: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> tuple[bool, str]:
    """
    Ask the local LLM to review agent-written code for obvious bugs before execution.

    Returns:
        (True,  "OK")           — code looks correct, proceed with execute_code
        (False, "<issue>")      — obvious problem found, return synthetic error to Claude
        (True,  "OK")           — on any failure (never block on prevalidation errors)

    The function is intentionally lenient on failure: if Ollama is offline or
    the call times out, the code goes through to execute_code unchanged.
    Pre-validation is a cost-saving optimisation, not a security gate.
    """
    system = (
        f"Review this {language} code for obvious bugs, syntax errors, or logic errors "
        f"that would cause it to fail. The code is intended to: {intent}. "
        "Reply with either 'OK' if the code looks correct, or 'ISSUE: ' followed by a "
        "one-sentence description of the problem. Nothing else."
    )

    payload: dict = {
        "model":      model,
        "prompt":     code,
        "system":     system,
        "stream":     False,
        "keep_alive": -1,
    }

    try:
        async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client:
            response = await client.post(f"{base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            verdict = data.get("response", "").strip()

        if verdict.upper().startswith("ISSUE:"):
            issue = verdict[6:].strip()
            logger.info(f"[local_llm] prevalidate_code: ISSUE detected — {issue!r}")
            return (False, issue)

        logger.debug("[local_llm] prevalidate_code: OK")
        return (True, "OK")

    except httpx.ConnectError:
        logger.warning("[local_llm] prevalidate_code: Ollama offline, passing through.")
    except httpx.TimeoutException:
        logger.warning("[local_llm] prevalidate_code: timed out, passing through.")
    except httpx.HTTPStatusError as e:
        logger.warning(f"[local_llm] prevalidate_code: HTTP {e.response.status_code}, passing through.")
    except Exception as e:
        logger.warning(f"[local_llm] prevalidate_code: unexpected error ({e}), passing through.")

    # Never block on prevalidation failure
    return (True, "OK")


async def select_relevant_tools(
    user_message: str,
    all_tool_names: list[str],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    max_tools: int = 10,
) -> list[str]:
    """
    Use the local LLM to select the most relevant tools for the current request.
    Reduces input token usage by 40-60% by not sending all tool definitions to Claude.

    Always includes core tools that should always be available:
    list_capabilities, recall_memory, log_research, log_fact, get_project_status.

    Returns the filtered list of tool names, or all_tool_names if selection fails
    or returns too few results.
    """
    ALWAYS_INCLUDE = {
        "list_capabilities", "recall_memory", "log_research",
        "log_fact", "get_project_status",
    }

    try:
        prompt = (
            f"Given this user request: '{user_message[:300]}'\n\n"
            f"Available tools: {', '.join(all_tool_names)}\n\n"
            f"Select the {max_tools} most relevant tools for handling this request. "
            "Consider what actions will likely be needed. "
            "Reply with ONLY a comma-separated list of tool names, nothing else."
        )
        response = await local_llm_call(prompt, model, base_url=base_url)
        if not response:
            logger.warning("[local_llm] select_relevant_tools: empty response — using full list.")
            return all_tool_names

        selected = [t.strip() for t in response.split(",") if t.strip() in all_tool_names]

        # Always include core tools
        for t in ALWAYS_INCLUDE:
            if t in all_tool_names and t not in selected:
                selected.append(t)

        # If selection is too small or failed, return all tools
        if len(selected) < 5:
            logger.warning("[local_llm] Tool selection returned too few tools — using full list.")
            return all_tool_names

        logger.info(f"[local_llm] Tool pre-filter: {len(all_tool_names)} → {len(selected)} tools")
        return selected

    except Exception as e:
        logger.warning(f"[local_llm] Tool selection failed ({e}) — using full tool list.")
        return all_tool_names


async def unload_model(model: str, base_url: str = DEFAULT_BASE_URL) -> None:
    """
    Explicitly unload the model from Ollama's memory.
    Called on WebSocket disconnect and server shutdown.
    Sends a minimal request with keep_alive=0 which tells Ollama
    to evict the model immediately instead of waiting for its timeout.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
            )
            logger.info(f"[local_llm] Unloaded model '{model}' from Ollama memory.")
    except Exception as e:
        logger.debug(f"[local_llm] unload_model: ignored error ({e})")
