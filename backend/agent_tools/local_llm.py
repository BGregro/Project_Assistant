"""
tools/local_llm.py  —  Local LLM Tier (Ollama)

This module is the interface to the local Ollama instance running qwen2.5:7b.
It is NOT a registered tool (Claude doesn't call it directly). Instead, agent_core.py
and main.py call it for preprocessing tasks before/after Claude API calls:

  optimize_prompt()    — Rewrite raw user input into a clean, precise prompt.
  summarize_history()  — Compress old conversation turns into a compact summary.
  local_llm_call()     — Generic single-turn completion.
  is_ollama_available()— Quick health check used for status display in the UI.

All functions handle Ollama being offline gracefully: they log a warning and return
None / the original input, never raising exceptions to the caller.
"""

import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

# Ollama's default address. Can be overridden via config.json → "ollama_base_url"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL    = "qwen2.5:7b"

# Generous timeout: local inference can be slow on CPU
GENERATE_TIMEOUT = 60.0
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
        "model":  model,
        "prompt": prompt,
        "stream": False,     # Get the entire response at once (not token-by-token)
    }
    if system:
        # Ollama supports a system field directly in the generate request
        payload["system"] = system

    try:
        async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client:
            response = await client.post(f"{base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            return text if text else None

    except httpx.ConnectError:
        # Ollama is not running — this is expected when working offline
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

    WHY THIS EXISTS:
    Raw user messages often have typos, ambiguous pronouns, missing context, or
    conversational filler that wastes Claude API tokens and can confuse the model.
    A lightweight local pass that standardises the prompt reduces Claude API cost
    and improves response quality — especially for short, unclear messages.

    Falls back to the ORIGINAL message if Ollama is offline or returns garbage.
    This means the system still works 100% without a local model.
    """
    # Keep the system prompt short — the local model is small (7B) and we want
    # fast inference, not a reasoning chain.
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
        # Ollama offline or failed — silently fall back to original
        logger.info("[local_llm] optimize_prompt: fallback to original (Ollama unavailable).")
        return raw_message

    # Sanity check: if the result is wildly longer than the input, something went wrong
    # (the model may have ignored the system prompt and started explaining itself)
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
    the recent-turns window. The summary is injected into Claude's system prompt
    (not the messages array), so it costs ~1× instead of N× the turn tokens.

    Token budget goal: compress N turns into ~100-200 tokens.

    Args:
        turns: List of {role, content, timestamp} dicts (the OLD portion of history,
               beyond the recent verbatim window).
        model: Local Ollama model to use for summarization.
        base_url: Ollama base URL.
        max_chars_per_turn: How many chars of each turn to include in the prompt
                            sent to the local model. Keeps the summarization
                            prompt itself from being huge.

    Returns:
        A short summary string, or None if Ollama is unavailable.
    """
    if not turns:
        return None

    # Build a compact transcript from the old turns to feed to the summarizer.
    # We only send max_chars_per_turn per entry to keep the summarization prompt lean.
    transcript_lines = []
    for entry in turns:
        role    = entry.get("role", "?")
        content = entry.get("content", "")[:max_chars_per_turn]
        # Truncation marker so the model knows content was cut
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
