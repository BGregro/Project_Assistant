"""
agent_tools/capability_tools.py  —  Phase 15a: Capability Gap Detection

Checks whether the agent's existing tools cover all steps needed for a goal
BEFORE work starts, so missing capabilities can be designed and written
proactively instead of discovered mid-task.

Runs entirely on the local qwen3:14b agent tier — zero Claude API cost.

Registers one tool via register_capability_tools():
    analyze_capability_gap(task_goal) -> {gaps, can_proceed, suggested_tools}

Workflow the agent should follow before a complex multi-step task:
    1. analyze_capability_gap(task_goal)
    2. If gaps found: design_tool(gap_description) -> implement_tool_from_design(spec_path)
    3. Proceed with the task using the newly registered tool(s)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import register_tool, list_tools

logger = logging.getLogger(__name__)


def _get_ollama_url() -> str:
    """Read ollama_base_url from config.json, fall back to default. Never raises."""
    config_path = Path(__file__).parent.parent.parent / "config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f).get("ollama_base_url", "http://localhost:11434")
    except Exception:
        return "http://localhost:11434"


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

async def analyze_capability_gap(task_goal: str) -> dict[str, Any]:
    """
    Ask the local qwen3:14b model whether the current tool set can complete
    every step of task_goal, before the agent commits to the task.

    Falls back to {"gaps": [], "can_proceed": True, "suggested_tools": []} on
    any local-LLM or parsing failure — a gap-check failure should never block
    the agent from attempting a task it might well be able to do.
    """
    from .local_llm import local_llm_call, strip_think_tags

    tool_names = list_tools()

    prompt = (
        f"An AI agent has these tools: {', '.join(tool_names)}\n\n"
        f'For this task: "{task_goal}"\n\n'
        "Identify any steps the agent cannot complete with these tools.\n"
        "Reply ONLY with a JSON object:\n"
        '{\n'
        '  "gaps": ["description of missing capability", ...],\n'
        '  "can_proceed": true|false,\n'
        '  "suggested_tools": ["tool_name_to_write", ...]\n'
        '}\n'
        'If no gaps, return {"gaps": [], "can_proceed": true, "suggested_tools": []}'
    )

    default_result = {
        "success": True,
        "task_goal": task_goal,
        "gaps": [],
        "can_proceed": True,
        "suggested_tools": [],
        "existing_tool_count": len(tool_names),
    }

    try:
        response = await local_llm_call(
            prompt, model="qwen3:14b", base_url=_get_ollama_url()
        )
        if not response:
            logger.warning(
                "[capability_tools] analyze_capability_gap: Ollama unavailable — "
                "assuming no gaps."
            )
            return default_result

        cleaned = strip_think_tags(response).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if "\n" in cleaned:
                cleaned = cleaned.split("\n", 1)[1]

        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            logger.debug(
                f"[capability_tools] Could not parse gap-analysis JSON: {e} — "
                f"raw response: {cleaned[:200]!r}"
            )
            return default_result

        gaps = parsed.get("gaps", [])
        if not isinstance(gaps, list):
            gaps = []
        suggested_tools = parsed.get("suggested_tools", [])
        if not isinstance(suggested_tools, list):
            suggested_tools = []
        can_proceed = bool(parsed.get("can_proceed", True))

        logger.info(
            f"[capability_tools] analyze_capability_gap({task_goal[:50]!r}): "
            f"{len(gaps)} gap(s), can_proceed={can_proceed}"
        )

        return {
            "success": True,
            "task_goal": task_goal,
            "gaps": gaps,
            "can_proceed": can_proceed,
            "suggested_tools": suggested_tools,
            "existing_tool_count": len(tool_names),
        }

    except Exception as e:
        logger.warning(f"[capability_tools] analyze_capability_gap failed (non-fatal): {e}")
        return default_result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_capability_tools() -> None:
    """Register analyze_capability_gap. Call once at startup from main.py."""

    register_tool(
        name="analyze_capability_gap",
        description=(
            "Analyze whether existing tools can cover all steps for a given goal. "
            "Returns capability gaps and suggested tool names to write. "
            "Call this before starting any complex task to check for missing capabilities. "
            "Runs on the local model — costs no Claude API tokens."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task_goal": {
                    "type": "string",
                    "description": "The goal or task the agent is about to attempt.",
                },
            },
            "required": ["task_goal"],
        },
        handler=analyze_capability_gap,
        is_destructive=False,  # Read-only analysis — no side effects
    )
