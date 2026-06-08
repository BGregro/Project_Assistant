"""
research_mode.py  —  Phase 3h: Structured Research Mode

Registers one tool:

  deep_research(goal, criteria, max_questions) → dict

deep_research is a scaffolding tool — it does NOT call the Claude API or
perform any web searches itself.  Instead it:

  1. Calls the local LLM to decompose the high-level goal into sub-questions.
  2. Reads the user profile (if present) to extract relevant context.
  3. Queries long-term memory to surface what the agent already knows.
  4. Returns a structured research plan that the agent uses to guide its own
     subsequent search_web / fetch_page / log_research tool calls.

The separation is deliberate: Claude decides *how* to execute the plan;
this tool only provides the *structure*.  This keeps the tool simple,
testable, and free of API dependencies.
"""

import json
import logging
import pathlib
import re

from agent_tools import register_tool
from agent_tools.local_llm import local_llm_call

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_profile() -> dict:
    """Read memory/user_profile.json, return {} on any error."""
    profile_path = pathlib.Path(__file__).parent.parent.parent / "memory" / "user_profile.json"
    if not profile_path.exists():
        return {}
    try:
        return json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[research_mode] Could not read user_profile.json: {e}")
        return {}


def _query_known(goal: str) -> list:
    """Return up to 5 past research entries related to the goal."""
    try:
        from memory import long_term
        return long_term.query_research(topic=goal, last_n=5)
    except Exception as e:
        logger.warning(f"[research_mode] Could not query long-term memory: {e}")
        return []


def _parse_sub_questions(raw: str) -> list[str]:
    """
    Try to parse the local LLM's response as a JSON array.
    Fall back to splitting on newlines if JSON parsing fails.
    Returns a non-empty list of strings (or a generic fallback).
    """
    cleaned = raw.strip()
    # Strip markdown fences if present
    cleaned = re.sub(r"^```[a-z]*\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    cleaned = cleaned.strip()

    # Attempt JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and parsed:
            return [str(q).strip() for q in parsed if str(q).strip()]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: split on newlines, strip numbering / bullets
    lines = [
        re.sub(r"^[\d\.\-\*\)]+\s*", "", line).strip()
        for line in cleaned.splitlines()
        if line.strip()
    ]
    lines = [l for l in lines if l]
    if lines:
        return lines

    # Last resort
    return [f"What are the key aspects of: {goal}?"]


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

async def _deep_research(
    goal: str,
    criteria: str = "",
    max_questions: int = 5,
) -> dict:
    """
    Generate a structured research plan for a high-level goal.

    This tool is a scaffolding helper — it returns a plan dict that the agent
    should then execute by calling search_web, fetch_page, and log_research for
    each sub-question before producing a final ranked report.

    Args:
        goal:          High-level research goal, e.g.
                       "find ways to make money online that fit my skills".
        criteria:      Optional evaluation criteria, e.g.
                       "legal, no upfront cost, suitable for evenings".
                       The agent should filter / rank findings against these.
        max_questions: Maximum number of sub-questions to generate (1–10).
                       Defaults to 5.

    Returns a dict with:
        goal           — the original goal (echoed for clarity)
        criteria       — the original criteria (echoed)
        sub_questions  — list of specific research questions to pursue
        already_known  — relevant findings from long-term memory
        profile_context — user profile fields relevant to the task
        instruction    — natural-language instructions for the agent
    """
    max_questions = max(1, min(10, int(max_questions)))

    # ── 1. Generate sub-questions via local LLM ──────────────────────
    criteria_hint = f" Criteria to evaluate against: '{criteria}'." if criteria else ""
    prompt = (
        f"Generate {max_questions} specific, actionable research sub-questions "
        f"for this goal: '{goal}'.{criteria_hint} "
        f"Each question should target a distinct angle (e.g. platforms, income potential, "
        f"required skills, risks, getting started). "
        f"Return ONLY a JSON array of strings, no other text, no markdown fences."
    )

    # Read the local LLM config at call time (lazy read, same pattern as all other tools)
    config_path = pathlib.Path(__file__).parent.parent.parent / "config.json"
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}

    local_model   = cfg.get("llm", {}).get("local", "qwen2.5:14b")
    ollama_url    = cfg.get("ollama_base_url", "http://localhost:11434")

    raw_questions = ""
    try:
        raw_questions = await local_llm_call(
            prompt=prompt,
            model=local_model,
            base_url=ollama_url,
        )
    except Exception as e:
        logger.warning(f"[research_mode] Local LLM call failed (non-fatal): {e}")

    if raw_questions:
        sub_questions = _parse_sub_questions(raw_questions)[:max_questions]
    else:
        # Local LLM unavailable — generate simple generic questions
        sub_questions = [
            f"What are the most common approaches to: {goal}?",
            f"What are the requirements and risks for: {goal}?",
            f"What tools or platforms are commonly used for: {goal}?",
            f"How do beginners typically get started with: {goal}?",
            f"What realistic income or outcomes can be expected from: {goal}?",
        ][:max_questions]

    # ── 2. Read user profile ──────────────────────────────────────────
    profile = _load_profile()
    # Extract only fields that are plausibly relevant to research tasks
    # (skills, goals, constraints, tools) to keep the plan compact.
    relevant_keys = {
        "skills", "programming_languages", "tools", "goals", "constraints",
        "available_time", "budget", "location", "occupation", "interests",
        "projects", "hardware",
    }
    profile_context = {k: v for k, v in profile.items() if k in relevant_keys} if profile else {}

    # ── 3. Check long-term memory for prior research ──────────────────
    already_known = _query_known(goal)

    # ── 4. Build instruction string ───────────────────────────────────
    n = len(sub_questions)
    instruction = (
        f"You have {n} sub-question(s) to research. For each one:\n"
        "  1. Call search_web with a focused query.\n"
        "  2. Call fetch_page on the most relevant result URL.\n"
        "  3. Evaluate the findings against the criteria (if any).\n"
        "  4. Call log_research with topic=sub-question, findings=summary, sources=URLs.\n"
        "  5. Emit a task_progress step labelled 'Researching: N/{n}' before starting "
        "each sub-question so the user sees live progress.\n\n"
        "After all sub-questions are done, synthesise the findings into a ranked "
        "report and save it as a .md file using write_file. The report should:\n"
        "  • Rank options by fit against the criteria.\n"
        "  • Include a short 'getting started' note for the top options.\n"
        "  • Note any personal fit issues based on the profile_context.\n"
        "  • List sources for each finding."
    )

    return {
        "success":         True,
        "goal":            goal,
        "criteria":        criteria,
        "sub_questions":   sub_questions,
        "already_known":   already_known,
        "profile_context": profile_context,
        "instruction":     instruction,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_research_tools() -> None:
    register_tool(
        name="deep_research",
        description=(
            "Generate a structured multi-angle research plan for a high-level goal. "
            "Returns sub-questions, user profile context, and prior findings from memory. "
            "Use this as the first step for any broad research task (finding opportunities, "
            "comparing options, investigating a topic). After calling this tool, follow the "
            "returned instruction: search each sub-question, fetch relevant pages, log "
            "findings with log_research, then produce a final ranked .md report."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": (
                        "High-level research goal, e.g. "
                        "'find ways to make money online that fit my skills'."
                    ),
                },
                "criteria": {
                    "type": "string",
                    "description": (
                        "Optional evaluation criteria for ranking results, e.g. "
                        "'legal, no upfront cost, suitable for evenings'."
                    ),
                },
                "max_questions": {
                    "type": "integer",
                    "description": "Maximum number of sub-questions to generate (1–10). Default: 5.",
                    "default": 5,
                },
            },
            "required": ["goal"],
        },
        handler=_deep_research,
        is_destructive=False,
    )
    logger.info("[research_mode] Registered: deep_research")
