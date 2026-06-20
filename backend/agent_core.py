"""
agent_core.py  —  Agent Core

The "brain" of the agent. Owns the full request lifecycle:

  1. Optionally optimise the user's raw message via local LLM (prompt optimizer).
  2. Optionally classify intent (Phase 3d) to route to local / haiku / sonnet tier.
  3. If local_mode is enabled, delegate entirely to local_agent_call() (no Claude API).
     Otherwise, send the message to Claude with the tool definitions.
  4. If Claude requests tools, dispatch them (with permission checks for destructive ones).
     Before execute_code, optionally pre-validate the code (Phase 3d).
     After receiving tool results, optionally compress them (Phase 3d).
  5. Feed (compressed) tool results back to Claude and repeat until Claude gives a final answer.
  6. Stream events to the frontend at every step so the UI stays live.
  7. After any tool result that contains a "tree" key, broadcast a tree_update event
     so the frontend sidebar stays in sync.

Phase 3b additions:
  - _run_claude_once(): single Claude API call with no loop; used by TaskRunner.
  - run_with_task_runner(): entry point that delegates to task_runner.run_task()
    instead of _run_claude(), keeping the existing run() method intact as fallback.

Phase 3e additions:
  - _should_plan(): heuristic that decides whether to show a plan card.
  - _generate_plan(): one Haiku call that returns a structured step list.
  - _wait_for_plan_response(): waits for user to approve/reject the plan card.
  These are called from run_with_task_runner() before the prompt optimizer
  so the approved plan can be prepended to the system prompt.

Vague message enrichment (improvement):
  - VAGUE_PATTERNS: frozenset of short/ambiguous phrases ("continue", "yes", …).
  - _enrich_vague_message(): if the incoming message matches a vague pattern or
    is under 30 chars, prepend a 200-char snippet of the last assistant reply so
    the local optimizer and Claude understand what it refers to.  Called in both
    run() and run_with_task_runner() right after history is available and before
    intent classification.

Phase 3d additions:
  - Intent routing: classify_intent() decides which tier handles this turn before
    any Claude API call is made.
  - Tool result compression: compress_tool_result() strips verbose output so Claude's
    context stays lean.
  - Code pre-validation: prevalidate_code() catches obvious bugs before execute_code
    is actually invoked, saving a subprocess round-trip.
  - self._current_goal: set at the start of each run; passed to compression calls.
  - self.use_intent_routing / use_tool_compression / use_code_prevalidation: runtime
    flags mirrored from config, toggleable via the settings panel.

The core knows NOTHING about how tools work internally — it only calls handlers
from the registry. This keeps the architecture modular.
"""

import asyncio
import json
import logging
import re
from typing import Any, Callable, Awaitable
from uuid import uuid4

import anthropic

from agent_tools import get_all_definitions, get_handler, is_destructive as tool_is_destructive
from agent_tools.local_llm import (
    optimize_prompt,
    local_llm_call,
    local_agent_call,
    classify_intent,
    compress_tool_result,
    prevalidate_code,
)

logger = logging.getLogger(__name__)

# Tools whose output is worth compressing (verbose by nature)
_COMPRESSIBLE_TOOLS = {"search_web", "fetch_page", "get_system_info", "list_directory"}

# ---------------------------------------------------------------------------
# Improvement 1: Tiered system prompt
#
# Each entry maps a logical topic to (a) the regex keywords that trigger it
# and (b) the prompt text to append.  Sections are matched against the
# user's raw message (case-insensitive).  The base prompt is always sent;
# only relevant sections are appended so the total stays under 2 000 tokens.
#
# Priority order = dict insertion order.  When the cap is about to be
# exceeded, lower-priority (later) sections are skipped.
# ---------------------------------------------------------------------------

# ~4 chars per token is a conservative estimate good enough for budgeting.
_TOKEN_ESTIMATE_CHARS = 4
_SYSTEM_PROMPT_TOKEN_CAP = 2000

SYSTEM_PROMPT_SECTIONS: dict[str, dict] = {
    "software_dev": {
        "keywords": [
            r"build", r"scaffold", r"implement", r"project", r"develop",
            r"write.*script", r"create.*app", r"make.*tool", r"patch.*file",
        ],
        "content": (
            "\n\nSOFTWARE DEVELOPMENT WORKFLOW: For any multi-file project: "
            "(1) Call scaffold_project first — never skip this. Wait for user approval. "
            "(2) Call get_project_status to see what's pending. "
            "(3) Implement the next_file shown in status, save it to outputs/{project_name}/{filename}. "
            "(4) Call mark_file_complete after each file. "
            "(5) Call get_project_status again to confirm progress and get the next file. "
            "(6) Repeat steps 3-5 until ready_to_test is True. "
            "(7) Call run_project_test — if it fails, read stderr carefully, fix the specific "
            "file that caused the error, and run_project_test again. Iterate until passed is True. "
            "When calling run_project_test, never pass a full path as the command — "
            "use only the filename or short path relative to the project directory, "
            "e.g. 'python main.py' not 'python outputs/calc_cli/main.py'. "
            "The test runs inside the project directory already. "
            "(8) Report completion with a summary of what was built. "
            "Always use this exact sequence — it ensures nothing is missed and the project "
            "works before you declare it done. "
            "When editing an existing file, prefer patch_file over write_file — "
            "it modifies only the specified lines and preserves the rest. "
            "If a project requires packages not yet installed, call install_package for each "
            "dependency before running the project. "
            "RESUMING A PROJECT: When the user mentions a project by name "
            "(e.g. 'continue X', 'work on X', 'status of X'), ALWAYS call "
            "read_project_state(project_name) as the FIRST tool call. Never read individual "
            "files before checking the state snapshot. If state.json doesn't exist, call "
            "get_project_status instead. "
            "MID-TASK QUESTIONS: If you are unsure about a key decision (which API to use, "
            "which approach to take, whether to overwrite an existing file), call "
            "ask_user(question) to pause and get the user's input. Much better than guessing."
        ),
    },
    "github": {
        "keywords": [r"github", r"repo", r"repository", r"push", r"commit", r"issue"],
        "content": (
            "\n\nGITHUB WORKFLOW: After completing and testing a project successfully, "
            "offer to push it to GitHub. If the user agrees: "
            "(1) Call github_create_repo using the project name as the repo name. "
            "(2) Call github_push_file for each file in implementation_order, "
            "reading each file from outputs/{project_name}/ with read_file first. "
            "(3) After all files are pushed, update scaffold.json to set github_repo. "
            "Before creating a repo, call github_list_repos to check for duplicates."
        ),
    },
    "research": {
        "keywords": [r"research", r"find", r"search", r"investigate", r"analyze", r"compare"],
        "content": (
            "\n\nRESEARCH: For high-level research goals, use deep_research first to get a "
            "structured plan. Then follow the plan: search each sub-question, fetch relevant "
            "pages, evaluate findings, log each angle with log_research, and produce a final "
            "ranked report saved as a .md file. Always read the user profile before research "
            "tasks that depend on personal fit. After any information gathering, call "
            "log_research before sending your final answer."
        ),
    },
    "memory": {
        "keywords": [r"remember", r"recall", r"log", r"save.*finding", r"past.*task"],
        "content": (
            "\n\nMEMORY: Before starting any research or build task, call recall_memory first "
            "to check if relevant work was already done. Before scaffolding a project, call "
            "recall_projects. This prevents duplicate work and reuses past findings. "
            "After completing any research task you MUST call log_research. "
            "After learning any specific fact about the user or their system, call log_fact. "
            "Use recall_memory to retrieve past tasks and facts before starting a new task "
            "that might overlap with prior work."
        ),
    },
    "browser": {
        "keywords": [r"browser", r"open.*url", r"visit", r"website", r"navigate", r"screenshot",
                     r"click", r"fill", r"form", r"login", r"submit"],
        "content": (
            "\n\nBROWSER: browser_open(url) navigates to a page in headless Chromium, "
            "browser_read(selector) extracts visible text, browser_screenshot(filename) saves "
            "a PNG to outputs/. Use browser tools when fetch_page returns empty content — many "
            "modern sites require JavaScript. Always call browser_open before browser_read. "
            "Use 'body' as the default selector, or more specific CSS selectors for targeted "
            "extraction (e.g. 'article', 'main', '#content'). "
            "browser_click(selector) clicks an element — use CSS selectors or text selectors "
            "like 'button:has-text(\"Submit\")'. browser_fill(selector, value) fills an input "
            "field — set press_enter=True to submit forms directly. "
            "browser_get_url() returns the current page URL and title. "
            "Always call browser_read before clicking to understand the page structure. "
            "Write actions (click, fill) require user approval."
        ),
    },
    "credentials": {
        "keywords": [r"credential", r"api.?key", r"token", r"store_credential", r"secret"],
        "content": (
            "\n\nCREDENTIAL MANAGER: Use store_credential(service, value) to save API keys "
            "securely — never ask the user to paste keys directly into chat. "
            "Use get_credential(service) to retrieve a stored key. "
            "Use list_credentials() to see what is already stored before asking the user."
        ),
    },
    "youtube": {
        "keywords": [r"youtube", r"video", r"shorts", r"channel", r"trending", r"upload"],
        "content": (
            "\n\nYOUTUBE: youtube_search finds videos/channels/playlists by keyword. "
            "youtube_get_video_stats returns views, likes, duration. "
            "youtube_get_trending shows popular videos by region — use for content strategy. "
            "youtube_get_video_comments reads top comments for sentiment analysis. "
            "youtube_get_channel_info returns subscriber count and total views. "
            "All YouTube tools require an API key stored as 'youtube_api_key' via "
            "store_credential — call list_credentials first to check."
        ),
    },
    "scheduler": {
        "keywords": [
            r"schedule", r"recurring", r"every day", r"every week", r"every hour",
            r"every minute", r"automatically", r"remind", r"periodic", r"once at",
            r"daily", r"weekly", r"hourly",
        ],
        "content": (
            "\n\nSCHEDULER: Use schedule_task(task_id, message, schedule) to run tasks "
            "automatically at a given time or interval. "
            "task_id is a short unique name (e.g. 'weekly_research'). "
            "message is what the agent will do when triggered (same as if you typed it). "
            "schedule examples: 'every hour', 'every 30 minutes', 'every day at 09:00', "
            "'every monday at 08:00', 'once at 2026-07-15 10:00'. "
            "Use list_scheduled_tasks() to see what is currently scheduled. "
            "Use cancel_scheduled_task(task_id) to remove a scheduled task. "
            "Scheduled tasks run even when you are not chatting — they broadcast updates "
            "to all active WebSocket connections. "
            "Always confirm the schedule string with the user before calling schedule_task, "
            "since recurring tasks consume API tokens autonomously."
        ),
    },
    "process": {
        "keywords": [
            r"start", r"run.*app", r"launch", r"process", r"server",
            r"background", r"running",
        ],
        "content": (
            "\n\nPROCESS MANAGEMENT: Use start_process(name, command, cwd) to launch apps "
            "or scripts as persistent background processes. "
            "Use read_process_output(name) to see stdout. "
            "Use send_process_input(name, text) to send commands. "
            "Use stop_process(name) to terminate. "
            "Use list_processes() to see all running. "
            "When you scaffold a project, a control tool is auto-generated in "
            "agent_tools/generated/ — use it to start/stop/status that specific app. "
            "Example: after building youtube_automation, call start_youtube_automation() directly."
        ),
    },
    "self_extend": {
        "keywords": [
            r"write.*tool", r"new tool", r"create.*tool",
            r"add.*capability", r"extend yourself",
            r"context.*usage", r"context.*window", r"analyze.*performance",
            r"improvement.*suggestion", r"how.*performing", r"performance.*analysis",
        ],
        "content": (
            "\n\nSELF-EXTENSION: Use write_tool(filename, code) to create new tools — "
            "it validates syntax and imports before saving. "
            "Then call reload_tool(filename) to register it live. "
            "New tools go in agent_tools/generated/ and auto-load on restart. "
            "After writing a tool, always test it with a simple call before relying on it "
            "for important tasks. "
            "The control tools auto-generated for your projects "
            "(like start_youtube_automation) follow this same pattern.\n\n"
            "REGISTER_TOOL — ALWAYS use keyword arguments, never positional:\n\n"
            "register_tool(\n"
            "    name=\"tool_name\",\n"
            "    description=\"What this tool does.\",\n"
            "    input_schema={\"type\": \"object\", \"properties\": {\"param\": {\"type\": \"string\", \"description\": \"...\"}}, \"required\": [\"param\"]},\n"
            "    handler=tool_function,\n"
            "    is_destructive=False,\n"
            ")\n\n"
            "Common mistakes that cause crashes:\n"
            "- Passing the handler function as the second argument (description position) → JSON crash\n"
            "- Using a list instead of a dict for input_schema → registration failure\n"
            "- Using parameters= instead of input_schema= → now accepted but avoid it\n"
            "Always use keyword arguments. Always use input_schema=. Always use is_destructive=.\n\n"
            "CONTEXT AWARENESS: Use get_context_usage() periodically during very long "
            "tasks to check how much of the context window is in use. "
            "If warning_level is 'high' or 'critical', summarize completed steps and "
            "clear old conversation turns before continuing — otherwise the next API "
            "call may fail with a context length error.\n\n"
            "SELF-IMPROVEMENT: Use analyze_performance() to get AI-generated improvement "
            "suggestions based on your task history (stored in long_term.json). "
            "Run this periodically to identify patterns in what fails or takes too long. "
            "Requires at least 5 completed tasks. The analysis costs no Claude API tokens "
            "— it runs entirely through the local LLM."
        ),
    },
    "email_management": {
        "keywords": [
            r"email", r"inbox", r"gmail", r"imap",
            r"delete.*email", r"clean.*inbox", r"unsubscribe",
        ],
        "content": (
            "\n\nEMAIL MANAGEMENT: "
            "Use email_connect(host, username) first — for Gmail use host='imap.gmail.com' "
            "and store an App Password via store_credential('gmail_password'). "
            "Then email_scan_inbox() to find old emails (headers only, never body). "
            "Then email_classify_and_plan(email_ids) to get a DELETE/KEEP plan via the local model. "
            "ALWAYS show the user the full plan and get explicit approval before deleting. "
            "Call email_delete_batch(ids, dry_run=True) first so the user sees the preview, "
            "then email_delete_batch(ids, dry_run=False) ONLY after explicit user approval. "
            "Never delete emails without the user reviewing and approving the specific batch. "
            "Call email_disconnect() when finished."
        ),
    },
    "media": {
        "keywords": [
            r"video", r"audio", r"mp4", r"mp3", r"ffmpeg", r"convert.*video",
            r"extract.*audio", r"trim.*clip", r"merge.*clip", r"media",
        ],
        "content": (
            "\n\nMEDIA PROCESSING: Use get_media_info(path) to inspect a file first. "
            "convert_video(path, format, quality) re-encodes to mp4/mkv/webm with quality "
            "low/medium/high. extract_audio(path, format, bitrate) strips the audio track. "
            "trim_clip(path, start_seconds, end_seconds) cuts a segment using stream copy. "
            "merge_clips(paths_csv, output_filename) concatenates clips (same codec/resolution). "
            "All output goes to outputs/media/. Requires ffmpeg in PATH — call get_media_info "
            "first to verify ffmpeg is available."
        ),
    },
}


class AgentCore:
    def __init__(self, config: dict) -> None:
        self.config = config

        llm_cfg = config.get("llm", {})
        self.primary_model:     str  = llm_cfg.get("primary",     "claude-haiku-4-5")
        self.complex_model:     str  = llm_cfg.get("complex",     "claude-sonnet-4-6")
        self.local_model:       str  = llm_cfg.get("local",       "qwen3:14b")
        self.local_agent_model: str  = llm_cfg.get("local_agent", "qwen3:14b")

        self.use_prompt_optimizer:  bool = config.get("use_prompt_optimizer",  True)
        self.use_intent_routing:    bool = config.get("use_intent_routing",    True)
        self.use_tool_compression:  bool = config.get("use_tool_compression",  True)
        self.use_code_prevalidation: bool = config.get("use_code_prevalidation", True)
        self.use_tool_prefilter:    bool = config.get("use_tool_prefilter",    False)
        self.local_fallback:        bool = config.get("local_fallback",        True)
        self.local_mode:            bool = config.get("local_mode",            False)
        self.ollama_url:            str  = config.get("ollama_base_url",       "http://localhost:11434")

        # Phase 9 — LOCAL_SUFFICIENT tier
        # Tracks pending tier-choice banners waiting for a user response.
        self._pending_tier_choices: dict[str, dict] = {}
        # "ask"   — always show the banner (default)
        # "local" — always use the local model silently
        # "claude"— always route to Claude (effectively disables the tier)
        self.local_sufficient_default: str = config.get("local_sufficient_default", "ask")

        # Per-request timeout for the local agentic loop (large models need time on CPU)
        self.local_agent_timeout: float = float(config.get("local_agent_timeout", 1200))

        # Per-model max_tokens caps — read from config so they can be tuned without
        # touching code.  Research/planning tasks need significantly more tokens than
        # simple tool calls, so we use separate limits per model tier.
        llm_cfg = config.get("llm", {})  # already read above, but re-read here for clarity
        self.max_tokens_primary: int = int(llm_cfg.get("max_tokens_primary", 8192))
        self.max_tokens_complex: int = int(llm_cfg.get("max_tokens_complex", 16000))

        ctx_cfg = config.get("context", {})
        self.max_iterations: int = ctx_cfg.get("max_iterations_per_turn", 10)

        # AsyncAnthropic reads ANTHROPIC_API_KEY from the environment automatically.
        # Do NOT hardcode keys here; use a .env file or set the env var manually.
        self.client = anthropic.AsyncAnthropic()

        # Current user goal — set at the start of each run() / run_with_task_runner()
        # call so tool result compression has context about what the user wants.
        self._current_goal: str = ""

        self._base_system_prompt = (
            "You are a universal personal AI assistant. You are not specialized for any single project. "
            "Always read the user's intent fresh. Keep responses concise — avoid unnecessary markdown "
            "formatting, tables, or emoji in conversational replies. Use formatting only when it "
            "genuinely helps (structured data, code, comparisons).\n\n"
            "You are a capable personal AI agent with access to filesystem tools. "
            "Use tools whenever they would help complete the user's request. "
            "Before using a tool, briefly state what you're about to do. "
            "After getting tool results, synthesise them into a clear final answer. "
            "Be concise. Avoid unnecessary preamble.\n\n"
            "You have access to an execute_code tool that runs Python or Bash directly "
            "on the host machine. Follow these rules when using it:\n"
            "1. ALWAYS pass the complete source code inside the 'code' argument of the "
            "execute_code tool call itself. Never show code in your text and then call "
            "execute_code separately — the code must be in the tool input JSON, not in "
            "your message text.\n"
            "2. Call execute_code directly. Do NOT call write_file first to save the "
            "script — that wastes a permission approval. Pass the code straight to "
            "execute_code.\n"
            "3. Always read both stdout AND stderr from the result before concluding "
            "success or failure.\n"
            "4. If the user denies an execute_code call, do NOT assume the environment "
            "is restricted or the code failed. Acknowledge the denial and ask if they "
            "want to approve the execution or try a different approach.\n"
            "5. If a run fails (non-zero exit code or stderr), read the error, fix the "
            "code, and call execute_code again. Do not give up after one failure.\n\n"
            "You can write and register new tools using write_tool and reload_tool. "
            "Tools must be written as Python async functions following this template: "
            "async def tool_name(param: str) -> dict — always return a dict with at least "
            "a 'success' key. Include a register_<toolname>_tools() function that calls "
            "register_tool() from agent_tools. "
            "After writing a tool with write_tool, always call reload_tool to activate it. "
            "New tools are saved to agent_tools/generated/ and persist across restarts. "
            "agent_core.py and main.py are read-only to you — only modify files in agent_tools/generated/.\n\n"
            "Save all generated files (reports, scripts, data, screenshots) to the "
            "outputs/ directory in the project root, not to backend/ or frontend/. "
            "Use write_file with a path like 'outputs/report.md' for text files. "
            "Use list_outputs() at the start of any task that might produce files, "
            "to check what already exists and avoid duplicating work.\n\n"
            # iGPU acceleration note — keep in sync with config.json llm.local
            "The local preprocessing model (qwen3:14b) runs on an Intel Arc 140V iGPU "
            "via ipex-llm Ollama and is fast — typical responses in 5-15 seconds. "
            "Use LOCAL_SUFFICIENT routing generously for tasks that don't require "
            "frontier reasoning: reading files, simple scripts, memory queries, "
            "email classification, single web searches, summarisation."
        )

        # Phase 3g: Always inject user profile into the base system prompt at
        # init time so Claude has personal context in every conversation without
        # needing a tool call or keyword-based detection.
        # Falls back silently if the file is missing or malformed.
        import json as _json
        import pathlib as _pathlib
        _profile_path = _pathlib.Path(__file__).resolve().parent.parent / "memory" / "user_profile.json"
        if _profile_path.exists():
            try:
                _profile = _json.loads(_profile_path.read_text(encoding="utf-8"))
                self._base_system_prompt += (
                    "\n\nUser profile (always available — no tool call needed):\n"
                    + _json.dumps(_profile, indent=2)
                )
                import logging as _logging
                _logging.getLogger(__name__).info("[agent] User profile loaded into base system prompt.")
            except Exception as _e:
                import logging as _logging
                _logging.getLogger(__name__).warning(f"[agent] Could not load user profile: {_e}")

    # ------------------------------------------------------------------
    # Improvement 1: Tiered system prompt builder
    # ------------------------------------------------------------------

    def _build_system_prompt(self, message: str) -> str:
        """
        Start with the always-sent base prompt and append only the contextual
        sections whose keywords match the user's message.

        Token budget: base + sections must stay under _SYSTEM_PROMPT_TOKEN_CAP.
        If adding a section would bust the cap, it is skipped (lower-priority
        sections, i.e. later entries in SYSTEM_PROMPT_SECTIONS, are skipped first
        because the dict is iterated in insertion order).

        Args:
            message: The raw (or enriched) user message for this turn.

        Returns:
            The assembled system prompt string.
        """
        prompt = self._base_system_prompt
        used_tokens = len(prompt) // _TOKEN_ESTIMATE_CHARS
        selected: list[str] = []

        for section_name, section in SYSTEM_PROMPT_SECTIONS.items():
            # Check whether any keyword regex matches the message
            matched = any(
                re.search(kw, message, re.IGNORECASE)
                for kw in section["keywords"]
            )
            if not matched:
                continue

            section_tokens = len(section["content"]) // _TOKEN_ESTIMATE_CHARS
            if used_tokens + section_tokens > _SYSTEM_PROMPT_TOKEN_CAP:
                logger.debug(
                    f"[agent] System prompt cap reached — skipping section '{section_name}' "
                    f"({section_tokens} tokens would exceed cap of {_SYSTEM_PROMPT_TOKEN_CAP})."
                )
                continue

            prompt += section["content"]
            used_tokens += section_tokens
            selected.append(section_name)

        logger.debug(f"[agent] System prompt sections: {selected} ({used_tokens} est. tokens)")
        return prompt

    # ------------------------------------------------------------------
    # Phase 3g: Self-directed task detection
    # ------------------------------------------------------------------

    _SELF_DIRECTED_KEYWORDS = frozenset([
        "yourself", "your capabilities", "what can you",
        "optimize yourself", "improve yourself", "learn about me",
        "about me", "my profile", "scan my", "what do you know about me",
    ])

    def _is_self_directed(self, message: str) -> bool:
        """
        Return True if the message implies the agent should act on its own
        initiative or learn about the user.

        Triggers profile injection into the system prompt so the agent has
        full user context without needing an extra tool round-trip.
        """
        lower = message.lower()
        return any(kw in lower for kw in self._SELF_DIRECTED_KEYWORDS)

    # ------------------------------------------------------------------
    # Vague message enrichment
    # ------------------------------------------------------------------

    # Short phrases that are meaningless without knowing what came before.
    # Matched case-insensitively against the stripped, punctuation-trimmed message.
    VAGUE_PATTERNS: frozenset[str] = frozenset({
        "continue", "proceed", "yes", "no", "ok", "okay", "sure",
        "do it", "go ahead", "go on", "next", "keep going",
        "finish it", "complete it", "finish", "done", "stop",
    })

    def _enrich_vague_message(self, message: str, history: list) -> str:
        """
        If the message is short and ambiguous, prepend recent context so the
        local optimizer and Claude understand what it refers to.
        Without this, 'continue' after a 35-minute build task produces
        'Continue what?' from a cold local model.

        Enrichment is added as a bracketed inline note — it does NOT replace
        the original message so the user's exact wording is still visible.

        Short-circuits immediately when:
          - The message is neither vague by length (< 30 chars) nor in
            VAGUE_PATTERNS, so normal messages are untouched.
          - History is empty (nothing to reference).
          - No previous assistant turn exists in history.
        """
        stripped = message.strip().lower().rstrip(".,!")
        is_vague = len(message.strip()) < 30 or stripped in self.VAGUE_PATTERNS
        if not is_vague or not history:
            return message

        # Find the most recent assistant message for context.
        # Use str() to handle both plain-string content and list-of-block content.
        last_assistant = next(
            (h["content"] for h in reversed(history) if h.get("role") == "assistant"),
            "",
        )
        if not last_assistant:
            return message

        # Keep the snippet short — just enough context, not a wall of text.
        context_snippet = str(last_assistant)[:200].replace("\n", " ")
        enriched = (
            f"{message} "
            f"[Context: the assistant last said: '{context_snippet}...']"
        )
        logger.debug(
            f"[agent] Vague message enriched: {message!r} → {enriched[:80]!r}"
        )
        return enriched

    # ------------------------------------------------------------------
    # Phase 3e: Task planning
    # ------------------------------------------------------------------

    _PLAN_KEYWORDS = frozenset([
        "create", "build", "write", "research", "find", "develop",
        "make", "analyze", "compare", "investigate", "optimize",
        "design", "implement",
    ])

    def _should_plan(self, message: str, intent: str) -> bool:
        """
        Return True when the message is a complex, action-oriented task that
        benefits from an explicit numbered plan shown to the user before
        execution begins.

        Rules (all must pass):
          1. intent == "COMPLEX"      — routing already classified it as hard
          2. len(message) > 80        — short messages are rarely multi-step
          3. at least one action verb present — filters out explanatory requests
             ("explain async/await in depth") that are COMPLEX but not task-like
        """
        if intent != "COMPLEX":
            return False
        if len(message) <= 80:
            return False
        lower = message.lower()
        return any(kw in lower for kw in self._PLAN_KEYWORDS)

    async def _generate_plan(
        self,
        message: str,
        send_event: Callable[[str, dict], Awaitable[None]],
        plan_id: str,
    ) -> list[dict]:
        """
        Ask Claude (primary model — Haiku is fine, it's cheap) to produce a
        numbered plan for the task.  Returns a list of step dicts, or an
        empty list if generation or parsing fails.

        On success, sends a ``task_plan`` WebSocket event to the frontend so
        the approval card can be rendered immediately.
        """
        system_prompt = (
            "You are a task planner. Given a user request, output ONLY a valid "
            "JSON array of steps with no other text, no markdown fences, no "
            "explanation. Each step must be: "
            "{\"step\": N, \"action\": \"short verb phrase max 6 words\", "
            "\"details\": \"one sentence describing what will be done and why\"}. "
            "Maximum 8 steps. If the task is simple enough to do in 1-2 steps, "
            "output only those steps."
        )
        try:
            response = await self.client.messages.create(
                model=self.primary_model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": f"Plan this task: {message}"}],
            )
            raw = _extract_text(response)
            # Strip accidental markdown fences before parsing
            raw = re.sub(r"^```[a-z]*\s*", "", raw.strip(), flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw.strip())
            steps = json.loads(raw)
            if not isinstance(steps, list) or not steps:
                return []
            # Normalise: ensure each entry has the expected keys
            normalised = []
            for i, s in enumerate(steps, start=1):
                normalised.append({
                    "step":    int(s.get("step", i)),
                    "action":  str(s.get("action", f"Step {i}")),
                    "details": str(s.get("details", "")),
                })
            await send_event("task_plan", {"plan_id": plan_id, "steps": normalised})
            return normalised
        except Exception as exc:
            # Planning is optional — never let it block the task.
            logger.warning(f"[agent] Plan generation failed (non-fatal): {exc}")
            return []

    async def _wait_for_plan_response(
        self,
        plan_id: str,
        pending_plans: dict,
        timeout: float = 300.0,
    ) -> tuple[bool, list]:
        """
        Block until the user approves or rejects the plan card, or until the
        5-minute timeout elapses.

        Returns:
            (approved: bool, edited_steps: list)
            edited_steps is non-empty only when the user edited the plan before
            approving; on rejection it is always [].
        """
        event = asyncio.Event()
        pending_plans[plan_id] = {"event": event, "result": None}
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            result = pending_plans[plan_id].get("result") or {}
            approved     = bool(result.get("approved", False))
            edited_steps = result.get("edited_steps") or []
            return approved, edited_steps
        except asyncio.TimeoutError:
            logger.warning(f"[agent] Plan response for '{plan_id}' timed out after {timeout}s.")
            return False, []
        finally:
            pending_plans.pop(plan_id, None)

    # ------------------------------------------------------------------
    # Phase 9: LOCAL_SUFFICIENT tier choice
    # ------------------------------------------------------------------

    async def _request_tier_choice(
        self,
        message_id: str,
        message_preview: str,
        send_event: Callable[[str, dict], Awaitable[None]],
    ) -> str:
        """
        When the local model determines the local tier is sufficient,
        ask the user whether to use local (slower, free) or Claude (faster, costs tokens).

        Behaviour is controlled by self.local_sufficient_default:
            "ask"   — send a tier_choice event and wait up to 15 s for the user's pick
            "local" — silently always use the local model
            "claude"— silently always route to Claude (feature is disabled)

        Returns "local" or "claude".
        """
        if self.local_sufficient_default != "ask":
            choice = self.local_sufficient_default
            return choice if choice in ("local", "claude") else "claude"

        event = asyncio.Event()
        self._pending_tier_choices[message_id] = {
            "event":  event,
            "choice": self.local_sufficient_default,  # fallback if timeout
        }

        await send_event("tier_choice", {
            "message_id":      message_id,
            "message_preview": message_preview[:60],
            "default":         self.local_sufficient_default,
            "timeout_seconds": 15,
        })

        try:
            await asyncio.wait_for(event.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.info("[agent] Tier choice timed out — defaulting to Claude")

        choice = self._pending_tier_choices.pop(message_id, {}).get("choice", "claude")
        # Ensure timeout/default never produces "ask" as a model name
        if choice not in ("local", "claude"):
            choice = "claude"
        logger.info(f"[agent] Tier choice resolved → {choice!r}")
        return choice

    def resolve_tier_choice(self, message_id: str, use_local: bool) -> None:
        """
        Called by the WebSocket handler when the frontend sends a tier_response message.
        Sets the choice and unblocks _request_tier_choice().
        """
        if message_id in self._pending_tier_choices:
            self._pending_tier_choices[message_id]["choice"] = "local" if use_local else "claude"
            self._pending_tier_choices[message_id]["event"].set()
        else:
            logger.warning(f"[agent] resolve_tier_choice: unknown message_id {message_id!r}")

    # ------------------------------------------------------------------
    # Public entry point — called from main.py WebSocket handler
    # (original; kept intact for backward compatibility)
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
        Process a single user turn end-to-end (original bounded loop).

        Args:
            user_message:           Raw text from the user.
            history:                Recent turns as [{role, content}].
            send_event:             Async callback — sends a typed event to the frontend.
            pending_confirmations:  Shared dict for the permission layer.
            context_summary:        Optional context note injected into the system prompt.

        Returns:
            The final text reply from the agent.
        """
        # Store the raw user message as the current goal for compression context.
        self._current_goal = user_message

        # ── Vague message enrichment ───────────────────────────────────────────
        # Prepend recent assistant context to short/ambiguous messages so the
        # intent router and prompt optimizer have enough signal to work with.
        # Must run before intent classification so "continue" routes correctly.
        user_message = self._enrich_vague_message(user_message, history)

        # ----------------------------------------------------------------
        # Step 1: Intent routing (Phase 3d) — runs FIRST on the raw message.
        # ----------------------------------------------------------------
        # Classifying before optimization means SIMPLE queries never pay the
        # optimizer cost at all.  For TOOL/COMPLEX the optimizer still runs
        # below (step 2) before the Claude API call.
        #
        # SIMPLE  → answer locally right now, return immediately
        # TOOL    → proceed with primary model (haiku) + optimizer
        # COMPLEX → proceed with complex model (sonnet) + optimizer
        model_override = None
        intent = "TOOL"  # default when routing is disabled
        if self.use_intent_routing and not self.local_mode:
            await send_event("status", {"text": "Routing intent…"})
            intent = await classify_intent(
                message=user_message,
                model=self.local_model,
                base_url=self.ollama_url,
            )

            # Improvement 2: Short affirmatives must never be answered locally —
            # they almost always refer to an ongoing tool-based task and need Claude.
            SHORT_AFFIRMATIVES = {
                "yes", "no", "ok", "okay", "sure", "proceed", "continue", "go",
                "do it", "go ahead", "go on", "next", "keep going", "finish",
                "complete it", "done", "stop", "y", "yep", "yeah", "nope",
                "correct", "right", "good", "great", "perfect", "sounds good",
            }
            stripped = user_message.strip().lower().rstrip(".,!")
            if intent == "SIMPLE" and (stripped in SHORT_AFFIRMATIVES or len(user_message.strip()) < 15):
                intent = "TOOL"
                logger.debug(f"[agent] Short affirmative override: SIMPLE → TOOL ('{stripped}')")

            if intent == "SIMPLE":
                # Bug fix (Phase 3g): self-directed tasks must never be answered
                # locally — the local model has no access to the user profile.
                if self._is_self_directed(user_message):
                    intent = "TOOL"
                    await send_event("status", {"text": "Self-directed — overriding SIMPLE → TOOL"})
                else:
                    await send_event("status", {"text": "Intent: SIMPLE → answering locally…"})
                    # Answer directly — no optimizer, no Claude API call
                    local_answer = await local_llm_call(
                        prompt=user_message,
                        model=self.local_model,
                        base_url=self.ollama_url,
                    )
                    if local_answer:
                        await send_event("message", {"text": local_answer, "source": "local"})
                        return local_answer
                    # Ollama offline or empty — fall through to Claude
                    logger.warning("[agent] SIMPLE intent but local LLM unavailable, falling through to Claude.")
                    await send_event("status", {"text": "Local unavailable — using Claude…"})

            elif intent == "LOCAL_SUFFICIENT":
                message_id = str(uuid4())
                tier = await self._request_tier_choice(message_id, user_message, send_event)
                if tier == "local":
                    await send_event("status", {"text": "Running locally (no Claude API)…"})
                    # Build minimal messages list and run the local agentic loop
                    _local_messages = history + [{"role": "user", "content": user_message}]
                    return await self._run_local(_local_messages, send_event, pending_confirmations)
                # else: user chose Claude — fall through treating as TOOL
                intent = "TOOL"
                await send_event("status", {"text": f"Using Claude for this task…"})

            if intent == "COMPLEX":
                model_override = self.complex_model
                await send_event("status", {
                    "text": f"Intent: COMPLEX → routing to {self.complex_model}…"
                })
            elif intent == "TOOL":
                await send_event("status", {
                    "text": f"Intent: TOOL → routing to {self.primary_model}…"
                })

        # ----------------------------------------------------------------
        # Step 2: Prompt optimisation — only for TOOL / COMPLEX intents.
        # SIMPLE already returned above; optimizing a trivial message wastes
        # a full model load/unload cycle with zero quality benefit.
        # ----------------------------------------------------------------
        optimized_message = user_message
        if self.use_prompt_optimizer and intent != "SIMPLE":
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
        # Step 3: Build system prompt — append context summary if present
        # ----------------------------------------------------------------
        system = self._build_system_prompt(user_message)
        if context_summary:
            system = f"{system}\n\n{context_summary}"

        # Phase 3g: Profile is already in _base_system_prompt (injected at init).

        # ----------------------------------------------------------------
        # Step 4: Build message list
        # ----------------------------------------------------------------
        messages = history + [{"role": "user", "content": optimized_message}]

        # ----------------------------------------------------------------
        # Step 5: Route to local mode or Claude API
        # ----------------------------------------------------------------
        if self.local_mode:
            return await self._run_local(messages, send_event, pending_confirmations)
        else:
            return await self._run_claude(
                messages, system, send_event, pending_confirmations,
                optimized_message, model_override=model_override,
            )

    # ------------------------------------------------------------------
    # Phase 3b: Task-runner entry point
    # ------------------------------------------------------------------

    async def run_with_task_runner(
        self,
        task_runner,                             # TaskRunner instance
        user_message: str,
        history: list[dict],
        send_event: Callable[[str, dict], Awaitable[None]],
        pending_confirmations: dict,
        context_summary: str = "",
        pending_plans: dict | None = None,       # Phase 3e
    ) -> str:
        """
        Entry point for the Phase 3b task runner.

        Mirrors the setup logic of run() (prompt optimisation, intent routing,
        system prompt, message list assembly, local-mode routing) but then
        delegates the agentic loop to task_runner.run_task() instead of
        _run_claude().

        Phase 3d: intent routing applied here too. SIMPLE intents are answered
        locally before the task runner is ever invoked. COMPLEX intents are
        signalled to task_runner via agent.complex_model so it can pass the
        right model to _run_claude_once().

        Phase 3e: if the intent is COMPLEX and the message is action-oriented,
        a plan is generated, shown to the user for approval, and—if approved—
        prepended to the system prompt.  planning_plans dict is passed in from
        main.py so the WebSocket handler can resolve approvals.
        """
        # Ensure pending_plans is always a dict even when not supplied
        if pending_plans is None:
            pending_plans = {}

        # Phase 3f bug fix: initialise plan_system_addon unconditionally so it
        # is always defined when we reach the system-prompt assembly block below,
        # regardless of whether _should_plan() returns True or False.
        plan_system_addon: str = ""

        # Store goal for compression context
        self._current_goal = user_message

        # ── Vague message enrichment ───────────────────────────────────────────
        # Same as run(): enrich before intent routing so the classifier sees
        # a meaningful message even when the user types only "continue" or "yes".
        user_message = self._enrich_vague_message(user_message, history)

        # ── Intent routing (Phase 3d) — runs FIRST on the raw message ─
        # SIMPLE queries return immediately without ever running the optimizer.
        model_override = None
        intent = "TOOL"  # default when routing is disabled
        if self.use_intent_routing and not self.local_mode:
            await send_event("status", {"text": "Routing intent…"})
            intent = await classify_intent(
                message=user_message,
                model=self.local_model,
                base_url=self.ollama_url,
            )

            # Improvement 2: Short affirmatives must never be answered locally —
            # they almost always refer to an ongoing tool-based task and need Claude.
            _SHORT_AFFIRMATIVES = {
                "yes", "no", "ok", "okay", "sure", "proceed", "continue", "go",
                "do it", "go ahead", "go on", "next", "keep going", "finish",
                "complete it", "done", "stop", "y", "yep", "yeah", "nope",
                "correct", "right", "good", "great", "perfect", "sounds good",
            }
            _stripped = user_message.strip().lower().rstrip(".,!")
            if intent == "SIMPLE" and (_stripped in _SHORT_AFFIRMATIVES or len(user_message.strip()) < 15):
                intent = "TOOL"
                logger.debug(f"[agent] Short affirmative override: SIMPLE → TOOL ('{_stripped}')")

            if intent == "SIMPLE":
                # Bug fix (Phase 3g): self-directed tasks must never be answered
                # locally — the local model has no access to the user profile.
                if self._is_self_directed(user_message):
                    intent = "TOOL"
                    await send_event("status", {"text": "Self-directed — overriding SIMPLE → TOOL"})
                else:
                    await send_event("status", {"text": "Intent: SIMPLE → answering locally…"})
                    local_answer = await local_llm_call(
                        prompt=user_message,
                        model=self.local_model,
                        base_url=self.ollama_url,
                    )
                    if local_answer:
                        await send_event("message", {"text": local_answer, "source": "local"})
                        return local_answer
                    logger.warning("[agent] SIMPLE intent but local LLM unavailable, falling through.")
                    await send_event("status", {"text": "Local unavailable — using Claude…"})

            elif intent == "LOCAL_SUFFICIENT":
                message_id = str(uuid4())
                tier = await self._request_tier_choice(message_id, user_message, send_event)
                if tier == "local":
                    await send_event("status", {"text": "Running locally (no Claude API)…"})
                    _local_messages = history + [{"role": "user", "content": user_message}]
                    return await self._run_local(_local_messages, send_event, pending_confirmations)
                # else: user chose Claude — fall through treating as TOOL
                intent = "TOOL"
                await send_event("status", {"text": f"Using Claude for this task…"})

            if intent == "COMPLEX":
                model_override = self.complex_model
                await send_event("status", {
                    "text": f"Intent: COMPLEX → routing to {self.complex_model}…"
                })
            elif intent == "TOOL":
                await send_event("status", {
                    "text": f"Intent: TOOL → routing to {self.primary_model}…"
                })

        # ── Phase 3e: Task planning ────────────────────────────────────
        # Only triggered when intent is COMPLEX and the message is action-
        # oriented (checked by _should_plan).  Planning never blocks a task —
        # if generation fails or the user rejects the plan we proceed as normal.
        plan_system_addon = ""
        if not self.local_mode and self._should_plan(user_message, intent):
            plan_id = str(uuid4())
            await send_event("status", {"text": "Generating task plan…"})
            steps = await self._generate_plan(user_message, send_event, plan_id)
            if steps:
                # Wait for user to approve / reject / edit
                await send_event("status", {"text": "Waiting for plan approval…"})
                approved, edited_steps = await self._wait_for_plan_response(
                    plan_id, pending_plans
                )
                if not approved:
                    await send_event("status", {"text": "Plan cancelled."})
                    return "Plan cancelled by user."
                # Use edited steps if the user changed anything, otherwise use generated
                final_steps = edited_steps if edited_steps else steps
                # Build a numbered plan text to prepend to the system prompt
                plan_lines = [
                    f"{s.get('step', i+1)}. {s.get('action','')}: {s.get('details','')}"
                    for i, s in enumerate(final_steps)
                ]
                plan_text = "\n".join(plan_lines)
                plan_system_addon = (
                    f"\n\nExecute this plan step by step, in order:\n{plan_text}\n\n"
                    "Do not skip steps. After completing each step, briefly confirm "
                    "what was done before starting the next."
                )

        # ── Prompt optimisation — only for TOOL / COMPLEX intents ─────
        # SIMPLE already returned above; no point optimizing a trivial message.
        optimized_message = user_message
        if self.use_prompt_optimizer and intent != "SIMPLE":
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

        # ── System prompt ──────────────────────────────────────────────
        system = self._build_system_prompt(user_message)
        if context_summary:
            system = f"{system}\n\n{context_summary}"
        # Phase 3e: append the approved plan (empty string = no change)
        if plan_system_addon:
            system = system + plan_system_addon

        # Phase 3f: prepend relevant past-session context (tasks / facts / research)
        # so Claude is aware of prior attempts without needing an extra tool call.
        try:
            from memory.long_term import get_context_summary
            past_context = get_context_summary(user_message)
            if past_context:
                system += f"\n\n{past_context}"
                logger.info(f"[agent] Long-term context injected ({len(past_context)} chars).")
        except Exception as _lt_err:
            logger.warning(f"[agent] Long-term context lookup failed (non-fatal): {_lt_err}")

        # Phase 3g: Profile is already in _base_system_prompt (injected at init).

        # ── Message list ───────────────────────────────────────────────
        messages = history + [{"role": "user", "content": optimized_message}]

        # ── Routing ────────────────────────────────────────────────────
        if self.local_mode:
            return await self._run_local(messages, send_event, pending_confirmations)

        return await task_runner.run_task(
            initial_message=user_message,
            messages=messages,
            system=system,
            send_event=send_event,
            pending_confirmations=pending_confirmations,
            agent=self,
            model_override=model_override,
        )

    # ------------------------------------------------------------------
    # Phase 3b: Single Claude API call (no loop, no event sending)
    # Used by TaskRunner to make exactly one round-trip per iteration.
    # Phase 3d: accepts optional model_override for COMPLEX intent routing.
    # ------------------------------------------------------------------

    async def _run_claude_once(
        self,
        messages: list[dict],
        system: str,
        model_override: str | None = None,
    ) -> anthropic.types.Message:
        """
        Make a single call to the Claude API and return the raw response.

        Raises API exceptions as-is so the caller (TaskRunner) can decide
        how to handle them (fallback, error event, etc.).

        Args:
            model_override: If set, use this model instead of self.primary_model.
                            Used by Phase 3d intent routing to invoke claude-sonnet-4-6
                            for COMPLEX intents without permanently changing the default.
        """
        # Fix 1: Repair any orphaned tool_use blocks before every API call.
        # This guards _run_claude() (the bounded fallback loop) in addition to
        # TaskRunner, which calls this method directly.
        messages = self._sanitize_messages(messages)

        model = model_override if model_override else self.primary_model
        # Use the larger cap when running the complex model (sonnet) — research and
        # planning tasks frequently generate more than 4096 tokens in a single turn.
        max_tok = (
            self.max_tokens_complex
            if model == self.complex_model
            else self.max_tokens_primary
        )

        # ── Tool pre-filter (optional, off by default) ─────────────────────
        # When enabled, ask the local LLM to select only the tools relevant to
        # the current goal before sending definitions to Claude.  Reduces input
        # tokens by ~40-60%.  Disabled in local_mode (no Claude call anyway).
        if self.use_tool_prefilter and not self.local_mode:
            from agent_tools.local_llm import select_relevant_tools
            all_defs = get_all_definitions()
            all_names = [t["name"] for t in all_defs]
            relevant_names = await select_relevant_tools(
                user_message=self._current_goal or "",
                all_tool_names=all_names,
                model=self.local_model,
                base_url=self.ollama_url,
                max_tools=12,
            )
            tools = [t for t in all_defs if t["name"] in relevant_names]
        else:
            tools = get_all_definitions()

        return await self.client.messages.create(
            model=model,
            max_tokens=max_tok,
            system=system,
            tools=tools,
            messages=messages,
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
            timeout=self.local_agent_timeout,
        )

        await send_event("message", {"text": final_text, "source": "local"})
        return final_text

    # ------------------------------------------------------------------
    # Claude agentic loop (original bounded version — kept as fallback)
    # Phase 3d: accepts model_override for intent-based tier selection.
    # ------------------------------------------------------------------

    async def _run_claude(
        self,
        messages: list[dict],
        system: str,
        send_event: Callable,
        pending_confirmations: dict,
        optimized_message: str,
        model_override: str | None = None,
    ) -> str:
        """The original Claude API agentic loop (bounded by max_iterations)."""
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            await send_event("status", {"text": "Thinking…"})

            try:
                response = await self._run_claude_once(
                    messages, system, model_override=model_override
                )
                # Only override the first call — subsequent iterations use primary
                model_override = None
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
    # Phase 3d: code pre-validation before execute_code
    #           tool result compression before returning to Claude
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

        # ── Auto-approve bypass for execute_code ───────────────────────
        # If the user has enabled auto_approve_code_execution in config,
        # skip the confirmation modal for execute_code only. All other
        # destructive tools still require approval as normal.
        if tool_name == "execute_code" and self.config.get("auto_approve_code_execution", False):
            # Jump straight to pre-validation + dispatch — no confirmation prompt.
            if self.use_code_prevalidation:
                code     = tool_input.get("code", "")
                language = tool_input.get("language", "python")
                valid, issue = await prevalidate_code(
                    code=code,
                    language=language,
                    intent=self._current_goal,
                    model=self.local_model,
                    base_url=self.ollama_url,
                )
                if not valid:
                    await send_event("status", {
                        "text": f"Pre-validation caught an issue: {issue} — asking agent to fix…"
                    })
                    synthetic_result = {
                        "success":   False,
                        "stdout":    "",
                        "stderr":    (
                            f"Pre-validation caught an issue before execution: {issue}. "
                            "Please fix the code and retry."
                        ),
                        "exit_code": -1,
                        "language":  language,
                    }
                    await send_event("tool_result", {
                        "tool":    tool_name,
                        "success": False,
                        "result":  synthetic_result,
                    })
                    return _make_tool_result(tool_use_id, synthetic_result)
            handler = get_handler(tool_name)
            if handler is None:
                result = {"success": False, "error": f"Unknown tool: {tool_name}"}
            else:
                result = await handler(**tool_input)
            await send_event("tool_result", {"tool": tool_name, "result": result, "tool_use_id": tool_use_id})
            return _make_tool_result(tool_use_id, result)

        # ── Permission check ───────────────────────────────────────────
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

        # ── Phase 3d: code pre-validation ─────────────────────────────
        # For execute_code calls, ask the local LLM to review the code for
        # obvious bugs before we spin up a subprocess. This saves an execution
        # round-trip when the agent has generated broken code.
        if tool_name == "execute_code" and self.use_code_prevalidation:
            code     = tool_input.get("code", "")
            language = tool_input.get("language", "python")
            valid, issue = await prevalidate_code(
                code=code,
                language=language,
                intent=self._current_goal,
                model=self.local_model,
                base_url=self.ollama_url,
            )
            if not valid:
                await send_event("status", {
                    "text": f"Pre-validation caught an issue: {issue} — asking agent to fix…"
                })
                synthetic_result = {
                    "success":   False,
                    "stdout":    "",
                    "stderr":    (
                        f"Pre-validation caught an issue before execution: {issue}. "
                        "Please fix the code and retry."
                    ),
                    "exit_code": -1,
                    "language":  language,
                }
                # Send the synthetic failure as a tool_result event so the UI
                # shows what happened, then return it to Claude without executing.
                await send_event("tool_result", {
                    "tool":    tool_name,
                    "success": False,
                    "result":  synthetic_result,
                })
                return _make_tool_result(tool_use_id, synthetic_result)

        # ── Dispatch to handler ────────────────────────────────────────
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

        # Always send the FULL uncompressed result to the UI so the user sees everything.
        await send_event("tool_result", {
            "tool":    tool_name,
            "success": result.get("success", False),
            "result":  result,
        })

        # Tree broadcast: if the tool result contains a "tree" key (filesystem tools),
        # forward it as a separate WebSocket event for the sidebar.
        if isinstance(result, dict) and "tree" in result:
            await send_event("tree_update", {"tree": result["tree"]})

        # ── Phase 3d: tool result compression ─────────────────────────
        # Compress verbose results before appending to Claude's context window.
        # The user-facing tool_result event above already carries the full output.
        claude_result = result
        if self.use_tool_compression and tool_name in _COMPRESSIBLE_TOOLS:
            claude_result = await compress_tool_result(
                tool_name=tool_name,
                result=result,
                user_goal=self._current_goal,
                model=self.local_model,
                base_url=self.ollama_url,
            )

        return _make_tool_result(tool_use_id, claude_result)

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

    # ------------------------------------------------------------------
    # Fix 1: Pre-flight history sanitizer (moved here from TaskRunner so
    # _run_claude_once can use it too — runs before every API call).
    # TaskRunner.run_task() now delegates to this method instead of its
    # own copy, keeping the repair logic in one place.
    # ------------------------------------------------------------------

    def _sanitize_messages(self, messages: list) -> list:
        """
        Pre-flight check before every Anthropic API call.

        If the last assistant message contains tool_use blocks with no following
        tool_result message (orphaned — caused by 429/network errors mid-response),
        inject synthetic tool_result messages so the API doesn't return 400.

        Handles both live SDK objects (block.type attribute) and plain dict entries
        so it works in both the bounded loop (_run_claude) and TaskRunner.run_task().
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

        logger.warning(
            f"[agent] Detected {len(tool_use_blocks)} orphaned tool_use block(s) "
            "— injecting synthetic tool_results to repair history."
        )

        synthetic = [
            {
                "type": "tool_result",
                "tool_use_id": (b["id"] if isinstance(b, dict) else b.id),
                "content": (
                    "Tool call was interrupted by a rate-limit or network error. "
                    "Please retry."
                ),
            }
            for b in tool_use_blocks
        ]

        return list(messages) + [{"role": "user", "content": synthetic}]


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
