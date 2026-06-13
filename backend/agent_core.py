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


class AgentCore:
    def __init__(self, config: dict) -> None:
        self.config = config

        llm_cfg = config.get("llm", {})
        self.primary_model:     str  = llm_cfg.get("primary",     "claude-haiku-4-5")
        self.complex_model:     str  = llm_cfg.get("complex",     "claude-sonnet-4-6")
        self.local_model:       str  = llm_cfg.get("local",       "qwen2.5:14b")
        self.local_agent_model: str  = llm_cfg.get("local_agent", "qwen2.5:14b")

        self.use_prompt_optimizer:  bool = config.get("use_prompt_optimizer",  True)
        self.use_intent_routing:    bool = config.get("use_intent_routing",    True)
        self.use_tool_compression:  bool = config.get("use_tool_compression",  True)
        self.use_code_prevalidation: bool = config.get("use_code_prevalidation", True)
        self.use_tool_prefilter:    bool = config.get("use_tool_prefilter",    False)
        self.local_fallback:        bool = config.get("local_fallback",        True)
        self.local_mode:            bool = config.get("local_mode",            False)
        self.ollama_url:            str  = config.get("ollama_base_url",       "http://localhost:11434")

        # Per-request timeout for the local agentic loop (large models need time on CPU)
        self.local_agent_timeout: float = float(config.get("local_agent_timeout", 300))

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
            "code, and call execute_code again. Do not give up after one failure."
            "You can write and register new tools using write_tool and reload_tool. "
            "Tools must be written as Python async functions following this template: "
            "async def tool_name(param: str) -> dict — always return a dict with at least "
            "a 'success' key. Include a register_<toolname>_tools() function that calls "
            "register_tool() from agent_tools (import: from agent_tools import register_tool). "
            "After writing a tool with write_tool, always call reload_tool to activate it. "
            "New tools are saved to agent_tools/generated/ and persist across restarts. "
            "agent_core.py and main.py are read-only to you — only modify files in agent_tools/generated/."
            "\n\nIMPORTANT: After completing any research task (web search, page fetching, "
            "or information gathering), you MUST call log_research before sending your final "
            "answer. After learning any specific fact about the user, their system, or their "
            "preferences, you MUST call log_fact. Never end a research task without logging "
            "findings — this is how you build persistent knowledge across sessions."
            "\n\nFor high-level research goals (finding opportunities, comparing options, "
            "investigating topics), use the deep_research tool first to get a structured "
            "research plan. Then follow the plan: search each sub-question, fetch relevant "
            "pages, evaluate findings against the criteria, log each angle with log_research, "
            "and produce a final ranked report saved as a .md file. "
            "Always read the user profile before research tasks that depend on personal fit."
            "\n\nYou have browser tools for interacting with JavaScript-rendered web pages: "
            "browser_open(url) navigates to a page in a real Chromium browser, "
            "browser_read(selector) extracts visible text from it, and "
            "browser_screenshot(filename) saves a PNG of the current page to outputs/. "
            "Use browser tools when fetch_page returns empty or incomplete content — many "
            "modern sites require JavaScript execution and will return nothing useful to a "
            "plain HTTP fetch. Always call browser_open before browser_read. Use 'body' as "
            "the default selector for full page text, or more specific CSS selectors for "
            "targeted extraction (e.g. 'article', 'main', '#content')."
            "\n\nSave all generated files (reports, scripts, data, screenshots) to the "
            "outputs/ directory in the project root, not to backend/ or frontend/. "
            "Use write_file with a path like 'outputs/report.md' for text files. "
            "Use list_outputs() at the start of any task that might produce files, "
            "to check what already exists and avoid duplicating work. "
            "If a report was already generated in a previous session, read it with "
            "read_file before generating a new one."
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
            "works before you declare it done."
            "\n\nWhen editing an existing file, prefer patch_file over write_file — "
            "it modifies only the specified lines and preserves the rest. "
            "Use analyze_file first to see line counts, then read_file to find the exact "
            "lines to change, then patch_file to apply the edit. "
            "If a project requires packages not yet installed, call install_package for each "
            "dependency before running the project. Check scaffold dependencies first."
            "\n\nGITHUB WORKFLOW: You have GitHub tools for managing code repositories. "
            "After completing and testing a project successfully (run_project_test passes), "
            "offer to push it to GitHub. If the user agrees: "
            "(1) Call github_create_repo using the project name as the repo name "
            "and the scaffold description as the repo description. "
            "(2) Call github_push_file for each file in implementation_order from the scaffold, "
            "reading each file from outputs/{project_name}/ with read_file first. "
            "(3) After all files are pushed, update scaffold.json to set github_repo "
            "to the repo's full_name (e.g. 'username/project-name'). "
            "For research reports and other valuable outputs, you can also push them to a "
            "dedicated notes or research repo if the user has one. "
            "Before creating a repo, call github_list_repos to check if one already exists "
            "for this project — avoid creating duplicates."
            "\n\nCREDENTIAL MANAGER: Use store_credential(service, value) to save API keys "
            "securely — never ask the user to paste keys directly into the chat. "
            "Use get_credential(service) to retrieve a stored key when a tool needs it. "
            "Use list_credentials() to see what is already stored before asking the user "
            "to provide a key. "
            "When storing a credential, the value comes from the user typing it into the "
            "permission approval modal — you call store_credential with the service name "
            "and the user supplies the value at approval time. "
            "Example: to store a YouTube API key, call "
            "store_credential('youtube_api_key', value) where value is whatever the user "
            "provides. Service names must be alphanumeric with underscores/hyphens only "
            "(e.g. youtube_api_key, openai_key, slack_token)."
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
        system = self._base_system_prompt
        if context_summary:
            system = f"{self._base_system_prompt}\n\n{context_summary}"

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
        system = self._base_system_prompt
        if context_summary:
            system = f"{self._base_system_prompt}\n\n{context_summary}"
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
