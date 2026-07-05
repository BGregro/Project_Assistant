# Personal AI Agent – Project Instructions

## Project Goal
Build a modular, universal, locally-run personal AI agent using Claude API as the
LLM "brain", with an extensible tool system, Python backend, and web UI.
The long-term goal is an agent capable of truly autonomous complex tasks —
developing software, deep research, self-optimization, and eventually extending
its own capabilities without human intervention for each step.
The agent is designed to evolve toward persistent identity, autonomous goal-pursuit,
continuous self-assessment, and strategy learning from experience.

## Developer Context
- **Developer:** 3rd-year Computer Engineering student
- **OS:** Windows 11 (WSL2 available if needed)
- **Hardware:** Lenovo ThinkPad E14 Gen7 (Intel Core Ultra 5 Lunar Lake,
  Intel Arc 140V iGPU 8 Xe2 cores, 32GB LPDDR5x 137 GB/s, Intel NPU 40 TOPS)
- **LLM:** Claude API (Anthropic) — claude-haiku-4-5 default, claude-sonnet-4-6 complex
- **Local LLM:** ipex-llm Ollama on Arc 140V iGPU (SYCL/XPU backend)
  - `qwen3:8b` — preprocessing tier (intent routing, compression, optimization)
  - `qwen3:14b` — agent loop + background maintenance
  - Both models resident simultaneously (OLLAMA_MAX_LOADED_MODELS=2)
  - DEFAULT_MODEL = "qwen3:8b", DEFAULT_AGENT_MODEL = "qwen3:14b" in local_llm.py
- **Self-hosted search:** SearXNG on Docker at http://localhost:8888
- **Goal:** Learning + practical utility, incrementally extensible system

## Tech Stack
- **Backend:** Python (FastAPI)
- **Frontend:** HTML/CSS/JS chat UI
- **LLM Integration:** Anthropic Python SDK, tool use (function calling)
- **Local LLM:** ipex-llm Ollama (http://localhost:11434), iGPU accelerated
- **Vector Memory:** ChromaDB + nomic-embed-text (via Ollama)
- **Web Search:** SearXNG (primary) + DuckDuckGo HTML scraping (fallback)
- **Code Execution:** Subprocess sandbox with permission layer
- **Browser Automation:** Playwright (read + write, Chromium headless)
- **Package Manager:** pip + virtualenv

## LLM Three-Tier Architecture
```
User message
     │
     ▼
[Local qwen3:8b — FAST preprocessing on Arc iGPU]
     ├── Intent routing: SIMPLE / LOCAL_SUFFICIENT / TOOL / COMPLEX
     │     → SIMPLE: answer locally immediately, no API
     │     → LOCAL_SUFFICIENT: show tier banner (user chooses local/Claude)
     │     → TOOL: route to Claude Haiku
     │     → COMPLEX: route to Claude Sonnet
     ├── Prompt optimization: rewrite raw input into clean prompt
     ├── Tool result compression: strip verbose output before Claude sees it
     ├── History summarization: compress old turns to save context space
     ├── Tool pre-filtering: select 8-10 relevant tools (~50% token saving)
     ├── Vague message enrichment: expand short messages using conversation context
     └── Tiered system prompt: append only relevant sections per message type
          │
          ▼
[Local qwen3:14b — QUALITY local tasks on Arc iGPU]
     ├── Full local agent loop (LOCAL_SUFFICIENT path)
     ├── Code pre-validation: thorough bug detection before execute_code
     ├── Mid-task step summarization: compress completed steps during long runs
     ├── Email classification: batch header analysis
     ├── Deep research sub-question generation
     ├── Background reflection generation (fire-and-forget after tasks)
     └── Background memory maintenance (nightly consolidation job)
          │
          ▼
     [Claude API — with prompt caching]
     ├── claude-haiku-4-5: tool calls, structured tasks, most interactions
     ├── claude-sonnet-4-6: complex reasoning, planning, self-modification
     ├── Prompt caching: system prompt + conversation history cached at 10% cost
     └── Batch processing: nightly maintenance and analysis jobs at 50% cost
```

**Key efficiency principles:**
- Claude's context window contains only clean, compressed, relevant information
- Local tier preprocesses everything — it's a pipeline, not just a fallback
- Both local models stay resident in iGPU shared memory (no reload delays)
- Prompt caching: conversation history reads at 10% of normal input token cost
- Batch processing: maintenance jobs run at 50% cost asynchronously overnight
- Background maintenance runs free of charge on the local model during idle time

**API Efficiency Stack:**
- `use_prompt_optimizer`: local prompt rewriting before API call
- `use_intent_routing`: local classification routes to cheapest capable model
- `use_tool_compression`: local stripping of verbose tool output
- `use_code_prevalidation`: local bug detection saves failed execute_code calls
- `use_tool_prefilter`: local selection of 8-10 relevant tools per request
- Prompt caching: `cache_control` on system prompt + automatic message caching
- Batch processing: Anthropic Messages Batches API for async overnight jobs

## Current Tool Inventory (Phases 1–12d complete)
60+ tools registered at startup. All visible to Claude via the tool registry.

| Tool | File | Description |
|---|---|---|
| `read_file` | filesystem.py | Read a file from disk (cached by mtime) |
| `write_file` | filesystem.py | Write/append to a file |
| `list_directory` | filesystem.py | List directory contents + emit tree update |
| `list_outputs` | filesystem.py | List files in outputs/ directory |
| `patch_file` | filesystem.py | Apply targeted line-range edits to a file |
| `list_capabilities` | capabilities.py | Introspect live tool registry at call time |
| `search_web` | web.py | SearXNG search with DDG fallback |
| `fetch_page` | web.py | Fetch + strip HTML from any URL |
| `get_system_info` | system_info.py | CPU, RAM, disk, Ollama models, iGPU status |
| `analyze_file` | file_analysis.py | Size, lines, words, estimated token count |
| `execute_code` | code_executor.py | Run Python or Bash, streamed output |
| `install_package` | code_executor.py | pip install with confirmation |
| `write_tool` | tool_writer.py | Write + validate a new tool file |
| `reload_tool` | tool_writer.py | Hot-reload a generated tool into registry |
| `log_research` | memory_tool.py | Save research findings to long-term memory |
| `recall_memory` | memory_tool.py | Query past tasks, facts, and research |
| `log_fact` | memory_tool.py | Store a specific fact to long-term memory |
| `recall_projects` | memory_tool.py | Query past built projects |
| `correlate_memories` | memory_tool.py | Find connections between two concepts |
| `timeline_memory` | memory_tool.py | Chronological activity in a date range |
| `query_memory` | memory_tool.py | Unified search across all memory layers |
| `read_user_profile` | self_knowledge.py | Read memory/user_profile.json |
| `scan_system` | self_knowledge.py | Scan tools, packages, projects, iGPU status |
| `get_context_usage` | self_knowledge.py | Estimate current context window usage |
| `analyze_performance` | self_knowledge.py | AI-generated improvement suggestions |
| `get_performance_metrics` | self_knowledge.py | Per-tool call counts and success rates |
| `update_user_profile` | profile_updater.py | Update fields in user_profile.json |
| `deep_research` | research_mode.py | Structured research plan with cache check |
| `ask_user` | interaction.py | Pause mid-task and ask user a question |
| `browser_open` | browser.py | Navigate to URL in headless Chromium |
| `browser_read` | browser.py | Extract text from current browser page |
| `browser_screenshot` | browser.py | Save PNG of current browser page |
| `browser_click` | browser.py | Click an element (destructive) |
| `browser_fill` | browser.py | Fill an input field (destructive) |
| `browser_get_url` | browser.py | Get current page URL and title |
| `scaffold_project` | project_scaffold.py | Architecture + auto-generate control tools |
| `get_project_status` | project_manager.py | Check files done vs pending |
| `mark_file_complete` | project_manager.py | Mark a project file as implemented |
| `read_project_state` | project_manager.py | Read rich state snapshot for resumption |
| `run_project_test` | project_tester.py | Run project entry point and capture output |
| `github_list_repos` | github_tool.py | List GitHub repositories |
| `github_create_repo` | github_tool.py | Create a new GitHub repository |
| `github_push_file` | github_tool.py | Push a file to a GitHub repository |
| `github_read_file` | github_tool.py | Read a file from a GitHub repository |
| `github_list_files` | github_tool.py | List files in a GitHub repository |
| `github_create_issue` | github_tool.py | Create a GitHub issue |
| `store_credential` | credentials.py | Encrypt and store an API key or token |
| `get_credential` | credentials.py | Decrypt and retrieve a stored credential |
| `list_credentials` | credentials.py | List stored credential service names |
| `youtube_search` | youtube_tool.py | Search YouTube videos |
| `youtube_get_video_stats` | youtube_tool.py | Get views, likes, duration for a video |
| `youtube_get_trending` | youtube_tool.py | Get trending videos by region |
| `youtube_get_video_comments` | youtube_tool.py | Read top comments on a video |
| `youtube_get_channel_info` | youtube_tool.py | Get channel stats and info |
| `youtube_search_captions` | youtube_tool.py | List available caption tracks |
| `start_process` | process_manager.py | Launch a persistent background process |
| `stop_process` | process_manager.py | Terminate a named process |
| `read_process_output` | process_manager.py | Read stdout from a running process |
| `send_process_input` | process_manager.py | Send stdin to a running process |
| `list_processes` | process_manager.py | List all tracked processes and status |
| `schedule_task` | scheduler_tool.py | Schedule a recurring or one-time task |
| `list_scheduled_tasks` | scheduler_tool.py | List all scheduled tasks with next run |
| `cancel_scheduled_task` | scheduler_tool.py | Cancel a scheduled task |
| `convert_video` | media_tool.py | Convert video format via ffmpeg |
| `extract_audio` | media_tool.py | Extract audio from video via ffmpeg |
| `trim_clip` | media_tool.py | Trim video to time range via ffmpeg |
| `merge_clips` | media_tool.py | Merge multiple video clips via ffmpeg |
| `get_media_info` | media_tool.py | Get video/audio metadata via ffprobe |
| `send_email` | notification_tool.py | Send SMTP email notification |
| `test_email_config` | notification_tool.py | Verify SMTP connection without sending |
| `watch_directory` | file_watcher.py | Watch folder for file system events |
| `stop_watching` | file_watcher.py | Stop a directory watcher |
| `list_watches` | file_watcher.py | List active directory watchers |
| `email_connect` | email_tool.py | Connect to IMAP inbox |
| `email_scan_inbox` | email_tool.py | Scan inbox for old emails by category |
| `email_classify_and_plan` | email_tool.py | Local LLM classifies emails keep/delete |
| `email_delete_batch` | email_tool.py | Delete approved emails (dry_run default) |
| `email_disconnect` | email_tool.py | Disconnect from IMAP |
| `log_reflection` | episode_memory.py | Add/update reflection on a task episode |
| `get_episode` | episode_memory.py | Retrieve full task episode with reflection |
| `get_recent_episodes` | episode_memory.py | Get last N task episodes with reflections |
| `query_knowledge_graph` | knowledge_graph.py | Query concept connections in semantic graph |
| `add_graph_edge` | knowledge_graph.py | Add a relationship to the knowledge graph |

Generated tools in `agent_tools/generated/` auto-load on startup.
All generated files (reports, scripts, data) saved to `outputs/`.

## Memory Architecture
- **Short-term:** last N conversation turns (verbatim, history.json)
- **Semantic:** ChromaDB vector store with nomic-embed-text embeddings
- **Long-term:** `memory/long_term.json` — tasks (with reflection + failure_type),
  facts (with age + staleness flag), research (semantically indexed), projects
- **Research index:** ChromaDB collection for semantic research retrieval
- **Performance metrics:** `memory/performance_metrics.json` — per-tool stats
- **Knowledge graph:** `memory/semantic_graph.json` — concept/tool/project relationships
- **User profile:** `memory/user_profile.json` — skills, goals, hardware, preferences
- **Task checkpoint:** `memory/current_task.json` — last task state, survives reconnect
- **Project progress:** `outputs/{name}/progress.json` — per-project build state
- **Project state:** `outputs/{name}/state.json` — rich resumption snapshot
- **Scheduled tasks:** `memory/scheduled_tasks.json` — persisted across restarts
- **Credentials:** `memory/credentials.json` — Fernet encrypted, gitignored

## Development Phases

### Phases 1–11 ✓ COMPLETE
Phases 1–5: Base agent, extended tools, autonomous agent, software dev agent,
quality & reliability, external integrations.
Phase 6: Long task reliability (ask_user, project state snapshots).
Phase 7: UI overhaul (processes, scheduler, analytics, task history panels).
Phase 8: Self-improvement infrastructure (tiered prompt, file cache, auto-profile).
Phase 9: Media & notifications (ffmpeg, SMTP, file watcher, email tool, LOCAL_SUFFICIENT).
Phase 10: Remote access (token auth, 0.0.0.0, Tailscale/Cloudflare support).
Phase 11: iGPU acceleration (ipex-llm, qwen3:8b+14b resident, hardware self-awareness).

---

### Phase 11.5 — API Efficiency Improvements (CURRENT TARGET)

**11.5a — Prompt Caching (implement now)**
Enable Anthropic prompt caching on all Claude API calls.
The system prompt and tool definitions are stable across requests within a session.
Conversation history grows but previous turns are cacheable.
Cache reads cost 10% of normal input token price — 60-90% savings on long tasks.

Implementation in `backend/agent_core.py` `_run_claude_once()`:
- Add `cache_control={"type": "ephemeral"}` at the top level of the API call
  (automatic caching — caches everything up to the last message automatically)
- For the system prompt: mark it with an explicit `cache_control` breakpoint
  so the stable system content is cached separately from the growing messages
- Track cache hit stats: log `response.usage.cache_read_input_tokens` to
  `memory/performance_metrics.json` under a new `"api_cache"` section
- Note: when `use_tool_prefilter` is enabled, tool definitions change per call
  and the tool cache invalidates. Prompt caching still applies to system + messages.

**11.5b — Batch Processing Infrastructure**
New file `backend/batch_processor.py`.
Wraps the Anthropic Message Batches API for async overnight jobs.

```python
# Key functions:
submit_batch(requests: list[dict], job_name: str) -> str  # returns batch_id
poll_batch(batch_id: str) -> dict  # {status, succeeded, failed, total}
get_results(batch_id: str) -> list[dict]  # [{custom_id, result_type, content}]
cancel_batch(batch_id: str) -> None
list_pending_batches() -> list[dict]  # batches not yet retrieved
```

Batch jobs store their IDs in `memory/pending_batches.json`.
A scheduled APScheduler job polls pending batches every 30 minutes and
processes results when they arrive.

**11.5c — Historical Reflection Backfill**
Use batch processing to generate reflections for the 100 historical tasks
that were logged before Phase 12a. Submit all 100 as a batch at 50% cost,
receive results within 1 hour, write them to `long_term.json` via `log_reflection`.
Triggered by a new tool: `backfill_reflections()` — submits the batch and
schedules a job to process results when ready.

---

### Phase 12 — Enhanced Memory Architecture ✓ COMPLETE

**12a — Episode journal with reflection ✓**
Background qwen3:14b reflection after every task. `log_reflection`, `get_episode`,
`get_recent_episodes` tools. Reflection field in every task entry.
Note: `DEFAULT_MODEL = "qwen3:8b"` in local_llm.py fixed model churn bug.

**12b — Performance metrics database ✓**
`memory/performance_metrics.json`, per-tool call/success/duration tracking.
`get_performance_metrics` tool, updated `analyze_performance` reads from metrics.

**12c — Semantic knowledge graph ✓**
`memory/semantic_graph.json`, nodes and edges auto-built from task history.
`query_knowledge_graph`, `add_graph_edge` tools. Background updates after tasks.

**12d — Advanced memory query API ✓**
`correlate_memories`, `timeline_memory`, `query_memory` tools.
Datetime-aware queries with age labels, staleness flags on facts.
Configurable research cache TTL (`research_cache_days` in config.json).

---

### Phase 13 — Goal Tracking System (NEXT)
*Roadmap Point 2: Autonomous Goal-Pursuit*

**13a — Goal registry (Alpha)**
New file `memory/goals.json`.
Goal structure: {goal_id, title, description, status, created_date, target_date,
milestones[], current_strategy, blockers[], related_tasks[], priority}.
New tools: `create_goal`, `update_goal`, `list_goals`, `get_goal`.
UI: Goals tab in right panel with status badges.

**13b — Progress tracking and milestone management (Beta)**
Link tasks to goals in `log_task()` (optional goal_id param).
Milestone completion detection and notification.
New tools: `log_goal_progress`, `get_goal_progress`.

**13c — Autonomous planning per goal (Gamma)**
New tools: `decompose_goal` (local model breaks into sub-tasks),
`schedule_goal_work` (APScheduler integration), `detect_goal_blocker`.

**13d — Proactive goal reporting (Delta)**
Weekly scheduled job: goal status report saved to outputs/goal_reports/.
Send via email if enabled. New tool: `generate_goal_report()`.

---

### Phase 14 — Continuous Self-Reflection & Assessment
*Roadmap Point 3: Self-Assessment*
*(14a already partially built via Phase 12a background reflection)*

**14b — Failure classification system (Beta)**
Classify task failures: tool_integration_error, logic_error, knowledge_gap,
resource_constraint, user_communication, external_failure.
Store `failure_type` field (already in schema, not yet populated automatically).
New tool: `classify_failure(task_id)`.

**14c — Pattern detection and improvement proposals (Gamma)**
Background job every 10 tasks: scan failures for patterns.
Store proposals in `memory/improvement_proposals.json`.
New tools: `get_improvement_proposals`, `apply_improvement_proposal`.

---

### Phase 15 — Advanced Self-Modification
*Roadmap Point 4: Capability Expansion*

**15a — Capability gap detection (Alpha)**
Local model checks tool coverage before complex tasks.
New tool: `analyze_capability_gap(task_goal)`.

**15b — Tool design pipeline (Beta)**
New tools: `design_tool(gap_description)`, `implement_tool_from_design(spec)`.
Design-first workflow before write_tool.

**15c — Tool performance tracking (Gamma)**
Registry metadata: created_date, call_count, success_count, created_by.
New tools: `get_tool_metadata`, `prune_unused_tools`.

---

### Phase 16 — Background Memory Maintenance
*New concept: Memory Consolidation + Batch Processing*

**16a — Memory deduplication and cleanup (Alpha)**
Nightly APScheduler job (3:00 AM, configurable).
Uses qwen3:14b to find duplicate facts, stale entries, redundant research.
Produces cleanup report in outputs/maintenance/. Never auto-deletes.

**16b — Memory summarization with batch processing (Beta)**
For research entries older than 30 days with >500 word findings:
EITHER qwen3:14b generates compressed 100-word summary locally,
OR submit as batch to Claude (50% cost) for higher quality summaries.
Batch approach: submit nightly, results arrive next morning.

**16c — Pattern extraction and cross-linking (Gamma)**
Local model scans episodes for recurring patterns.
Updates knowledge graph with newly discovered edges.
Generates weekly "patterns learned" summary.
Feeds into Phase 17 strategy registry.

**16d — Memory health reporting (Delta)**
Monthly report: entry counts, growth rate, duplicate rate, knowledge gaps.
New tools: `run_memory_maintenance()`, `get_maintenance_report()`.
Batch processing: submit large analysis jobs overnight at 50% cost.

---

### Phase 17 — Strategy Evolution Loop
*Roadmap Point 5: Continuous Learning*
*Depends on Phases 12, 13, 14, 15, 16*

**17a — Strategy extraction via batch processing (Alpha)**
New file `memory/strategies.json`.
Strategy structure: {strategy_id, problem_type, steps[], evidence, success_rate}.
Batch submit groups of similar tasks to Claude overnight for pattern extraction.
Results processed next morning, strategies added to registry.
Minimum 3 successful uses before a strategy is created.

**17b — Contextual strategy application (Beta)**
New tools: `recall_relevant_strategies`, `adapt_strategy`,
`track_strategy_effectiveness`.
System prompt: check strategies before planning complex tasks.

**17c — Domain proficiency tracking (Gamma)**
Track success rate per domain (research, coding, email, video, etc.).
New tools: `get_domain_proficiency`, `identify_weak_domains`.

**17d — Knowledge synthesis reports via batch processing (Delta)**
Monthly synthesis report: what was learned, open questions, tool impact.
Submitted as batch to Claude Sonnet overnight (50% cost).
Sent via email, saved to outputs/synthesis/.
New tool: `generate_synthesis_report(period)`.

---

## Project Structure
```
agent/
├── backend/
│   ├── main.py
│   ├── agent_core.py            # prompt caching in _run_claude_once()
│   ├── task_runner.py
│   ├── task_scheduler.py
│   ├── batch_processor.py       # Phase 11.5b — Anthropic Batches API wrapper
│   ├── run.py                   # Windows launcher + ipex-llm auto-start
│   ├── agent_tools/
│   │   ├── __init__.py
│   │   ├── filesystem.py
│   │   ├── capabilities.py
│   │   ├── web.py
│   │   ├── system_info.py
│   │   ├── file_analysis.py
│   │   ├── local_llm.py         # DEFAULT_MODEL="qwen3:8b", DEFAULT_AGENT_MODEL="qwen3:14b"
│   │   ├── code_executor.py
│   │   ├── tool_writer.py
│   │   ├── hot_reload.py
│   │   ├── memory_tool.py       # + correlate_memories, timeline_memory, query_memory
│   │   ├── self_knowledge.py    # + get_performance_metrics
│   │   ├── profile_updater.py
│   │   ├── research_mode.py
│   │   ├── interaction.py
│   │   ├── browser.py
│   │   ├── project_scaffold.py
│   │   ├── project_manager.py
│   │   ├── project_tester.py
│   │   ├── github_tool.py
│   │   ├── credentials.py
│   │   ├── youtube_tool.py
│   │   ├── process_manager.py
│   │   ├── scheduler_tool.py
│   │   ├── media_tool.py
│   │   ├── notification_tool.py
│   │   ├── file_watcher.py
│   │   ├── email_tool.py
│   │   ├── episode_memory.py
│   │   ├── knowledge_graph.py
│   │   ├── # Phase 13+:
│   │   ├── goal_tracker.py
│   │   ├── # Phase 14+:
│   │   ├── reflection_engine.py
│   │   ├── # Phase 15+:
│   │   ├── capability_tools.py
│   │   ├── # Phase 16+:
│   │   ├── memory_maintenance.py
│   │   ├── # Phase 17+:
│   │   ├── strategy_tools.py
│   │   └── generated/
│   └── memory/
│       ├── context.py
│       ├── embeddings.py
│       ├── long_term.py
│       └── performance.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── memory/
│   ├── history.json
│   ├── current_task.json
│   ├── long_term.json
│   ├── user_profile.json
│   ├── credentials.json         # gitignored
│   ├── scheduled_tasks.json
│   ├── performance_metrics.json
│   ├── semantic_graph.json
│   ├── pending_batches.json     # Phase 11.5b
│   ├── improvement_proposals.json  # Phase 14c
│   ├── goals.json               # Phase 13
│   ├── strategies.json          # Phase 17
│   └── vectors/
├── outputs/
│   ├── {project_name}/
│   ├── goal_reports/
│   ├── maintenance/
│   └── synthesis/
├── config.json
├── .env
├── .gitignore
├── requirements.txt
├── SEARXNG_SETUP.md
└── PLAYWRIGHT_SETUP.md
```

## Coding Principles
1. **Always modular** — each tool in its own file; no circular dependencies
2. **Token-efficient** — local preprocessing + prompt caching + batch processing
3. **Incremental complexity** — never skip a phase; stabilize before extending
4. **Security first** — permission checks; self-modification restricted to generated/
5. **Commented code** — developer is learning; explain non-trivial parts
6. **Windows compatible** — always use run.py (ProactorEventLoop)
7. **Designed for long tasks** — context management, checkpointing, compression
8. **Local model first** — qwen3:8b/14b on iGPU; Claude API when quality matters
9. **Background first** — never block user on maintenance/reflection/analysis

## How to Help
- **Always provide every created or edited file as a downloadable file**
- Full file content, not diffs, unless partial is explicitly requested
- 2-3 options with pro/con for architectural decisions
- Explicitly flag security risks
- After files: "What changed / What to do next" summary
- After files: "Test it" section with 3-5 concrete test prompts

## Terminology
- **Preprocessing tier:** qwen3:8b on iGPU — fast local preprocessing pipeline
- **Agent tier:** qwen3:14b on iGPU — quality local reasoning and agent loop
- **Intent router:** SIMPLE/LOCAL_SUFFICIENT/TOOL/COMPLEX routing
- **Prompt caching:** Claude API feature, 10% cost on cached content reads
- **Batch processing:** Anthropic Batches API, 50% cost, async overnight jobs
- **Task runner:** long-running autonomous execution loop
- **Hot-reload:** re-registering tools at runtime without server restart
- **Memory consolidation:** nightly background maintenance on local model
- **Episode:** completed task with goal, tools, outcome, reflection, failure_type
- **Knowledge graph:** semantic graph of concepts, tools, projects, relationships
- **Strategy registry:** proven approaches extracted from past successful tasks
- **Domain proficiency:** agent's measured success rate per task category
