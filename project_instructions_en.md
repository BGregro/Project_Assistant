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
     └── Background memory maintenance (nightly consolidation job)
          │
          ▼
     [Claude API]
     ├── claude-haiku-4-5: tool calls, structured tasks, most interactions
     └── claude-sonnet-4-6: complex reasoning, planning, self-modification,
                             research synthesis, multi-file app development
```

**Key efficiency principles:**
- Claude's context window contains only clean, compressed, relevant information
- Local tier preprocesses everything — it's a pipeline, not just a fallback
- Both local models stay resident in iGPU shared memory (no reload delays)
- Background maintenance runs free of charge on the local model during idle time

**Config controls (all runtime-adjustable via settings panel):**
- `use_prompt_optimizer`, `use_intent_routing`, `use_tool_compression`
- `use_code_prevalidation`, `use_tool_prefilter`
- `local_fallback`, `local_mode`, `local_sufficient_default` (ask/local/claude)
- `primary` (claude-haiku-4-5), `complex` (claude-sonnet-4-6)
- `local` (qwen3:8b), `local_agent` (qwen3:14b)
- `local_agent_timeout` (default 180s with iGPU), `auto_approve_code_execution`
- `igpu.enabled`, `igpu.max_loaded_models` (2)

## Current Tool Inventory (Phases 1–11 complete)
55+ tools registered at startup. All visible to Claude via the tool registry.

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
| `read_user_profile` | self_knowledge.py | Read memory/user_profile.json |
| `scan_system` | self_knowledge.py | Scan tools, packages, projects, iGPU status |
| `get_context_usage` | self_knowledge.py | Estimate current context window usage |
| `analyze_performance` | self_knowledge.py | AI-generated improvement suggestions |
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

Generated tools in `agent_tools/generated/` auto-load on startup.
All generated files (reports, scripts, data) saved to `outputs/`.

## Memory Architecture
- **Short-term:** last N conversation turns (verbatim, history.json)
- **Semantic:** ChromaDB vector store with nomic-embed-text embeddings
- **Long-term:** `memory/long_term.json` — tasks (with reflection), facts,
  research (semantically indexed), projects
- **Research index:** ChromaDB collection for semantic research retrieval
- **User profile:** `memory/user_profile.json` — skills, goals, hardware, preferences
- **Task checkpoint:** `memory/current_task.json` — last task state, survives reconnect
- **Project progress:** `outputs/{name}/progress.json` — per-project build state
- **Project state:** `outputs/{name}/state.json` — rich resumption snapshot
- **Scheduled tasks:** `memory/scheduled_tasks.json` — persisted across restarts
- **Credentials:** `memory/credentials.json` — Fernet encrypted, gitignored

## Development Phases

### Phases 1–11 ✓ COMPLETE

**Phases 1–5:** Base agent, extended tools, autonomous agent, software dev agent,
quality & reliability, external integrations (GitHub, credentials, YouTube, browser,
scheduler).

**Phase 6 — Long Task Reliability ✓**
- ask_user: mid-task question pause/resume
- Project state snapshots for reliable resumption

**Phase 7 — UI Overhaul ✓**
- Processes panel, scheduler view, credential manager UI
- Agent analytics dashboard, live process output streaming
- Collapsible task containers, status bar, task history

**Phase 8 — Self-Improvement Infrastructure ✓**
- get_context_usage, analyze_performance tools
- Auto-profile updating from task history (background)
- Tiered system prompt (only relevant sections per message)
- File content cache (mtime-based, 20 entries)

**Phase 9 — Media & Notifications ✓**
- ffmpeg wrapper tools (convert, extract, trim, merge, info)
- SMTP email notifications with task completion trigger
- File watcher with pattern-based action triggers
- Email inbox scanner + local LLM classifier (batch_size=20)
- LOCAL_SUFFICIENT tier: user chooses local vs Claude per task

**Phase 10 — Remote Access ✓**
- Token-based auth middleware
- 0.0.0.0 binding, Tailscale/Cloudflare Tunnel support
- Login page, mobile-accessible UI

**Phase 11 — iGPU Acceleration ✓**
- ipex-llm Ollama on Intel Arc 140V (SYCL/XPU backend)
- Auto-start via run.py with .env management
- Two-model strategy: qwen3:8b (preprocessing) + qwen3:14b (agent)
- Both models resident simultaneously in shared iGPU memory
- Hardware self-awareness in scan_system and user_profile

---

### Phase 12 — Enhanced Memory Architecture (CURRENT TARGET)
*Roadmap Point 1: Expand Persistent Memory*

**12a — Episode journal with reflection (Alpha)**
Add a `reflection` field to every task entry in long_term.json.
After task completion, qwen3:14b generates a 2-3 sentence reflection:
what worked, what didn't, what would be done differently.
New tool: `log_reflection(task_id, reflection_text)`.
New tool: `get_episode(task_id)` — retrieve full episode including reflection.

**12b — Performance metrics database (Beta)**
New file `memory/performance_metrics.json`.
Track per-tool: call count, success count, average duration, last_failure_reason.
Track per-task-type (classified by the local model): success rate, avg duration.
New tool: `get_performance_metrics(tool_name_or_task_type)`.
Updated `analyze_performance()` reads from metrics rather than re-computing.

**12c — Semantic knowledge graph (Gamma)**
New file `memory/semantic_graph.json`.
Nodes: concepts, projects, tools, skills, people, external services.
Edges: {from, to, relationship, strength, last_seen}.
Built automatically: when a task uses tools X and Y together → add edge.
When research finds concept A related to B → add edge.
New tool: `query_knowledge_graph(concept, depth=2)` — return connected nodes.
New tool: `add_graph_edge(from_node, to_node, relationship)`.

**12d — Advanced memory query API (Delta)**
New tool: `correlate_memories(concept1, concept2)` — find tasks/facts/research
that mention both concepts, return connection analysis via local model.
New tool: `timeline_memory(start_date, end_date)` — chronological review
of all agent activity in a date range.
New tool: `query_memory(query, memory_types, filters)` — unified search
across all memory layers with type filtering.

---

### Phase 13 — Goal Tracking System
*Roadmap Point 2: Autonomous Goal-Pursuit*

**13a — Goal registry (Alpha)**
New file `memory/goals.json`.
Goal data structure: {goal_id, title, description, status, created_date,
target_date, milestones[], current_strategy, blockers[], related_tasks[],
related_projects[], priority}.
New tools: `create_goal`, `update_goal`, `list_goals`, `get_goal`.
UI: Goals tab in right panel showing active goals with status badges.

**13b — Progress tracking and milestone management (Beta)**
Link tasks to goals: when logging a task, optionally specify goal_id.
Milestone completion detection: when all sub-tasks for a milestone are done,
auto-mark milestone complete and notify user.
New tool: `log_goal_progress(goal_id, progress_note, milestone_completed)`.
New tool: `get_goal_progress(goal_id)` — full history of progress entries.

**13c — Autonomous planning per goal (Gamma)**
New tool: `decompose_goal(goal_id)` — local model breaks goal into sub-tasks
with estimated durations. Returns task list for user approval.
New tool: `schedule_goal_work(goal_id, hours_per_week)` — automatically
schedule decomposed sub-tasks using the existing APScheduler system.
New tool: `detect_goal_blocker(goal_id)` — analyze recent progress entries,
flag if no progress in >7 days, suggest pivot strategies.

**13d — Proactive goal reporting (Delta)**
Scheduled weekly job: generate goal status report for all active goals.
Report includes: on-track goals, at-risk goals, upcoming milestones, suggested
next actions for highest-priority goal.
Send via email if enabled, always save to outputs/goal_reports/.
New tool: `generate_goal_report()` — trigger on-demand version.

---

### Phase 14 — Continuous Self-Reflection & Assessment
*Roadmap Point 3: Self-Assessment*

**14a — Automatic post-task reflection (Alpha)**
In task_runner.py, after every completed task, fire a background job:
call qwen3:14b with the task goal, tools used, outcome, and duration.
Generate structured reflection: what worked, what failed, what assumptions
were made, what would be done differently.
Store reflection in long_term.json task entry (Phase 12a).
Never blocks task completion — pure background fire-and-forget.

**14b — Failure classification system (Beta)**
When a task fails, automatically classify the failure type using local model:
- `tool_integration_error`: tool call failed or returned wrong format
- `logic_error`: agent reasoned incorrectly about the task
- `knowledge_gap`: agent lacked needed information
- `resource_constraint`: rate limit, context overflow, timeout
- `user_communication`: misunderstood the goal
- `external_failure`: API down, file missing, network error
Store classification in failed_at_tool + new `failure_type` field.
New tool: `classify_failure(task_id)` — run classification on demand.

**14c — Pattern detection and improvement proposals (Gamma)**
Background job runs every 10 completed tasks:
scan failure logs for repeated failure types on same tools or task types.
If tool X has >30% failure rate: generate specific improvement proposal
(add retry logic, write wrapper tool, switch approach).
Store proposals in `memory/improvement_proposals.json`.
New tool: `get_improvement_proposals()` — retrieve pending proposals.
New tool: `apply_improvement_proposal(proposal_id)` — agent acts on proposal.

---

### Phase 15 — Advanced Self-Modification
*Roadmap Point 4: Capability Expansion*

**15a — Capability gap detection (Alpha)**
During task planning (before first tool call), local model checks:
"Do existing tools cover all steps needed for this goal?"
If gaps identified: emit a status event "Capability gap detected: [description]"
and optionally propose writing a new tool.
New system prompt instruction: "Before starting a complex task, check if any
steps require capabilities not in the tool registry. If so, consider writing
a new tool first."
New tool: `analyze_capability_gap(task_goal)` — explicit gap analysis.

**15b — Tool design and implementation pipeline (Beta)**
New tool: `design_tool(gap_description)` — local model generates tool spec:
function name, parameters, return type, core logic, test cases, dependencies.
Returns design document for user approval before implementation.
Extends existing write_tool + reload_tool pipeline with design-first step.
New tool: `implement_tool_from_design(design_spec)` — writes, validates,
and reloads tool in one step.

**15c — Tool performance tracking and registry metadata (Gamma)**
Extend tool registry to store per-tool metadata:
{created_date, purpose, call_count, success_count, last_used, created_by}.
`created_by`: "system" for built-in tools, "agent" for generated tools.
New tool: `get_tool_metadata(tool_name)`.
New tool: `prune_unused_tools()` — list generated tools unused for >30 days,
propose removal with user confirmation.
Background job: weekly tool health report summarizing tool usage patterns.

---

### Phase 16 — Background Memory Maintenance
*New concept: Memory Consolidation*

The idea: a nightly background process using qwen3:14b that maintains memory
health without any Claude API cost. Similar to how human brains consolidate
memories during sleep.

**16a — Memory deduplication and cleanup (Alpha)**
Nightly APScheduler job at 3:00 AM (configurable).
Uses qwen3:14b to:
- Find duplicate or near-identical facts in long_term.json
- Identify stale facts (>90 days old, no longer referenced)
- Merge redundant research entries on the same topic
- Flag facts that may be outdated (e.g., "current Python version is X")
Produces a cleanup report saved to outputs/maintenance/.
Never deletes automatically — flags for review, user approves bulk cleanup.

**16b — Memory summarization and compression (Beta)**
For research entries older than 30 days with >500 word findings:
local model generates a compressed 100-word summary.
Store both full and compressed version — use compressed for context injection,
full for when agent explicitly requests the entry.
For task entries older than 7 days: compress tool_results in episode log.
This keeps context injection fast even with large memory stores.

**16c — Pattern extraction and cross-linking (Gamma)**
After deduplication pass, local model scans recent episodes and:
- Identifies recurring patterns ("agent always searches before coding")
- Extracts implicit strategies from successful task sequences
- Updates semantic knowledge graph with new edges discovered
- Generates weekly "patterns learned" summary
Feeds into Phase 17 strategy registry.

**16d — Memory health reporting (Delta)**
Monthly report on memory health:
- Total entries per type, growth rate, oldest entries
- Duplicate rate (before vs. after consolidation)
- Most-referenced facts and research topics
- Knowledge gaps identified ("agent has never successfully done X")
Sent via email if enabled, always saved to outputs/maintenance/.
New tool: `run_memory_maintenance()` — trigger maintenance cycle on demand.
New tool: `get_maintenance_report()` — read latest maintenance report.

---

### Phase 17 — Strategy Evolution Loop
*Roadmap Point 5: Continuous Learning*
*Depends on Phases 12, 13, 14, 15, 16*

**17a — Experience aggregation and strategy extraction (Alpha)**
After Phase 16c has run at least 4 weeks, begin building strategy registry.
New file `memory/strategies.json`.
Strategy structure: {strategy_id, problem_type, description, steps[],
preconditions, expected_success_rate, evidence{successes, failures, notes},
derived_from[task_ids], last_used, created_date}.
New tool: `extract_strategy(task_ids)` — from a set of similar successful
tasks, local model generalizes a reusable strategy.
Minimum evidence threshold: 3+ successful uses before a strategy is created.

**17b — Contextual strategy application (Beta)**
New tool: `recall_relevant_strategies(task_goal)` — before starting any task,
search strategy registry for applicable past approaches.
New tool: `adapt_strategy(strategy_id, new_context)` — modify past strategy
to fit current situation.
New tool: `track_strategy_effectiveness(strategy_id, outcome)` — update
success rate after applying a strategy.
System prompt addition: "Before planning a complex task, call
recall_relevant_strategies to check if a proven approach already exists."

**17c — Domain proficiency tracking (Gamma)**
Track agent success rate per domain: web research, coding, email management,
video processing, data analysis, system administration, etc.
Domain classification done by local model per task.
New tool: `get_domain_proficiency()` — return proficiency levels with trends.
New tool: `identify_weak_domains()` — suggest which domains to practice.
Weak domains flagged in weekly goal report (Phase 13d).

**17d — Knowledge synthesis and evolution reports (Delta)**
Monthly synthesis report generated by qwen3:14b:
- "What I learned this month" (new strategies, improved domains)
- "Open questions I still can't reliably answer"
- "Tools I've created and their impact"
- "Goals progressed and milestones hit"
Sent via email, saved to outputs/synthesis/.
New tool: `generate_synthesis_report(period)` — trigger on demand.

---

## Project Structure
```
agent/
├── backend/
│   ├── main.py                  # FastAPI app, WebSocket, settings, broadcast
│   ├── agent_core.py            # LLM loop, tool dispatch, routing, tiered prompt
│   ├── task_runner.py           # Long-running task loop, sanitizer, compression
│   ├── task_scheduler.py        # APScheduler wrapper, persistent schedules
│   ├── run.py                   # Windows launcher + ipex-llm auto-start
│   ├── agent_tools/
│   │   ├── __init__.py          # Tool registry with get_all_definitions()
│   │   ├── filesystem.py        # File tools + cache + patch_file
│   │   ├── capabilities.py      # list_capabilities
│   │   ├── web.py               # search_web, fetch_page
│   │   ├── system_info.py       # get_system_info (iGPU aware)
│   │   ├── file_analysis.py     # analyze_file
│   │   ├── local_llm.py         # Ollama client, all local LLM tasks
│   │   ├── code_executor.py     # execute_code, install_package, streaming
│   │   ├── tool_writer.py       # write_tool + design_tool, reload_tool
│   │   ├── hot_reload.py        # validation + importlib hot-reload
│   │   ├── memory_tool.py       # log_research, recall_memory, log_fact,
│   │   │                        # recall_projects, query_memory, correlate_memories
│   │   ├── self_knowledge.py    # read_user_profile, scan_system,
│   │   │                        # get_context_usage, analyze_performance
│   │   ├── profile_updater.py   # update_user_profile
│   │   ├── research_mode.py     # deep_research (with cache check)
│   │   ├── interaction.py       # ask_user (mid-task pause/resume)
│   │   ├── browser.py           # browser_open/read/screenshot/click/fill/url
│   │   ├── project_scaffold.py  # scaffold_project + control tool generation
│   │   ├── project_manager.py   # get_project_status, mark_file_complete,
│   │   │                        # read_project_state
│   │   ├── project_tester.py    # run_project_test (with project memory logging)
│   │   ├── github_tool.py       # github_* tools
│   │   ├── credentials.py       # store/get/list_credentials (Fernet)
│   │   ├── youtube_tool.py      # youtube_* tools (read-only, public data)
│   │   ├── process_manager.py   # start/stop/read/send/list processes
│   │   ├── scheduler_tool.py    # schedule/list/cancel scheduled tasks
│   │   ├── media_tool.py        # convert/extract/trim/merge/info via ffmpeg
│   │   ├── notification_tool.py # send_email, test_email_config
│   │   ├── file_watcher.py      # watch_directory, stop_watching, list_watches
│   │   ├── email_tool.py        # email_connect/scan/classify/delete/disconnect
│   │   ├── # Phase 12+ (to be added):
│   │   ├── episode_memory.py    # log_reflection, get_episode, correlate_memories
│   │   ├── knowledge_graph.py   # query_knowledge_graph, add_graph_edge
│   │   ├── goal_tracker.py      # create/update/list/get_goal, log_goal_progress
│   │   ├── reflection_engine.py # reflect_on_task, classify_failure, proposals
│   │   ├── capability_tools.py  # analyze_capability_gap, design_tool
│   │   ├── memory_maintenance.py# run_memory_maintenance, get_maintenance_report
│   │   ├── strategy_tools.py    # extract/recall/adapt/track strategies
│   │   ├── generated/           # agent-written + scaffold control tools
│   │   └── SEARXNG_SETUP.md
│   └── memory/
│       ├── context.py           # History load/save/trim
│       ├── embeddings.py        # ChromaDB vector store
│       └── long_term.py         # Tasks, facts, research, projects + semantic index
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── memory/
│   ├── history.json
│   ├── current_task.json
│   ├── long_term.json           # tasks (with reflections), facts, research, projects
│   ├── user_profile.json        # skills, goals, hardware, preferences
│   ├── credentials.json         # encrypted, gitignored
│   ├── scheduled_tasks.json
│   ├── vectors/                 # ChromaDB persistent storage
│   ├── # Phase 12+ (to be added):
│   ├── performance_metrics.json # per-tool and per-task-type statistics
│   ├── semantic_graph.json      # knowledge graph nodes and edges
│   ├── improvement_proposals.json
│   ├── # Phase 13+:
│   ├── goals.json               # goal registry with milestones and blockers
│   ├── # Phase 17+:
│   └── strategies.json          # proven strategies with evidence tracking
├── outputs/
│   ├── {project_name}/
│   │   ├── scaffold.json
│   │   ├── progress.json
│   │   └── state.json
│   ├── goal_reports/            # Phase 13d weekly goal status reports
│   ├── maintenance/             # Phase 16 memory maintenance reports
│   └── synthesis/               # Phase 17d monthly synthesis reports
├── config.json                  # All settings, runtime-editable via UI
├── .env                         # ANTHROPIC_API_KEY, gitignored
├── .gitignore
├── requirements.txt
├── SEARXNG_SETUP.md
└── PLAYWRIGHT_SETUP.md
```

## Coding Principles
1. **Always modular** — each tool in its own file; core must not depend on tool
   implementations; new capabilities slot in without touching existing code
2. **Token-efficient** — local LLM preprocesses everything before Claude sees it;
   Claude's context should be clean, dense, and relevant
3. **Incremental complexity** — never skip a phase; stabilize before extending
4. **Security first** — permission check before every file/code/network operation;
   destructive actions always require user confirmation;
   agent self-modification restricted to agent_tools/generated/ only;
   agent_core.py, main.py, and task_runner.py are read-only to the agent
5. **Commented code** — developer is learning; explain non-trivial parts
6. **Windows compatible** — paths, subprocess calls must work on Windows;
   always use run.py (ProactorEventLoop) not uvicorn directly
7. **Designed for long tasks** — context management, checkpointing, compression
   must support tasks running for minutes or hours with many tool calls
8. **Local model first** — use qwen3:8b/14b on iGPU for any task where quality
   is sufficient; Claude API only when reasoning quality matters

## How to Help
- **Always provide every created or edited file as a downloadable file** —
  never just show code in a code block when a file is being created or modified
- When writing code, always provide the full file content (not just diffs),
  unless a partial snippet is explicitly requested
- For architectural decisions, provide 2-3 options with a short pro/con list
- When debugging, ask for the full error message and relevant code
- When developing a new tool, always show how it fits into the existing system
- Explicitly flag any security risks
- After delivering files: short "What changed / What to do next" summary
- After delivering files: "Test it" section with 3-5 concrete test prompts

## Terminology
- **Agent core:** the LLM + tool dispatch logic
- **Tool:** a concrete function the agent can call
- **Tool registry:** the live registry of all registered tools
- **Permission layer:** the user approval system for destructive actions
- **Context window:** the active LLM context, managed deliberately
- **Preprocessing tier:** qwen3:8b on iGPU — fast local preprocessing pipeline
- **Agent tier:** qwen3:14b on iGPU — quality local reasoning and agent loop
- **Intent router:** local classifier routing SIMPLE/LOCAL_SUFFICIENT/TOOL/COMPLEX
- **LOCAL_SUFFICIENT:** tasks local model can handle — user chooses local vs Claude
- **Prompt optimizer:** local rewrite of raw user input before Claude API call
- **Tool result compression:** local stripping of verbose tool output
- **Tool pre-filter:** local selection of 8-10 relevant tools per request
- **Tiered system prompt:** base prompt + contextual sections per message type
- **Task runner:** the long-running autonomous execution loop
- **Hot-reload:** re-registering tools at runtime without server restart
- **Generated tools:** agent-written + scaffold control tools in agent_tools/generated/
- **Project scaffold:** upfront architecture plan before multi-file implementation
- **Control tool:** auto-generated tool for starting/stopping a built project
- **State snapshot:** rich project state file updated at every step
- **Memory consolidation:** nightly background maintenance on local model
- **Episode:** a completed task with goal, tools, outcome, and reflection
- **Knowledge graph:** semantic graph of concepts, projects, tools, and their relations
- **Strategy registry:** proven approaches extracted from past successful tasks
- **Domain proficiency:** agent's measured success rate per task category
