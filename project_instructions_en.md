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
  - `qwen3:14b` — agent loop + background maintenance + reflections
  - Both models resident simultaneously (OLLAMA_MAX_LOADED_MODELS=2)
  - DEFAULT_MODEL = "qwen3:8b", DEFAULT_AGENT_MODEL = "qwen3:14b" in local_llm.py
- **Self-hosted search:** SearXNG on Docker at http://localhost:8888

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
     ├── Prompt optimization, tool result compression, history summarization
     ├── Tool pre-filtering: select 8-10 relevant tools (~50% token saving)
     ├── Vague message enrichment, tiered system prompt selection
     └── Offline fallback full local agent loop
          │
          ▼
[Local qwen3:14b — QUALITY local tasks on Arc iGPU]
     ├── Full local agent loop (LOCAL_SUFFICIENT path)
     ├── Code pre-validation (thorough, thinking mode enabled)
     ├── Background reflection generation (fire-and-forget, num_predict=512)
     ├── Email classification, deep research sub-questions
     └── Background memory maintenance (nightly consolidation)
          │
          ▼
     [Claude API — with prompt caching]
     ├── claude-haiku-4-5: tool calls, structured tasks, most interactions
     ├── claude-sonnet-4-6: complex reasoning, planning, self-modification
     ├── Prompt caching: system block cached at 10% cost on repeated reads
     └── Batch processing: nightly maintenance and analysis at 50% cost
```

**API Efficiency Stack:**
- Intent routing, tool pre-filter, compression — local, zero API cost
- Prompt caching: `cache_control` inside system_block in `_run_claude_once()`
- Batch processing: Anthropic Messages Batches API, async overnight jobs
- Task count cap: `_MAX_TASKS = 500`, `_MAX_RESEARCH = 1000`, `_MAX_FACTS = 500`

## Current Tool Inventory (Phases 1–12d complete)
65+ tools registered at startup.

| Tool | File | Description |
|---|---|---|
| `read_file` | filesystem.py | Read a file from disk (mtime cache) |
| `write_file` | filesystem.py | Write/append to a file |
| `list_directory` | filesystem.py | List directory contents + tree update |
| `list_outputs` | filesystem.py | List files in outputs/ |
| `patch_file` | filesystem.py | Targeted line-range edits |
| `list_capabilities` | capabilities.py | Introspect live tool registry |
| `search_web` | web.py | SearXNG + DDG fallback |
| `fetch_page` | web.py | Fetch + strip HTML from URL |
| `get_system_info` | system_info.py | CPU, RAM, disk, iGPU, Ollama |
| `analyze_file` | file_analysis.py | Size, lines, words, token estimate |
| `execute_code` | code_executor.py | Run Python/Bash, streamed output |
| `install_package` | code_executor.py | pip install with confirmation |
| `write_tool` | tool_writer.py | Write + validate new tool file |
| `reload_tool` | tool_writer.py | Hot-reload generated tool |
| `log_research` | memory_tool.py | Save research to long-term memory |
| `recall_memory` | memory_tool.py | Query tasks, facts, research |
| `log_fact` | memory_tool.py | Store a fact |
| `recall_projects` | memory_tool.py | Query past projects |
| `correlate_memories` | memory_tool.py | Find connections between two concepts |
| `timeline_memory` | memory_tool.py | Chronological activity in date range |
| `query_memory` | memory_tool.py | Unified search across all memory layers |
| `read_user_profile` | self_knowledge.py | Read user_profile.json |
| `scan_system` | self_knowledge.py | Scan tools, packages, iGPU status |
| `get_context_usage` | self_knowledge.py | Estimate context window usage |
| `analyze_performance` | self_knowledge.py | AI improvement suggestions |
| `get_performance_metrics` | self_knowledge.py | Per-tool stats from metrics DB |
| `update_user_profile` | profile_updater.py | Update user_profile.json |
| `deep_research` | research_mode.py | Research plan + cache check |
| `ask_user` | interaction.py | Pause mid-task for user input |
| `browser_open` | browser.py | Navigate in headless Chromium |
| `browser_read` | browser.py | Extract text from current page |
| `browser_screenshot` | browser.py | Save PNG of current page |
| `browser_click` | browser.py | Click element (destructive) |
| `browser_fill` | browser.py | Fill input field (destructive) |
| `browser_get_url` | browser.py | Get current URL and title |
| `scaffold_project` | project_scaffold.py | Architecture + control tool gen |
| `get_project_status` | project_manager.py | Files done vs pending |
| `mark_file_complete` | project_manager.py | Mark file as implemented |
| `read_project_state` | project_manager.py | Rich state snapshot for resumption |
| `run_project_test` | project_tester.py | Run project entry point |
| `github_list_repos` | github_tool.py | List GitHub repos |
| `github_create_repo` | github_tool.py | Create GitHub repo |
| `github_push_file` | github_tool.py | Push file to GitHub |
| `github_read_file` | github_tool.py | Read file from GitHub |
| `github_list_files` | github_tool.py | List files in GitHub repo |
| `github_create_issue` | github_tool.py | Create GitHub issue |
| `store_credential` | credentials.py | Encrypt + store API key |
| `get_credential` | credentials.py | Decrypt + retrieve credential |
| `list_credentials` | credentials.py | List credential service names |
| `youtube_search` | youtube_tool.py | Search YouTube videos |
| `youtube_get_video_stats` | youtube_tool.py | Views, likes, duration |
| `youtube_get_trending` | youtube_tool.py | Trending videos by region |
| `youtube_get_video_comments` | youtube_tool.py | Top comments on video |
| `youtube_get_channel_info` | youtube_tool.py | Channel stats |
| `youtube_search_captions` | youtube_tool.py | Available caption tracks |
| `start_process` | process_manager.py | Launch persistent background process |
| `stop_process` | process_manager.py | Terminate named process |
| `read_process_output` | process_manager.py | Read process stdout |
| `send_process_input` | process_manager.py | Send stdin to process |
| `list_processes` | process_manager.py | List tracked processes |
| `schedule_task` | scheduler_tool.py | Schedule recurring/one-time task |
| `list_scheduled_tasks` | scheduler_tool.py | List scheduled tasks |
| `cancel_scheduled_task` | scheduler_tool.py | Cancel scheduled task |
| `convert_video` | media_tool.py | Convert video via ffmpeg |
| `extract_audio` | media_tool.py | Extract audio via ffmpeg |
| `trim_clip` | media_tool.py | Trim video via ffmpeg |
| `merge_clips` | media_tool.py | Merge clips via ffmpeg |
| `get_media_info` | media_tool.py | Video/audio metadata via ffprobe |
| `send_email` | notification_tool.py | Send SMTP email |
| `test_email_config` | notification_tool.py | Verify SMTP connection |
| `watch_directory` | file_watcher.py | Watch folder for events |
| `stop_watching` | file_watcher.py | Stop directory watcher |
| `list_watches` | file_watcher.py | List active watchers |
| `email_connect` | email_tool.py | Connect to IMAP inbox |
| `email_scan_inbox` | email_tool.py | Scan inbox by category |
| `email_classify_and_plan` | email_tool.py | Local LLM classify emails |
| `email_delete_batch` | email_tool.py | Delete approved emails |
| `email_disconnect` | email_tool.py | Disconnect IMAP |
| `log_reflection` | episode_memory.py | Add reflection to task episode |
| `get_episode` | episode_memory.py | Retrieve full task episode |
| `get_recent_episodes` | episode_memory.py | Get last N episodes |
| `query_knowledge_graph` | knowledge_graph.py | Query concept connections |
| `add_graph_edge` | knowledge_graph.py | Add relationship to graph |

Generated tools in `agent_tools/generated/` auto-load on startup.
All generated files saved to `outputs/`.

## Memory Architecture
- **Short-term:** last N conversation turns (verbatim, history.json)
- **Semantic:** ChromaDB + nomic-embed-text embeddings
- **Long-term:** `memory/long_term.json`
  - tasks: goal, outcome, tools_used, duration, reflection, failure_type, timestamp
  - facts: key, value, source, timestamp, age, may_be_stale, expires
  - research: topic, findings, sources, timestamp, age (semantically indexed)
  - projects: name, description, outcome, file_count, dependencies
  - Caps: _MAX_TASKS=500, _MAX_RESEARCH=1000, _MAX_FACTS=500
- **Research index:** ChromaDB collection for semantic research retrieval
- **Performance metrics:** `memory/performance_metrics.json`
- **Knowledge graph:** `memory/semantic_graph.json`
  - Node types: concept, tool, project, skill, service, topic
  - Edge types: solved_with, used_together, related_to, caused_by, leads_to, prevented_by
- **User profile:** `memory/user_profile.json`
- **Session distillations:** `memory/session_distillations.json` (Phase 16e)
- **Improvement proposals:** `memory/improvement_proposals.json` (Phase 14c)
- **Goals:** `memory/goals.json` (Phase 13)
- **Strategies:** `memory/strategies.json` (Phase 17)
- **Pending batches:** `memory/pending_batches.json` (Phase 11.5b)

## Development Phases

### Phases 1–12 ✓ COMPLETE
All original phases complete. Key capabilities:
- Full tool ecosystem (65+ tools), autonomous task loop
- iGPU-accelerated local inference (qwen3:8b + qwen3:14b resident)
- Prompt caching (system block, 10% cost on cache reads)
- Episode journal with background reflections (num_predict=512 cap)
- Performance metrics database
- Semantic knowledge graph (auto-built from task history)
- Advanced memory query API (correlate, timeline, unified search)
- Datetime-aware queries, staleness flags, configurable research cache TTL

### Phase 11.5 — API Efficiency (IN PROGRESS)

**11.5a — Prompt Caching ✓ COMPLETE**
`cache_control` inside `system_block` list in `_run_claude_once()`.
Cache stats logged: `cache_read_input_tokens`, `cache_creation_input_tokens`.

**11.5b — Batch Processing Infrastructure (CURRENT TARGET)**
New `backend/batch_processor.py` wrapping Anthropic Message Batches API.
```python
submit_batch(requests, job_name) -> batch_id
poll_batch(batch_id) -> {status, succeeded, failed, total}
get_results(batch_id) -> [{custom_id, result_type, content}]
cancel_batch(batch_id) -> None
list_pending_batches() -> list[dict]
```
Batch IDs stored in `memory/pending_batches.json`.
APScheduler polls every 30 minutes, processes completed batches automatically.

**11.5c — Historical Reflection Backfill**
New tool: `backfill_reflections()` — submits all tasks without reflections
as a batch to Claude at 50% cost. Results written via `log_reflection()`.

---

### Phase 12 — Enhanced Memory Architecture ✓ COMPLETE

**12c upgrade — Causal graph edges (implement alongside 11.5b)**
Extend `backend/agent_tools/knowledge_graph.py` to support causal edge types:
`caused_by`, `leads_to`, `prevented_by` in addition to existing types.
When a task fails, the background graph update should add:
`{failure_reason} caused_by {last_tool_used}` and
`{last_tool_used} leads_to {failure_reason}`.
This enables "why did this happen?" queries — critical for Phase 14 failure analysis.
No new tools needed — `add_graph_edge` already accepts arbitrary relationship strings.
Just update the `_update_knowledge_graph()` in `task_runner.py` to include causal
edges when `outcome != "success"` and `failed_at_tool` is known.

---

### Phase 13 — Goal Tracking System (NEXT AFTER 11.5b)

**13a — Goal registry**
New file `memory/goals.json`.
```json
{
  "goal_id": "uuid",
  "title": "string",
  "description": "string",
  "status": "active|paused|complete|abandoned",
  "priority": 1-5,
  "created_date": "ISO",
  "target_date": "ISO|null",
  "milestones": [{"id","title","done","date_completed"}],
  "current_strategy": "string",
  "blockers": ["string"],
  "related_tasks": ["task_id"],
  "related_projects": ["project_name"],
  "progress_notes": [{"timestamp","note"}]
}
```
New tools: `create_goal`, `update_goal`, `list_goals`, `get_goal`.
UI: Goals tab in right panel with status badges and progress bars.

**13b — Progress tracking**
`log_task()` accepts optional `goal_id` param to link tasks to goals.
Milestone completion auto-detected from task goals.
New tools: `log_goal_progress(goal_id, note, milestone_id?)`,
`get_goal_progress(goal_id)`.

**13c — Autonomous planning**
`decompose_goal(goal_id)` — qwen3:14b breaks goal into sub-tasks with estimates.
`schedule_goal_work(goal_id, hours_per_week)` — APScheduler integration.
`detect_goal_blocker(goal_id)` — flags stalled goals (>7 days no progress).

**13d — Proactive reporting**
Weekly scheduled job generates goal status report.
`generate_goal_report()` tool for on-demand version.
Saved to `outputs/goal_reports/`, emailed if enabled.

---

### Phase 14 — Continuous Self-Reflection & Assessment

**14b — Failure classification**
Auto-classify failures into: tool_integration_error, logic_error, knowledge_gap,
resource_constraint, user_communication, external_failure.
Runs as background job on failed tasks. Populates `failure_type` field.
New tool: `classify_failure(task_id)`.

**14c — Pattern detection and rule generation**
*Research-informed upgrade: rules must be concrete and versioned.*
Background job every 10 tasks scans failures for patterns.
Each proposal generates a structured rule:
```json
{
  "rule_id": "uuid",
  "if": "condition description",
  "then": "action to take",
  "evidence": ["task_ids"],
  "status": "proposed|tested|active|retired",
  "created": "ISO",
  "effectiveness": null
}
```
Active rules (status="active") are appended to the system prompt as a
`[LEARNED RULES]` section — concrete preventive knowledge.
Store in `memory/improvement_proposals.json`.
New tools: `get_improvement_proposals()`, `apply_improvement_proposal(rule_id)`.

---

### Phase 15 — Advanced Self-Modification

**15a — Capability gap detection**
Local model checks tool coverage before complex tasks.
New tool: `analyze_capability_gap(task_goal)`.

**15b — Tool design pipeline**
Design-first workflow: `design_tool(gap_description)` generates spec,
user approves, `implement_tool_from_design(spec)` writes and registers it.
Adds sandbox validation (test run on simple input) before hot-reload.

**15c — Tool performance tracking**
Registry metadata per tool: created_date, call_count, success_count, created_by.
New tools: `get_tool_metadata(tool_name)`, `prune_unused_tools()`.

---

### Phase 16 — Background Memory Maintenance

**16a — Memory deduplication and cleanup with retention tiers**
*Research-informed upgrade: category-specific retention periods.*
Nightly APScheduler job at 3:00 AM.
Retention rules (never auto-deletes — flags only):
- `current_facts`: 90 days (exchange rates, current versions)
- `preferences`: 180 days (user preferences, settings)
- `skills`: 730 days / 2 years (learned capabilities)
- `project_history`: indefinite (never flagged stale)
- `research`: configurable via `research_cache_days` (default 7)
Produces cleanup report in `outputs/maintenance/`.

**16b — Memory summarization with batch processing**
Research entries >30 days old with >500 word findings:
Submit nightly batch to Claude (50% cost) for high-quality 100-word summaries.
Store both full + compressed versions.

**16c — Pattern extraction and cross-linking**
Local model scans recent episodes, updates knowledge graph with new edges.
Generates weekly "patterns learned" summary.
Feeds Phase 17 strategy registry.

**16d — Memory health reporting**
Monthly report: counts, growth rate, duplicate rate, knowledge gaps.
New tools: `run_memory_maintenance()`, `get_maintenance_report()`.

**16e — Session distillation (MiMoCode pattern)**
*Research-informed addition: extract key learnings at session end.*
On WebSocket disconnect, fire background qwen3:14b job:
reads current session's tasks (last 2 hours), extracts 3-5 compressed learnings.
Store in `memory/session_distillations.json` (keep last 30).
At next session start, last 3 distillations injected into context:
"Key learnings from recent sessions: ..."
This gives the agent continuity without re-reading full history.

---

### Phase 17 — Strategy Evolution Loop

**17a — Strategy extraction via batch processing**
`memory/strategies.json`. Minimum 3 successful uses per strategy.
Batch submit similar task groups to Claude overnight.

**17b — Contextual strategy application**
New tools: `recall_relevant_strategies`, `adapt_strategy`,
`track_strategy_effectiveness`.

**17c — Domain proficiency tracking**
Success rate per domain. New tools: `get_domain_proficiency`,
`identify_weak_domains`.

**17d — Knowledge synthesis reports**
Monthly batch to Claude Sonnet (50% cost). Saved to `outputs/synthesis/`.
New tool: `generate_synthesis_report(period)`.

---

## Research-Derived Architecture Notes
*From Tier 4 research: "Best Practices for Autonomous AI Agents 2026"*

**Confirmed correct in current architecture:**
- Multi-layer memory (working + session + persistent) ✓
- Hybrid local + cloud strategy ✓
- Graph-based memory as dominant paradigm ✓
- Closed-loop self-improvement: error→RCA→rules→memory→skills = Phases 12-17 ✓

**Specific gaps being addressed in roadmap:**
- Causal graph edges (why relationships) → Phase 12c upgrade
- Knowledge distillation at session end → Phase 16e
- Concrete rule generation from failures → Phase 14c upgrade
- Category-specific memory retention → Phase 16a upgrade
- Validator subagent for generated tools → Phase 15b upgrade

**Metric to track (from research):**
- Error frequency: same bugs recurring? → failure_type field + Phase 14c
- Rule effectiveness: new rules preventing failures? → Phase 14c status tracking
- Skill coverage: what task classes can agent handle? → Phase 17c domain proficiency
- Latency: getting faster over time? → performance_metrics.json trend

## Project Structure
```
agent/
├── backend/
│   ├── main.py
│   ├── agent_core.py            # system_block caching, _run_claude_once()
│   ├── task_runner.py           # causal edges in _update_knowledge_graph()
│   ├── task_scheduler.py
│   ├── batch_processor.py       # Phase 11.5b
│   ├── run.py
│   ├── agent_tools/
│   │   ├── (all existing files)
│   │   ├── episode_memory.py
│   │   ├── knowledge_graph.py   # + causal edge types
│   │   ├── goal_tracker.py      # Phase 13
│   │   ├── reflection_engine.py # Phase 14
│   │   ├── capability_tools.py  # Phase 15
│   │   ├── memory_maintenance.py # Phase 16
│   │   ├── strategy_tools.py    # Phase 17
│   │   └── generated/
│   └── memory/
│       ├── context.py
│       ├── embeddings.py
│       ├── long_term.py         # _MAX_TASKS=500
│       └── performance.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── memory/
│   ├── (all existing files)
│   ├── pending_batches.json
│   ├── session_distillations.json
│   ├── improvement_proposals.json
│   ├── goals.json
│   └── strategies.json
└── outputs/
    ├── goal_reports/
    ├── maintenance/
    └── synthesis/
```

## Coding Principles
1. **Always modular** — each tool in its own file; no circular dependencies
2. **Token-efficient** — local preprocessing + prompt caching + batch processing
3. **Incremental complexity** — never skip a phase; stabilize before extending
4. **Security first** — permission checks; self-modification to generated/ only
5. **Commented code** — developer is learning; explain non-trivial parts
6. **Windows compatible** — always use run.py (ProactorEventLoop)
7. **Designed for long tasks** — context management, checkpointing, compression
8. **Local model first** — qwen3 on iGPU; Claude API when quality matters
9. **Background first** — never block user on maintenance/reflection/analysis

## How to Help
- **Always provide every created or edited file as a downloadable file**
- Full file content, not diffs, unless partial is explicitly requested
- 2-3 options with pro/con for architectural decisions
- Explicitly flag security risks
- After files: "What changed / What to do next" summary
- After files: "Test it" section with 3-5 concrete test prompts

## Terminology
- **Preprocessing tier:** qwen3:8b on iGPU
- **Agent tier:** qwen3:14b on iGPU
- **Intent router:** SIMPLE/LOCAL_SUFFICIENT/TOOL/COMPLEX
- **Prompt caching:** 10% cost on cache reads, system_block format
- **Batch processing:** 50% cost, async overnight, Anthropic Batches API
- **Session distillation:** compressed learnings extracted at session end
- **Causal graph:** why relationships (caused_by, leads_to, prevented_by)
- **Rule generation:** concrete if/then rules from failure patterns
- **Retention tiers:** category-specific memory staleness thresholds
- **Task runner:** long-running autonomous execution loop
- **Episode:** completed task with goal, tools, outcome, reflection, failure_type
- **Strategy registry:** proven approaches with evidence and success rates
- **Domain proficiency:** agent success rate per task category
