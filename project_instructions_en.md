# Personal AI Agent – Project Instructions

## Project Goal
Build a modular, universal, locally-run personal AI agent using Claude API as the
LLM "brain", with an extensible tool system, Python backend, and web UI.
The long-term goal is an agent capable of truly autonomous complex tasks —
developing software, deep research, self-optimization, and eventually extending
its own capabilities without human intervention for each step.

## Developer Context
- **Developer:** 3rd-year Computer Engineering student
- **OS:** Windows (WSL2 available if needed)
- **Hardware:** Lenovo ThinkPad E14 Gen7 (Intel Core Ultra 5, Intel Arc 140V, 32GB DDR5 RAM)
- **LLM:** Claude API (Anthropic) – claude-haiku-4-5 by default (token efficiency),
  claude-sonnet-4-6 for complex tasks
- **Local LLM:** Ollama (qwen2.5:14b for ALL local tasks)
- **Self-hosted search:** SearXNG running on Docker at http://localhost:8888
- **Goal:** Learning + practical utility, incrementally extensible system

## Tech Stack
- **Backend:** Python (FastAPI)
- **Frontend:** HTML/CSS/JS (chat UI), migratable to React later
- **LLM Integration:** Anthropic Python SDK, tool use (function calling)
- **Local LLM:** Ollama (http://localhost:11434), qwen2.5:14b for all local tasks
- **Vector Memory:** ChromaDB + nomic-embed-text (via Ollama)
- **Web Search:** SearXNG (primary) + DuckDuckGo HTML scraping (fallback)
- **Code Execution:** Subprocess sandbox with permission layer
- **Browser Automation:** Playwright (read + write, Chromium headless)
- **Package Manager:** pip + virtualenv

## LLM Two-Tier Architecture
```
User message
     │
     ▼
[Local LLM - qwen2.5:14b]
     ├── Intent routing: classify message complexity + type
     │     → simple/conversational: answer locally, skip Claude API
     │     → tool call needed: route to Claude Haiku
     │     → complex reasoning: route to Claude Sonnet
     ├── Prompt optimization: rewrite raw input into clean prompt
     ├── Tool result compression: strip verbose output before Claude sees it
     ├── History summarization: compress old turns to save context space
     ├── Mid-task step summarization: summarize completed steps during long runs
     ├── Code pre-validation: catch obvious bugs before execute_code is called
     ├── Tool pre-filtering: select 10-12 relevant tools per request (~50% token saving)
     ├── Vague message enrichment: expand short messages using conversation context
     └── Offline fallback: full local agentic loop when Claude API unavailable
          │
          ▼
     [Claude API]
     ├── claude-haiku-4-5: tool calls, structured tasks, most interactions
     └── claude-sonnet-4-6: complex reasoning, planning, self-modification,
                             research synthesis, app development
```

**Key efficiency principle:** Claude's context window should only ever contain
clean, compressed, relevant information. The local tier is a preprocessing
pipeline, not just a fallback.

**Config controls (all runtime-adjustable via settings panel):**
- `use_prompt_optimizer`: toggle local prompt rewriting on/off
- `use_intent_routing`: toggle local intent classification on/off
- `use_tool_compression`: toggle local tool result compression on/off
- `use_code_prevalidation`: toggle local code pre-validation on/off
- `use_tool_prefilter`: toggle local tool pre-filtering on/off
- `local_fallback`: fall back to local LLM if Claude API is unreachable
- `local_mode`: run entire agent loop through Ollama, no Claude API
- `primary`: default Claude model (claude-haiku-4-5)
- `complex`: model for complex tasks (claude-sonnet-4-6)
- `local`: Ollama model for all local tasks (qwen2.5:14b)
- `local_agent`: Ollama model for local mode agentic loop (qwen2.5:14b)
- `local_agent_timeout`: per-request timeout in seconds (default 300)

## Current Tool Inventory (Phase 5 complete)
All tools are registered at startup and visible to Claude via the tool registry.

| Tool | File | Description |
|---|---|---|
| `read_file` | filesystem.py | Read a file from disk |
| `write_file` | filesystem.py | Write/append to a file (permission check) |
| `list_directory` | filesystem.py | List directory contents + emit tree update |
| `list_outputs` | filesystem.py | List files in outputs/ directory |
| `patch_file` | filesystem.py | Apply targeted line-range edits to a file |
| `list_capabilities` | capabilities.py | Introspect live tool registry at call time |
| `search_web` | web.py | SearXNG search with DDG fallback |
| `fetch_page` | web.py | Fetch + strip HTML from any URL |
| `get_system_info` | system_info.py | CPU, RAM, disk, running Ollama models |
| `analyze_file` | file_analysis.py | Size, lines, words, estimated token count |
| `execute_code` | code_executor.py | Run Python or Bash in subprocess sandbox |
| `install_package` | code_executor.py | pip install a package with confirmation |
| `write_tool` | tool_writer.py | Write + validate a new tool file |
| `reload_tool` | tool_writer.py | Hot-reload a generated tool into the live registry |
| `log_research` | memory_tool.py | Save research findings to long-term memory |
| `recall_memory` | memory_tool.py | Query past tasks, facts, and research |
| `log_fact` | memory_tool.py | Store a specific fact to long-term memory |
| `recall_projects` | memory_tool.py | Query past built projects |
| `read_user_profile` | self_knowledge.py | Read memory/user_profile.json |
| `scan_system` | self_knowledge.py | Scan installed tools, packages, projects |
| `update_user_profile` | profile_updater.py | Update fields in user_profile.json |
| `deep_research` | research_mode.py | Generate structured research plan for a goal |
| `browser_open` | browser.py | Navigate to URL in headless Chromium |
| `browser_read` | browser.py | Extract text from current browser page |
| `browser_screenshot` | browser.py | Save PNG of current browser page |
| `browser_click` | browser.py | Click an element on the current page |
| `browser_fill` | browser.py | Fill an input field on the current page |
| `browser_get_url` | browser.py | Get current page URL and title |
| `scaffold_project` | project_scaffold.py | Generate project architecture + control tools |
| `get_project_status` | project_manager.py | Check which files are done vs pending |
| `mark_file_complete` | project_manager.py | Mark a project file as implemented |
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
| `list_processes` | process_manager.py | List all tracked processes and their status |
| `schedule_task` | scheduler_tool.py | Schedule a recurring or one-time task |
| `list_scheduled_tasks` | scheduler_tool.py | List all scheduled tasks with next run time |
| `cancel_scheduled_task` | scheduler_tool.py | Cancel a scheduled task |

Generated tools live in `agent_tools/generated/` and auto-load on startup.
Scaffolding a project auto-generates a companion control tool in generated/.
All generated files (reports, scripts, data) are saved to `outputs/`.

## Memory Architecture
- **Short-term:** last N conversation turns (verbatim, JSON)
- **Semantic:** ChromaDB vector store with nomic-embed-text embeddings
- **Long-term:** `memory/long_term.json` — tasks, facts, research, projects
- **Research index:** ChromaDB collection for semantic research retrieval
- **User profile:** `memory/user_profile.json` — manually maintained
- **Task checkpoint:** `memory/current_task.json` — last task state, survives reconnect
- **Project progress:** `outputs/{name}/progress.json` — per-project build state
- **Scheduled tasks:** `memory/scheduled_tasks.json` — persisted across restarts

## Development Phases

### Phases 1–5 ✓ COMPLETE
All original planned phases are complete:
- Base agent, extended tools, autonomous agent (3a–3i)
- Software development agent (4a–4d)
- Quality & reliability (4.5)
- External integrations (5a GitHub, 5b Credentials, 5c YouTube, 5d Write Browser, 5e Scheduler)

### Phase 6 — Long Task Reliability (CURRENT TARGET)
Making long autonomous tasks more robust and resumable.

**6a — ask_user tool**
Agent can pause mid-task, ask the user a question, and resume with the answer.
Prevents the agent from guessing when it's unsure during a long build.

**6b — Project state snapshots**
Rich state file updated at every step: what's done, what failed, key decisions,
next step. Makes resumption after interruption reliable without re-reading all files.

### Phase 7 — UI Overhaul
Frontend updated to expose all Phase 5+ capabilities visually.
- Running processes panel (start/stop/output)
- Scheduler view (list, add, cancel)
- Credential manager UI (list, add — never shows values)
- Agent analytics dashboard (task success rates, tool usage, duration trends)
- Live process output streaming in chat

### Phase 8 — Self-Improvement Infrastructure
- Agent analytics: track success rates, failure patterns, tool usage
- Auto-updating user profile from conversation patterns
- Agent-written improvement suggestions based on failure analysis
- Context usage awareness tool

### Phase 9 — Media & Notifications
- ffmpeg wrapper tools: convert_video, extract_audio, trim_clip, merge_clips
- SMTP email tool: send notifications when tasks complete
- File watcher: react to file system events, trigger tasks automatically

## Project Structure
```
agent/
├── backend/
│   ├── main.py                  # FastAPI app, WebSocket, settings, broadcast
│   ├── agent_core.py            # LLM loop, tool dispatch, routing, tiered prompt
│   ├── task_runner.py           # Long-running task loop, sanitizer, compression
│   ├── task_scheduler.py        # APScheduler wrapper, persistent schedules
│   ├── run.py                   # Windows-compatible server launcher
│   ├── agent_tools/
│   │   ├── __init__.py          # Tool registry with get_all_definitions()
│   │   ├── filesystem.py        # File tools + list_outputs + patch_file + cache
│   │   ├── capabilities.py      # list_capabilities
│   │   ├── web.py               # search_web, fetch_page
│   │   ├── system_info.py       # get_system_info
│   │   ├── file_analysis.py     # analyze_file
│   │   ├── local_llm.py         # Ollama client, all local LLM tasks
│   │   ├── code_executor.py     # execute_code, install_package, streaming
│   │   ├── tool_writer.py       # write_tool (with validation), reload_tool
│   │   ├── hot_reload.py        # validation + importlib hot-reload
│   │   ├── memory_tool.py       # log_research, recall_memory, log_fact, recall_projects
│   │   ├── self_knowledge.py    # read_user_profile, scan_system
│   │   ├── profile_updater.py   # update_user_profile
│   │   ├── research_mode.py     # deep_research
│   │   ├── browser.py           # browser_open/read/screenshot/click/fill/get_url
│   │   ├── project_scaffold.py  # scaffold_project + auto control tool generation
│   │   ├── project_manager.py   # get_project_status, mark_file_complete
│   │   ├── project_tester.py    # run_project_test
│   │   ├── github_tool.py       # github_* tools
│   │   ├── credentials.py       # store/get/list_credentials (Fernet encrypted)
│   │   ├── youtube_tool.py      # youtube_* tools
│   │   ├── process_manager.py   # start/stop/read/send/list processes
│   │   ├── scheduler_tool.py    # schedule/list/cancel scheduled tasks
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
│   ├── long_term.json
│   ├── user_profile.json
│   ├── credentials.json         # encrypted, gitignored
│   ├── scheduled_tasks.json
│   └── vectors/
├── outputs/
│   └── {project_name}/
│       ├── scaffold.json
│       ├── progress.json
│       ├── state.json           # rich state snapshot (Phase 6b)
│       └── (project files)
├── config.json
├── .env
├── .gitignore
├── requirements.txt
├── SEARXNG_SETUP.md
└── PLAYWRIGHT_SETUP.md
```

## Coding Principles
1. **Always modular** – each tool in its own file; core must not depend on tool
   implementations; new capabilities should slot in without touching existing code
2. **Token-efficient** – local LLM preprocesses everything before Claude sees it;
   Claude's context should be clean and dense, never verbose and redundant
3. **Incremental complexity** – never skip a phase; stabilize before extending
4. **Security first** – permission check before every file/code/network operation;
   destructive actions always require user confirmation;
   agent self-modification restricted to agent_tools/generated/ only;
   agent_core.py, main.py, and task_runner.py are read-only to the agent
5. **Commented code** – developer is learning, explain non-trivial parts
6. **Windows compatible** – paths, subprocess calls, etc. must work on Windows;
   always use run.py (not uvicorn directly) to ensure ProactorEventLoop
7. **Designed for long tasks** – context management, checkpointing, and compression
   must support tasks that run for minutes or hours with many tool calls

## How to Help
- **Always provide every created or edited file as a downloadable file** —
  never just show code in a code block when a file is being created or modified
- When writing code, always provide the full file content (not just diffs),
  unless a partial snippet is explicitly requested
- For architectural decisions, provide 2-3 options with a short pro/con list
- When debugging, ask for the full error message and relevant code
- When developing a new tool, always show how it fits into the existing system
- Explicitly flag any security risks you notice
- After delivering files, add a short "What changed / What to do next" summary
- After delivering files, always include a "Test it" section with 3-5 concrete
  prompts to verify the new functionality works correctly

## Terminology
- **Agent core:** the LLM + tool dispatch logic
- **Tool:** a concrete function the agent can call
- **Tool registry:** the live registry of all registered tools
- **Permission layer:** the user approval system for destructive actions
- **Context window:** the active LLM context, managed deliberately
- **Local LLM tier:** qwen2.5:14b via Ollama — preprocessing pipeline
- **Intent router:** local classifier that decides which model tier handles a message
- **Prompt optimizer:** local rewrite of raw user input before Claude API call
- **Tool result compression:** local stripping of verbose tool output
- **Task runner:** the long-running autonomous execution loop
- **Hot-reload:** re-registering tools at runtime without server restart
- **Generated tools:** agent-written + scaffold control tools in agent_tools/generated/
- **Project scaffold:** upfront architecture plan before multi-file implementation
- **Control tool:** auto-generated tool for starting/stopping a built project
- **State snapshot:** rich project state file updated at every step (Phase 6b)
