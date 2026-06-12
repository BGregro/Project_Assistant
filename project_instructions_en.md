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
- **Browser Automation:** Playwright (read-only, Chromium headless)
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

## Current Tool Inventory (Phase 5a complete)
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
| `write_tool` | tool_writer.py | Write a new tool file to agent_tools/generated/ |
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
| `scaffold_project` | project_scaffold.py | Generate project architecture before coding |
| `get_project_status` | project_manager.py | Check which files are done vs pending |
| `mark_file_complete` | project_manager.py | Mark a project file as implemented |
| `run_project_test` | project_tester.py | Run project entry point and capture output |
| `github_list_repos` | github_tool.py | List GitHub repositories |
| `github_create_repo` | github_tool.py | Create a new GitHub repository |
| `github_push_file` | github_tool.py | Push a file to a GitHub repository |
| `github_read_file` | github_tool.py | Read a file from a GitHub repository |
| `github_list_files` | github_tool.py | List files in a GitHub repository |
| `github_create_issue` | github_tool.py | Create a GitHub issue |

Generated tools live in `agent_tools/generated/` and auto-load on startup.
All generated files (reports, scripts, data) are saved to `outputs/`.

## Memory Architecture
- **Short-term:** last N conversation turns (verbatim, JSON)
- **Semantic:** ChromaDB vector store with nomic-embed-text embeddings
- **Long-term:** `memory/long_term.json` — tasks, facts, research, projects
- **Research index:** ChromaDB collection for semantic research retrieval
- **User profile:** `memory/user_profile.json` — manually maintained
- **Task checkpoint:** `memory/current_task.json` — last task state, survives reconnect
- **Project progress:** `outputs/{name}/progress.json` — per-project build state

## Development Phases

### Phase 1 – Base Agent ✓ COMPLETE
### Phase 2 – Extended Tools ✓ COMPLETE
### Phase 3 – Autonomous Agent ✓ COMPLETE
- 3a Code execution, 3b Long-running tasks, 3c Self-modification
- 3d Efficiency layer, 3e Task planning, 3f Long-term memory
- 3g Deep self-knowledge, 3h Structured research, 3i Browser automation
- Gap fixes: semantic memory matching, outputs awareness

### Phase 4 – Software Development Agent ✓ COMPLETE
- 4a Project scaffolding (scaffold_project tool)
- 4b Incremental implementation with progress tracking
- 4c Integration testing (run_project_test tool)
- 4d Project memory (logged to long_term.json on success)

### Phase 4.5 – Quality & Reliability ✓ COMPLETE
- patch_file tool for targeted line-range edits
- install_package tool with pip + confirmation
- Streaming execution output (line-by-line via WebSocket)
- UI overhaul: dark theme, collapsible task containers, status bar,
  task history panel, tool block collapsing
- State persistence on WebSocket reconnect
- Consecutive failure detection (stops infinite retry loops)
- Message history sanitizer (repairs orphaned tool_use blocks after 429/errors)
- Tool pre-filtering via local LLM (~50% input token reduction)

### Phase 5 – External Service Integrations (IN PROGRESS)

**5a – GitHub integration ✓ COMPLETE**
- Personal access token (GITHUB_TOKEN in .env)
- github_list_repos, github_create_repo, github_push_file,
  github_read_file, github_list_files, github_create_issue

**5b – Credential manager (CURRENT TARGET)**
- Encrypted local storage for API keys and tokens (Fernet)
- store_credential, get_credential, list_credentials tools
- Keys stored in memory/credentials.json (gitignored)

**5c – YouTube Data API**
- Search videos, get channel analytics, manage playlists
- Upload support (requires 5b for credential storage)
- Direct enabler for YouTube Shorts automation project

**5d – Write-mode browser automation**
- browser_click and browser_fill tools
- Enables form submission, logins, web automation workflows
- All write actions require explicit user approval

**5e – Scheduled/recurring tasks**
- APScheduler-based task queue
- Config-driven schedules (interval, cron, one-time)
- Persisted across server restarts

## Planned Improvements
- **Vague message context injection:** short/ambiguous messages ("continue",
  "yes", "proceed") are enriched with recent task context before routing
- **Tiered system prompt:** base prompt always sent (~500 tokens), contextual
  sections appended only when relevant — saves ~1,500 tokens per call
- **Parallel tool execution:** asyncio.gather for concurrent tool dispatch
  when Claude returns multiple tool_use blocks in one response
- **File content caching:** in-memory cache keyed by (path, mtime) in
  filesystem.py — avoids re-reading unchanged files during long projects

## Project Structure
```
agent/
├── backend/
│   ├── main.py                  # FastAPI app, WebSocket, settings handlers
│   ├── agent_core.py            # LLM loop, tool dispatch, routing
│   ├── task_runner.py           # Long-running task loop
│   ├── run.py                   # Windows-compatible server launcher
│   ├── agent_tools/
│   │   ├── __init__.py          # Tool registry
│   │   ├── filesystem.py        # File tools + list_outputs + patch_file
│   │   ├── capabilities.py      # list_capabilities
│   │   ├── web.py               # search_web, fetch_page
│   │   ├── system_info.py       # get_system_info
│   │   ├── file_analysis.py     # analyze_file
│   │   ├── local_llm.py         # Ollama client, all local LLM tasks
│   │   ├── code_executor.py     # execute_code, install_package
│   │   ├── tool_writer.py       # write_tool, reload_tool
│   │   ├── hot_reload.py        # validation + importlib hot-reload
│   │   ├── memory_tool.py       # log_research, recall_memory, log_fact, recall_projects
│   │   ├── self_knowledge.py    # read_user_profile, scan_system
│   │   ├── profile_updater.py   # update_user_profile
│   │   ├── research_mode.py     # deep_research
│   │   ├── browser.py           # browser_open, browser_read, browser_screenshot
│   │   ├── project_scaffold.py  # scaffold_project
│   │   ├── project_manager.py   # get_project_status, mark_file_complete
│   │   ├── project_tester.py    # run_project_test
│   │   ├── github_tool.py       # github_* tools (Phase 5a)
│   │   ├── generated/           # agent-written tools (auto-loaded)
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
│   └── vectors/
├── outputs/
│   └── {project_name}/
│       ├── scaffold.json
│       ├── progress.json
│       └── (project files)
├── config.json
├── .env
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
- **Generated tools:** agent-written tools in agent_tools/generated/
- **Project scaffold:** upfront architecture plan before multi-file implementation
