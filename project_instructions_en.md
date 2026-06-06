# Personal AI Agent – Project Instructions

## Project Goal
Build a modular, locally-run personal AI agent using Claude API as the LLM "brain",
with an extensible tool system, Python backend, and web UI.

## Developer Context
- **Developer:** 3rd-year Computer Engineering student
- **OS:** Windows (WSL2 available if needed)
- **Hardware:** Lenovo ThinkPad E14 Gen7 (Intel Core Ultra 5, Intel Arc 140V, 32GB DDR5 RAM)
- **LLM:** Claude API (Anthropic) – claude-haiku-4-5 by default (token efficiency),
  claude-sonnet-4-6 for complex tasks
- **Local LLM:** Ollama (qwen2.5:7b prompt optimizer / qwen2.5:14b local agent mode)
- **Self-hosted search:** SearXNG running on Docker at http://localhost:8888
- **Goal:** Learning + practical utility, incrementally extensible system

## Tech Stack
- **Backend:** Python (FastAPI)
- **Frontend:** HTML/CSS/JS (chat UI), migratable to React later
- **LLM Integration:** Anthropic Python SDK, tool use (function calling)
- **Local LLM:** Ollama (http://localhost:11434), used as a secondary tier
- **Vector Memory:** ChromaDB + nomic-embed-text (via Ollama)
- **Web Search:** SearXNG (primary) + DuckDuckGo HTML scraping (fallback)
- **Code Execution:** Sandbox (subprocess + restricted permissions) — Phase 3
- **Package Manager:** pip + virtualenv

## LLM Two-Tier Architecture
```
User message
     │
     ▼
[Local LLM - Ollama]  ──  prompt optimization, intent classification,
     │                     context compression, offline fallback
     │
     └──  complex tasks, tool use, final answers  ──►  [Claude API]
```

**Local LLM (Ollama) is used for:**
- Rewriting raw user messages into clean prompts before Claude API calls
- Summarizing long conversation history to save context space
- Full local-mode agentic loop (no Claude API) when toggled on

**Claude API is used for:**
- All tool use and agentic reasoning (default mode)
- Complex multi-step tasks
- Final answers to the user

**Config controls (all runtime-adjustable via settings panel):**
- `use_prompt_optimizer`: toggle local prompt rewriting on/off
- `local_fallback`: fall back to local LLM if Claude API is unreachable
- `local_mode`: run entire agent loop through Ollama, no Claude API
- `primary`: default Claude model (claude-haiku-4-5)
- `complex`: model for complex tasks (claude-sonnet-4-6)
- `local`: Ollama model for prompt optimizer (qwen2.5:7b)
- `local_agent`: Ollama model for local mode agentic loop (qwen2.5:14b)
- `local_agent_timeout`: per-request timeout in seconds (default 300)

## Current Tool Inventory (Phase 2 complete)
All tools are registered at startup and visible to Claude via the tool registry.

| Tool | File | Description |
|---|---|---|
| `read_file` | filesystem.py | Read a file from disk |
| `write_file` | filesystem.py | Write/append to a file (permission check) |
| `list_directory` | filesystem.py | List directory contents + emit tree update |
| `list_capabilities` | capabilities.py | Introspect live tool registry at call time |
| `search_web` | web.py | SearXNG search with DDG fallback |
| `fetch_page` | web.py | Fetch + strip HTML from any URL |
| `get_system_info` | system_info.py | CPU, RAM, disk, running Ollama models |
| `analyze_file` | file_analysis.py | Size, lines, words, estimated token count |

After any `write_file` or `list_directory` call, a `tree_update` WebSocket event
is broadcast to the frontend with the current project folder tree.

## Development Phases

### Phase 1 – Base Agent ✓ COMPLETE
- FastAPI backend, WebSocket, Chat UI
- Claude API with tool use loop
- Filesystem tools + permission layer
- Ollama client + prompt optimizer
- Persistent conversation history (JSON)

### Phase 2 – Extended Tools ✓ COMPLETE
- Semantic vector memory (ChromaDB + nomic-embed-text)
- Tiered context: recent verbatim + LLM summary + semantic retrieval
- Web search (SearXNG self-hosted + DDG fallback)
- Page fetching (httpx + HTML stripping)
- System info tool (psutil + Ollama model list)
- File analysis tool
- Self-aware capabilities tool
- Settings panel in UI (model, mode, all config values)
- Local-only mode toggle (full agentic loop via Ollama)
- Runtime config changes via WebSocket

### Phase 3 – Autonomous Agent (CURRENT TARGET)
The goal is for the agent to work independently on complex tasks — writing and running
code, planning multi-step work, modifying its own tools, and running indefinitely
until a task is complete or it needs user input.

**3a – Code execution sandbox**
- New tool `execute_code(code, language)` in `agent_tools/code_executor.py`
- Runs code in a subprocess with timeout + output capture (stdout, stderr, exit code)
- Subprocess sandbox: restricted, no network by default, configurable timeout
- Returns structured result; agent can read output and iterate
- Permission check before first execution in a session

**3b – Long-running task loop**
- Replace hard `max_iterations_per_turn` cap with a resumable task runner
- Task state persisted to disk (JSON) so runs survive interruptions
- UI shows live step-by-step progress (step name, status, duration)
- "Stop" button cancels gracefully, saving last checkpoint
- Agent can pause and ask user a question mid-task, then resume

**3c – Self-modification**
- Agent can write new tool files to `agent_tools/`
- Hot-reload mechanism: registry re-scans `agent_tools/` without server restart
- New tools are validated (syntax check + sandbox test run) before registration
- Self-modification restricted to `agent_tools/` only —
  `agent_core.py` and `main.py` are read-only to the agent

**3d – Task planning**
- Before complex tasks, agent generates explicit numbered plan
- Plan shown to user for approval before execution starts
- Each step's output feeds into the next step's context
- User can intervene, redirect, or abort between steps

**3e – Long-term task memory**
- Persistent task log: what was attempted, what failed, what succeeded
- Stored separately from conversation history, survives across sessions
- Agent can query past task outcomes to avoid repeating mistakes

## Project Structure
```
agent/
├── backend/
│   ├── main.py                  # FastAPI app, WebSocket, settings handlers
│   ├── agent_core.py            # LLM loop, tool dispatch, local/claude routing
│   ├── agent_tools/
│   │   ├── __init__.py          # Tool registry
│   │   ├── filesystem.py        # read_file, write_file, list_directory
│   │   ├── capabilities.py      # list_capabilities
│   │   ├── web.py               # search_web, fetch_page
│   │   ├── system_info.py       # get_system_info
│   │   ├── file_analysis.py     # analyze_file
│   │   ├── local_llm.py         # Ollama client, prompt optimizer
│   │   ├── code_executor.py     # execute_code (Phase 3a)
│   │   └── SEARXNG_SETUP.md     # Docker setup guide
│   └── memory/
│       ├── context.py           # History load/save/trim
│       └── embeddings.py        # ChromaDB vector store
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── memory/
│   └── history.json             # Persisted conversation history
├── config.json                  # All settings (runtime-editable via UI)
├── .env                         # ANTHROPIC_API_KEY (never commit)
├── env.example                  # Template for .env
├── requirements.txt
└── SEARXNG_SETUP.md             # (should also be in project root)
```

## Coding Principles
1. **Always modular** – each tool in its own file; core must not depend on tool implementations
2. **Token-efficient** – use local LLM for preprocessing; system prompts short and informative
3. **Incremental complexity** – never skip a phase; stabilize before extending
4. **Security first** – permission check before every file/code/network operation;
   destructive actions always require user confirmation;
   agent self-modification restricted to agent_tools/ only
5. **Commented code** – developer is learning, explain non-trivial parts
6. **Windows compatible** – paths, subprocess calls, etc. must work on Windows

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

## Terminology
- **Agent core:** the LLM + tool dispatch logic
- **Tool:** a concrete function the agent can call (e.g. write file)
- **Tool registry:** the registry of tools the agent knows about
- **Permission layer:** the user approval system
- **Context window:** the active LLM context, to be managed deliberately
- **Local LLM tier:** Ollama-based local model for cheap, fast, offline subtasks
- **Prompt optimizer:** local LLM subtask that rewrites raw user input before Claude API call
- **Task runner:** the Phase 3 long-running autonomous execution loop
- **Hot-reload:** re-registering tools at runtime without restarting the server
