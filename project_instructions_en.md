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
- `local_fallback`: fall back to local LLM if Claude API is unreachable
- `local_mode`: run entire agent loop through Ollama, no Claude API
- `primary`: default Claude model (claude-haiku-4-5)
- `complex`: model for complex tasks (claude-sonnet-4-6)
- `local`: Ollama model for all local tasks (qwen2.5:14b)
- `local_agent`: Ollama model for local mode agentic loop (qwen2.5:14b)
- `local_agent_timeout`: per-request timeout in seconds (default 300)

## Current Tool Inventory (Phase 3c complete)
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
| `execute_code` | code_executor.py | Run Python or Bash in a subprocess sandbox |
| `write_tool` | tool_writer.py | Write a new tool file to agent_tools/generated/ |
| `reload_tool` | tool_writer.py | Hot-reload a generated tool into the live registry |

After any `write_file` or `list_directory` call, a `tree_update` WebSocket event
is broadcast to the frontend with the current project folder tree.
Agent-written tools live in `agent_tools/generated/` and auto-load on startup.

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

### Phase 3 – Autonomous Agent (IN PROGRESS)

**3a – Code execution sandbox ✓ COMPLETE**
- execute_code(code, language) in code_executor.py
- Subprocess with timeout + output capture
- Permission check before execution
- Agent iterates on failure (system prompt enforced)

**3b – Long-running task loop ✓ COMPLETE**
- TaskRunner: resumable, cancellable, indefinite iteration
- Task state persisted to disk (memory/current_task.json)
- Mid-task user message injection via asyncio.Queue
- Stop button with graceful cancellation
- Live step-by-step progress in UI Tasks tab
- asyncio.create_task keeps WebSocket responsive during runs

**3c – Self-modification ✓ COMPLETE**
- Agent writes tool files to agent_tools/generated/
- Validation: syntax check + async def + register_ function required
- Hot-reload via importlib without server restart
- Auto-load generated tools on server startup
- Self-modification restricted to agent_tools/generated/ only

**3d – Efficiency layer (CURRENT TARGET)**
Intent routing, tool result compression, mid-task summarization,
code pre-validation — all running locally to reduce Claude API cost
and enable longer autonomous runs.

- Intent router: local model classifies each message and routes to the
  right model tier (local / haiku / sonnet) before any API call is made
- Tool result compression: local model strips verbose tool outputs down
  to relevant content before appending to Claude's context
- Mid-task step summarization: during long runs, completed steps are
  compressed by local model to prevent context window overflow
- Code pre-validation: local model reviews agent-written code before
  execute_code is called, catching obvious bugs without an API round-trip

**3e – Task planning (NEXT AFTER 3d)**
Before complex tasks, agent generates an explicit numbered plan.
Plan shown to user for approval before execution starts.
Each step's output feeds into the next step's context.
User can intervene, redirect, or abort between steps.
Designed for genuinely long tasks (app development, deep research).

**3f – Long-term memory**
Persistent task log: what was attempted, what failed, what succeeded.
Stored separately from conversation history, survives across sessions.
Agent queries past outcomes to avoid repeating mistakes.
Research findings stored and retrievable across sessions.
Foundation for self-improvement over time.

**3g – Deep self-knowledge**
Rich user profile tool: reads memory/user_profile.json (manually
maintained by user) containing skills, constraints, available tools,
accounts, projects, goals.
System scanner: installed software, programming languages detected
from project files, disk contents summary.
Agent reads profile automatically before any self-directed task.
User can ask agent to update the profile based on conversation.

**3h – Structured research mode**
Given a high-level goal, agent generates a set of sub-questions.
Pursues each sub-question with web search + page fetching.
Evaluates findings against user-defined criteria (legal, cost,
effort, relevance to user profile).
Produces a structured ranked report as a final deliverable.
Enables prompts like "find me ways to make money online that
fit my skills and constraints."

**3i – Browser automation**
Playwright integration for full web interaction beyond static fetching.
Form filling, navigation, login flows, content extraction from
JavaScript-rendered pages.
Required for: interacting with web platforms, submitting work,
reading dynamic content.
All browser actions require explicit user approval.

**3j – External service integrations**
API integrations for platforms relevant to user goals.
Credential management (encrypted local storage).
Enables agent to autonomously interact with external services
after user approval of each action class.

## Project Structure
```
agent/
├── backend/
│   ├── main.py                  # FastAPI app, WebSocket, settings handlers
│   ├── agent_core.py            # LLM loop, tool dispatch, routing
│   ├── task_runner.py           # Long-running task loop (Phase 3b)
│   ├── agent_tools/
│   │   ├── __init__.py          # Tool registry
│   │   ├── filesystem.py        # read_file, write_file, list_directory
│   │   ├── capabilities.py      # list_capabilities
│   │   ├── web.py               # search_web, fetch_page
│   │   ├── system_info.py       # get_system_info
│   │   ├── file_analysis.py     # analyze_file
│   │   ├── local_llm.py         # Ollama client, all local LLM tasks
│   │   ├── code_executor.py     # execute_code
│   │   ├── tool_writer.py       # write_tool, reload_tool
│   │   ├── hot_reload.py        # validation + importlib hot-reload
│   │   ├── generated/           # agent-written tools (auto-loaded)
│   │   └── SEARXNG_SETUP.md
│   └── memory/
│       ├── context.py           # History load/save/trim
│       └── embeddings.py        # ChromaDB vector store
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── memory/
│   ├── history.json             # Persisted conversation history
│   ├── current_task.json        # Last task state (Phase 3b)
│   └── user_profile.json        # User skills, goals, constraints (Phase 3g)
├── config.json                  # All settings (runtime-editable via UI)
├── .env                         # ANTHROPIC_API_KEY (never commit)
├── env.example
├── requirements.txt
└── SEARXNG_SETUP.md
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
6. **Windows compatible** – paths, subprocess calls, etc. must work on Windows
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
- **Task runner:** the Phase 3b long-running autonomous execution loop
- **Hot-reload:** re-registering tools at runtime without server restart
- **Generated tools:** agent-written tools in agent_tools/generated/
