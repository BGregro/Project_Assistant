# Personal AI Agent – Project Instructions

## Project Goal
Build a modular, locally-run personal AI agent using Claude API as the LLM "brain",
with an extensible tool system, Python backend, and web UI.

## Developer Context
- **Developer:** 3rd-year Computer Engineering student
- **OS:** Windows (WSL2 available if needed)
- **Hardware:** Lenovo ThinkPad E14 Gen7 (Intel Core Ultra 5, Intel Arc 140V)
- **LLM:** Claude API (Anthropic) – claude-haiku-3-5 by default (token efficiency),
  claude-sonnet-4-5 for complex tasks
- **Local LLM:** Ollama (qwen2.5:7b) for offline fallback, prompt optimization,
  intent classification, and context compression
- **Goal:** Learning + practical utility, incrementally extensible system

## Tech Stack
- **Backend:** Python (FastAPI)
- **Frontend:** HTML/CSS/JS (chat UI), migratable to React later
- **LLM Integration:** Anthropic Python SDK, tool use (function calling)
- **Local LLM:** Ollama (http://localhost:11434), used as a secondary tier
- **Code Execution:** Sandbox (subprocess + restricted permissions)
- **Package Manager:** pip + virtualenv

## LLM Two-Tier Architecture
The agent uses two LLM tiers with distinct responsibilities:

```
User message
     │
     ▼
[Local LLM - Ollama]  ──  prompt optimization, intent classification,
     │                     context compression, offline fallback
     │
     └──  complex tasks, tool use, final answers  ──►  [Claude API]
```

**Local LLM (Ollama / qwen2.5:7b) is used for:**
- Rewriting raw user messages into clean, precise prompts before sending to Claude API
- Classifying intent: does this need a tool call or is it a simple question?
- Summarizing long tool outputs before they enter the Claude context window
- Offline fallback when Claude API is unavailable

**Claude API is used for:**
- All tool use and agentic reasoning
- Complex multi-step tasks
- Final answers to the user

**Config controls:**
- `use_prompt_optimizer`: toggle local prompt rewriting on/off
- `local_fallback`: fall back to local LLM if Claude API is unreachable
- `primary`: default Claude model (claude-haiku-3-5)
- `complex`: model for complex tasks (claude-sonnet-4-5)
- `local`: local Ollama model (qwen2.5:7b)

## Development Phases

### Phase 1 – Base Agent (CURRENT)
- FastAPI backend structure
- Chat UI (simple, functional)
- Claude API connection with tool use
- Filesystem tool (read/write/list)
- Basic permission layer
- Ollama client + prompt optimizer (local_llm.py)

### Phase 2 – Extended Tools
- Code execution in sandbox
- Web scraping and file downloads
- External API calls (web search, etc.)
- Intent classifier (local LLM routes simple vs. complex queries)
- Context compression (local LLM summarizes long tool outputs)

### Phase 3 – Self-Improvement
- Agent-written and registered new tools
- Long-term memory (vector DB or simple JSON)
- Overlay UI (Windows)

## Project Structure (target)
```
agent/
├── backend/
│   ├── main.py              # FastAPI app, WebSocket
│   ├── agent_core.py        # LLM loop, tool dispatch
│   ├── tools/
│   │   ├── __init__.py      # Tool registry
│   │   ├── filesystem.py    # File operations
│   │   ├── local_llm.py     # Ollama client, prompt optimizer
│   │   ├── code_executor.py # Code execution (Phase 2)
│   │   └── web.py           # Web/download (Phase 2)
│   └── memory/
│       └── context.py       # Context management
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── config.json              # API keys, settings, model selection
└── requirements.txt
```

## Coding Principles
1. **Always modular** – each tool in its own file; core must not depend on tool implementations
2. **Token-efficient** – use local LLM for preprocessing; system prompts short and informative;
   never repeat context redundantly in the Claude API calls
3. **Incremental complexity** – never skip a phase; stabilize before extending
4. **Security first** – permission check before every file/code/network operation;
   always ask user confirmation for destructive actions
5. **Commented code** – developer is learning, explain non-trivial parts
6. **Windows compatible** – paths, subprocess calls, etc. must work on Windows

## How to Help
- **Always provide every created or edited file as a downloadable file** — never just
  show code in a code block when a file is being created or modified
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
