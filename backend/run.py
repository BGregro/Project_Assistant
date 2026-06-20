"""
run.py  —  Development server launcher

Replaces `uvicorn main:app --reload` on Windows because uvicorn's WatchFiles
reloader spawns a fresh worker process that ignores any event loop policy set
in the parent.  We work around this by:

  1. Setting WindowsProactorEventLoopPolicy globally (affects this process and
     any subprocess that inherits the environment).
  2. Passing loop="none" to uvicorn so it does NOT create its own loop before
     our app code runs.
  3. Explicitly creating and installing a ProactorEventLoop before handing
     control to uvicorn.Server.serve().

Playwright (and asyncio subprocesses in general, including execute_code) need
ProactorEventLoop on Windows to spawn child processes.  SelectorEventLoop —
uvicorn's default on Windows — raises NotImplementedError for subprocess calls.

Usage (from backend/ with venv active):
    python run.py

iGPU NOTE — ipex-llm Ollama .env settings:
    The ipex-llm Ollama folder (IPEX_OLLAMA_DIR below) should contain a .env
    file with these values for optimal Arc iGPU performance:

        OLLAMA_NUM_CTX=8192
        OLLAMA_KEEP_ALIVE=-1
        OLLAMA_HOST=127.0.0.1:11434

    This file is written automatically by ensure_ollama_running() if missing.
"""

import sys
import asyncio
import subprocess
import time
import pathlib

import httpx

# ---------------------------------------------------------------------------
# ipex-llm Ollama auto-start (Improvement 1 & 2)
# ---------------------------------------------------------------------------

# Path to the ipex-llm Ollama folder — update this if your installation moved.
IPEX_OLLAMA_DIR = pathlib.Path(
    r"C:\Users\bodag\Downloads\ollama-ipex-llm-2.3.0b20250612-win"
)
OLLAMA_URL = "http://localhost:11434"

# Optimal .env content for Arc iGPU acceleration via ipex-llm Ollama.
# Written to IPEX_OLLAMA_DIR/.env if the file is absent or outdated.
_DESIRED_ENV = (
    "OLLAMA_NUM_CTX=8192\n"
    "OLLAMA_KEEP_ALIVE=-1\n"
    "OLLAMA_HOST=127.0.0.1:11434\n"
)


def _ensure_ipex_env() -> None:
    """
    Write the optimal .env file for ipex-llm Ollama if it is missing or
    has different content.  This configures:
      - OLLAMA_NUM_CTX=8192   — 8k context window for qwen3:14b on 32 GB RAM
      - OLLAMA_KEEP_ALIVE=-1  — keep model resident for the Ollama process lifetime
      - OLLAMA_HOST=...       — bind to loopback only (security)
    """
    env_path = IPEX_OLLAMA_DIR / ".env"
    try:
        if not env_path.exists() or env_path.read_text(encoding="utf-8") != _DESIRED_ENV:
            env_path.write_text(_DESIRED_ENV, encoding="utf-8")
            print("[run] Written ipex-llm .env with optimal iGPU settings")
        else:
            print("[run] ipex-llm .env already correct — no changes needed")
    except Exception as exc:
        print(f"[run] WARNING: Could not write ipex-llm .env: {exc}")


def ensure_ollama_running() -> None:
    """
    Start ipex-llm Ollama on the Arc iGPU if it isn't already running.
    Called automatically at agent startup — no manual step needed.
    Falls back gracefully if the ipex-llm folder isn't found.

    Sequence:
      1. Check if any Ollama instance is already responsive at OLLAMA_URL.
         If yes, skip launch and return immediately.
      2. Verify (and write if needed) the .env file for optimal iGPU settings.
      3. Launch start-ollama.bat in a new console window.
      4. Poll /api/tags for up to 30 seconds; log progress every second.
    """
    # Step 1: check if already running
    try:
        httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
        print("[run] Ollama already running — skipping auto-start")
        return
    except Exception:
        pass  # Not running — proceed with launch

    # Step 2: verify ipex-llm folder exists
    bat_path = IPEX_OLLAMA_DIR / "start-ollama.bat"
    if not bat_path.exists():
        print(f"[run] WARNING: ipex-llm not found at {IPEX_OLLAMA_DIR}")
        print("[run] Continuing without iGPU acceleration — start Ollama manually if needed")
        return

    # Step 2b: write optimal .env before starting the server
    _ensure_ipex_env()

    # Step 3: launch in a new console so the user can see Ollama's output
    print(f"[run] Starting ipex-llm Ollama on Arc iGPU from {IPEX_OLLAMA_DIR} ...")
    subprocess.Popen(
        ["cmd", "/c", "start-ollama.bat"],
        cwd=str(IPEX_OLLAMA_DIR),
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )

    # Step 4: wait up to 30 seconds for Ollama to be ready
    for i in range(30):
        try:
            httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
            print(f"[run] Ollama ready after {i + 1}s")
            return
        except Exception:
            time.sleep(1)

    print("[run] WARNING: Ollama didn't respond in 30 s — continuing anyway")


# ---------------------------------------------------------------------------
# Server launch
# ---------------------------------------------------------------------------

# Step 1: set the policy process-wide so any new loop created anywhere uses Proactor
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import uvicorn


def main() -> None:
    # Auto-start ipex-llm Ollama on Arc iGPU before the FastAPI server comes up.
    # This is a synchronous call because we haven't started the event loop yet.
    ensure_ollama_running()

    config = uvicorn.Config(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["./"],
        # "none" tells uvicorn not to touch the event loop — we manage it below
        loop="none",
    )
    server = uvicorn.Server(config)

    if sys.platform == "win32":
        # Step 2: create a ProactorEventLoop explicitly and install it
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()
    else:
        # Non-Windows: let uvicorn handle the loop normally
        asyncio.run(server.serve())


if __name__ == "__main__":
    main()
