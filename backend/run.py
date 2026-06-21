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
    r"C:\\Users\\bodag\\OneDrive\\PROJECTS\\Project_Assistant\\ollama-ipex-llm"
)
OLLAMA_URL = "http://localhost:11434"

# Optimal .env content for Arc iGPU acceleration via ipex-llm Ollama.
# Written to IPEX_OLLAMA_DIR/.env if the file is absent or outdated.
_DESIRED_ENV = (
    "OLLAMA_NUM_CTX=8192\n"
    "OLLAMA_KEEP_ALIVE=-1\n"
    "OLLAMA_HOST=127.0.0.1:11434\n"
    "OLLAMA_MAX_LOADED_MODELS=2\n"
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


def _is_igpu_accelerated() -> bool:
    """
    Check if the running Ollama instance is the ipex-llm iGPU build.

    Detection strategy (in order of reliability):
      1. Process name check: ipex-llm Ollama spawns 'ollama-lib.exe' as its
         GPU backend. Standard Ollama never has this process. This is the most
         reliable indicator and works even when no models are loaded.
      2. /api/ps VRAM check: if any loaded model reports size_vram > 0, GPU
         acceleration is confirmed (fallback for non-ipex GPU builds).

    Returns True if ipex-llm / iGPU acceleration is detected, False otherwise.
    """
    # Primary check: look for ollama-lib.exe in the process list.
    # This process is exclusive to the ipex-llm Ollama build.
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq ollama-lib.exe", "/NH"],
            capture_output=True, text=True, timeout=5
        )
        if "ollama-lib.exe" in result.stdout:
            print("[run] ipex-llm detected via ollama-lib.exe process")
            return True
    except Exception:
        pass  # tasklist unavailable — fall through to API check

    # Fallback: query /api/ps for loaded models with VRAM usage
    try:
        ps = httpx.get(f"{OLLAMA_URL}/api/ps", timeout=3.0).json()
        models = ps.get("models", [])
        if models and any(m.get("size_vram", 0) > 0 for m in models):
            return True
    except Exception:
        pass

    return False


def _stop_ollama() -> bool:
    """
    Kill any running Ollama process and wait until port 11434 is released.

    Kills both 'ollama.exe' (the server) and 'ollama app.exe' (the tray app
    that would otherwise restart the server). Falls back to finding and killing
    whatever process holds port 11434 by PID via netstat.

    Returns True if the port is confirmed free, False if still occupied after
    the maximum wait time (caller should NOT re-launch in that case).
    """
    # Kill the server process
    try:
        result = subprocess.run(
            ["taskkill", "/F", "/IM", "ollama.exe"],
            capture_output=True, timeout=10
        )
        print("[run] Sent kill signal to ollama.exe")
    except Exception as e:
        print(f"[run] Could not kill ollama.exe (non-fatal): {e}")

    # Also kill the tray app — it will restart the server if left running
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", "ollama app.exe"],
            capture_output=True, timeout=10
        )
        print("[run] Sent kill signal to 'ollama app.exe' (tray)")
    except Exception as e:
        pass  # Not present — that's fine

    print("[run] Waiting for port 11434 to be released...")

    # Poll every second for up to 20 seconds
    for i in range(20):
        time.sleep(1)
        try:
            httpx.get(f"{OLLAMA_URL}/api/tags", timeout=1.0)
            # Still responding — try netstat kill as fallback after 5s
            if i == 4:
                _kill_port_11434_by_pid()
        except Exception:
            print(f"[run] Port 11434 free after {i + 1}s")
            return True  # Port released — safe to launch

    print("[run] WARNING: Port 11434 still occupied after 20s — will not re-launch")
    return False


def _kill_port_11434_by_pid() -> None:
    """
    Use netstat to find the PID of whatever process is listening on port 11434
    and kill it directly. This handles cases where the process name is unknown
    or where a service manager has restarted it under a different name.
    """
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            # Look for LISTENING on port 11434
            if ":11434" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit() and pid != "0":
                    print(f"[run] Killing PID {pid} holding port 11434 (netstat fallback)")
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        capture_output=True, timeout=5
                    )
                    return
    except Exception as e:
        print(f"[run] netstat fallback failed (non-fatal): {e}")


def ensure_ollama_running() -> None:
    """
    Ensure ipex-llm Ollama is running on the Arc iGPU.

    If standard Ollama (CPU) is already running, stops it and waits for the
    port to be fully released before launching the ipex-llm version. This
    prevents the bind error that occurs when ipex-llm tries to claim a port
    that ollama.exe hasn't finished releasing yet.

    Sequence:
      1. Bail early if the ipex-llm bat file doesn't exist.
      2. Check if any Ollama instance is responsive at OLLAMA_URL.
         - Not running → skip to launch.
         - Running + iGPU confirmed → already correct, return.
         - Running + CPU/unknown → stop it and wait for port to free.
           If port doesn't free in 10s → abort (don't double-bind).
      3. Write optimal .env and launch start-ollama.bat.
      4. Poll /api/tags for up to 30 seconds.
    """
    bat_path = IPEX_OLLAMA_DIR / "start-ollama.bat"
    if not bat_path.exists():
        print(f"[run] WARNING: ipex-llm not found at {IPEX_OLLAMA_DIR}")
        print("[run] Continuing without iGPU — start Ollama manually if needed")
        return

    # Step 1: check if Ollama is already running
    already_running = False
    try:
        httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
        already_running = True
    except Exception:
        pass

    if already_running:
        if _is_igpu_accelerated():
            print("[run] ipex-llm Ollama already running on iGPU — skipping start")
            return
        else:
            print("[run] Standard Ollama (CPU) detected — stopping it to start ipex-llm version")
            port_free = _stop_ollama()
            if not port_free:
                # Port still occupied; ipex-llm would get a bind error.
                # The agent will still work, just possibly on CPU.
                print("[run] Skipping ipex-llm launch to avoid port conflict")
                return
            # Port is confirmed free — fall through to launch below

    # Write optimal .env before starting
    _ensure_ipex_env()

    print(f"[run] Starting ipex-llm Ollama on Arc iGPU from {IPEX_OLLAMA_DIR} ...")
    subprocess.Popen(
        ["cmd", "/c", "start-ollama.bat"],
        cwd=str(IPEX_OLLAMA_DIR),
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )

    # Wait up to 30 seconds for Ollama to be ready
    for i in range(30):
        try:
            httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
            print(f"[run] ipex-llm Ollama ready after {i + 1}s")
            return
        except Exception:
            time.sleep(1)

    print("[run] WARNING: Ollama didn't respond in 30s — continuing anyway")


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
