"""
agent_tools/process_manager.py  —  Phase 5d: Persistent Process Manager

Allows the agent to start, monitor, and control long-running background
processes (web servers, worker scripts, applications it has built, etc.).

Tools registered:
    start_process(name, command, cwd, env_vars)  — launch a background process
    stop_process(name, force)                    — terminate a running process
    read_process_output(name, max_lines)         — read stdout non-blocking
    send_process_input(name, text)               — write to stdin
    list_processes()                             — list all tracked processes

Security note:
    start_process executes arbitrary shell commands with the current user's
    permissions.  It is marked destructive and requires user approval before
    running.  The cwd argument is validated to stay within the project root
    and outputs directory — it cannot escape to arbitrary filesystem paths.

Windows note:
    Non-blocking stdout reads use a background reader thread + queue because
    select() does not work on Windows pipes.  The thread is started lazily the
    first time read_process_output is called for a given process.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from agent_tools import register_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level process registry
# ---------------------------------------------------------------------------

# Maps process name → Popen instance
_processes: dict[str, subprocess.Popen] = {}

# Maps process name → background stdout reader queue (Windows-compatible)
_output_queues: dict[str, queue.Queue] = {}

# How long (seconds) read_process_output will wait for new lines total
PROCESS_TIMEOUT = 30

# Project root — three levels up from this file
# (backend/agent_tools/process_manager.py → backend/agent_tools → backend → project)
_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
_OUTPUTS_DIR  = _PROJECT_ROOT / "outputs"

# Valid name pattern: letters, digits, hyphens, underscores only
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_name(name: str) -> tuple[bool, str]:
    """Return (True, '') if name is safe, else (False, reason)."""
    if not name:
        return False, "name must not be empty"
    if not _SAFE_NAME_RE.match(name):
        return False, (
            f"name {name!r} contains invalid characters. "
            "Only letters, digits, hyphens, and underscores are allowed."
        )
    return True, ""


def _resolve_cwd(cwd: str) -> tuple[Path | None, str]:
    """
    Resolve cwd relative to the project root.

    Returns (resolved_path, '') on success or (None, error_message) if the
    path escapes outside the allowed roots (project root or outputs/).
    """
    if not cwd:
        return _PROJECT_ROOT, ""

    candidate = (_PROJECT_ROOT / cwd).resolve()

    # Allow paths inside project root or the outputs directory
    allowed = [_PROJECT_ROOT, _OUTPUTS_DIR]
    if any(
        str(candidate).startswith(str(root))
        for root in allowed
    ):
        if not candidate.exists():
            return None, f"cwd does not exist: {candidate}"
        return candidate, ""

    return None, (
        f"cwd {cwd!r} resolves outside the allowed directories. "
        f"Must stay within project root ({_PROJECT_ROOT}) or outputs/ ({_OUTPUTS_DIR})."
    )


def _start_reader_thread(name: str, proc: subprocess.Popen) -> queue.Queue:
    """
    Start a daemon thread that reads proc.stdout and pushes lines to a Queue.

    This is required on Windows because select() does not support file handles
    from subprocess.Popen.  On Unix the same approach works correctly too.
    """
    q: queue.Queue = queue.Queue()
    _output_queues[name] = q

    def _reader():
        try:
            for line in proc.stdout:
                q.put(line.rstrip("\n"))
        except Exception:
            pass
        finally:
            q.put(None)  # sentinel — process ended

    t = threading.Thread(target=_reader, daemon=True, name=f"proc-reader-{name}")
    t.start()
    return q


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def start_process(
    name: str,
    command: str,
    cwd: str = "",
    env_vars: str = "",
) -> dict[str, Any]:
    """
    Launch a shell command as a persistent background process.

    Args:
        name:      Unique identifier for this process (letters, digits, _ -).
        command:   Shell command to execute (e.g. "python main.py --port 8080").
        cwd:       Working directory relative to project root. Defaults to
                   project root. Must stay inside the project or outputs/.
        env_vars:  Extra environment variables as "KEY=VALUE,KEY2=VALUE2".
                   Merged on top of the current process environment.

    Returns:
        {"success": True,  "name": ..., "pid": ..., "command": ...}
        {"success": False, "error": ...}
    """
    # ── Validate name ─────────────────────────────────────────────────────────
    ok, reason = _validate_name(name)
    if not ok:
        return {"success": False, "error": reason}

    # ── Check for existing running process with same name ─────────────────────
    if name in _processes and _processes[name].poll() is None:
        return {
            "success": False,
            "error": (
                f"A process named {name!r} is already running (PID {_processes[name].pid}). "
                "Call stop_process first, or choose a different name."
            ),
        }

    # ── Resolve working directory ─────────────────────────────────────────────
    resolved_cwd, err = _resolve_cwd(cwd)
    if err:
        return {"success": False, "error": err}

    # ── Parse extra env vars ──────────────────────────────────────────────────
    env = os.environ.copy()
    if env_vars.strip():
        for pair in env_vars.split(","):
            pair = pair.strip()
            if "=" not in pair:
                logger.warning(f"[process] Skipping malformed env_var entry: {pair!r}")
                continue
            k, v = pair.split("=", 1)
            env[k.strip()] = v.strip()

    # ── Launch ────────────────────────────────────────────────────────────────
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=str(resolved_cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        return {"success": False, "error": f"Failed to launch process: {e}"}

    _processes[name] = proc
    _start_reader_thread(name, proc)
    logger.info(f"[process] Started '{name}' (PID {proc.pid}): {command!r} in {resolved_cwd}")

    # ── Give it 1.5 s to catch immediate crashes ──────────────────────────────
    await asyncio.sleep(1.5)
    if proc.poll() is not None:
        # Process already exited — read stderr for the error message
        try:
            stderr_out = proc.stderr.read(2000) if proc.stderr else ""
        except Exception:
            stderr_out = "(could not read stderr)"
        del _processes[name]
        _output_queues.pop(name, None)
        return {
            "success":   False,
            "error":     f"Process '{name}' exited immediately (code {proc.returncode}).",
            "stderr":    stderr_out.strip(),
        }

    return {
        "success": True,
        "name":    name,
        "pid":     proc.pid,
        "command": command,
    }


async def stop_process(name: str, force: bool = False) -> dict[str, Any]:
    """
    Terminate a tracked background process.

    Args:
        name:  Process name as given to start_process.
        force: If True, send SIGKILL (process.kill()) instead of SIGTERM
               (process.terminate()).  Use when a process ignores normal
               termination signals.

    Returns:
        {"success": True,  "name": ..., "exit_code": ...}
        {"success": False, "error": ...}
    """
    if name not in _processes:
        return {"success": False, "error": f"No process named {name!r} is tracked."}

    proc = _processes[name]

    if proc.poll() is not None:
        # Already dead — clean up and report
        exit_code = proc.returncode
        del _processes[name]
        _output_queues.pop(name, None)
        return {"success": True, "name": name, "exit_code": exit_code, "note": "Process had already exited."}

    action = "kill" if force else "terminate"
    logger.info(f"[process] {action.capitalize()}ing '{name}' (PID {proc.pid})")

    try:
        if force:
            proc.kill()
        else:
            proc.terminate()

        # Wait up to 5 seconds for the process to exit
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Escalate to kill if terminate timed out
            logger.warning(f"[process] '{name}' did not exit after terminate — killing.")
            proc.kill()
            proc.wait(timeout=5)

    except Exception as e:
        return {"success": False, "error": f"Error stopping '{name}': {e}"}

    exit_code = proc.returncode
    del _processes[name]
    _output_queues.pop(name, None)
    logger.info(f"[process] '{name}' stopped (exit code {exit_code}).")

    return {"success": True, "name": name, "exit_code": exit_code}


async def read_process_output(name: str, max_lines: int = 50) -> dict[str, Any]:
    """
    Read buffered stdout lines from a running (or recently exited) process.

    Uses a background thread queue — safe on Windows where select() cannot
    poll subprocess stdout.  Never blocks indefinitely; drains whatever is
    available at call time up to max_lines.

    Args:
        name:       Process name.
        max_lines:  Maximum number of lines to return (default 50, max 200).

    Returns:
        {"success": True, "name": ..., "lines": [...], "is_running": bool}
        {"success": False, "error": ...}
    """
    if name not in _processes:
        return {"success": False, "error": f"No process named {name!r} is tracked."}

    proc = _processes[name]
    max_lines = min(max_lines, 200)

    # Ensure the reader thread is running (idempotent — only starts once)
    if name not in _output_queues:
        _start_reader_thread(name, proc)

    q = _output_queues[name]
    lines: list[str] = []

    # Drain the queue non-blocking until empty or limit reached
    while len(lines) < max_lines:
        try:
            item = q.get_nowait()
            if item is None:
                # Sentinel — process ended; re-put so subsequent calls see it
                q.put(None)
                break
            lines.append(item)
        except queue.Empty:
            break

    is_running = proc.poll() is None
    return {
        "success":    True,
        "name":       name,
        "lines":      lines,
        "is_running": is_running,
    }


async def send_process_input(name: str, text: str) -> dict[str, Any]:
    """
    Write a line of text to a process's stdin.

    Useful for interactive CLI tools that read commands from stdin (e.g.
    Python REPLs, database shells, menu-driven scripts).

    Args:
        name: Process name.
        text: Text to send. A newline is appended automatically.

    Returns:
        {"success": True, "name": ..., "sent": ...}
        {"success": False, "error": ...}
    """
    if name not in _processes:
        return {"success": False, "error": f"No process named {name!r} is tracked."}

    proc = _processes[name]

    if proc.poll() is not None:
        return {"success": False, "error": f"Process '{name}' has already exited."}

    if proc.stdin is None:
        return {"success": False, "error": f"Process '{name}' has no stdin pipe."}

    try:
        proc.stdin.write(text + "\n")
        proc.stdin.flush()
        logger.info(f"[process] Sent input to '{name}': {text!r}")
    except Exception as e:
        return {"success": False, "error": f"Could not write to stdin of '{name}': {e}"}

    return {"success": True, "name": name, "sent": text}


async def list_processes() -> dict[str, Any]:
    """
    List all tracked processes and their current status.

    Auto-cleans entries for processes that have already exited so the
    registry stays tidy.

    Returns:
        {
            "success": True,
            "processes": [
                {"name": ..., "pid": ..., "is_running": bool, "exit_code": int | None}
            ]
        }
    """
    result = []
    to_remove = []

    for name, proc in _processes.items():
        exit_code = proc.poll()
        is_running = exit_code is None
        result.append({
            "name":       name,
            "pid":        proc.pid,
            "is_running": is_running,
            "exit_code":  exit_code,
        })
        if not is_running:
            to_remove.append(name)

    # Clean up dead processes from registry
    for name in to_remove:
        del _processes[name]
        _output_queues.pop(name, None)
        logger.debug(f"[process] Auto-removed exited process '{name}' from registry.")

    return {"success": True, "processes": result}


# ---------------------------------------------------------------------------
# Cleanup — called by main.py on server shutdown
# ---------------------------------------------------------------------------

def cleanup_all_processes() -> None:
    """
    Kill all running tracked processes.  Called from main.py's shutdown
    handler so background processes don't become orphans when the server exits.
    """
    for name, proc in list(_processes.items()):
        if proc.poll() is None:
            logger.info(f"[process] Shutdown: killing '{name}' (PID {proc.pid})")
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception as e:
                logger.warning(f"[process] Could not kill '{name}': {e}")
    _processes.clear()
    _output_queues.clear()
    logger.info("[process] All tracked processes cleaned up.")


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register_process_tools() -> None:
    """Register all process management tools. Call once from main.py at startup."""

    register_tool(
        name="start_process",
        description=(
            "Launch a shell command as a persistent background process. "
            "Use this to start web servers, workers, or apps the agent has built. "
            "The process keeps running until stop_process is called. "
            "name must be unique (letters, digits, _ -). "
            "cwd is relative to the project root (e.g. 'outputs/my_app'). "
            "env_vars format: 'KEY=VALUE,KEY2=VALUE2'. "
            "Returns: {success, name, pid, command}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type":        "string",
                    "description": "Unique process identifier (letters, digits, hyphens, underscores).",
                },
                "command": {
                    "type":        "string",
                    "description": "Shell command to run (e.g. 'python main.py --port 8080').",
                },
                "cwd": {
                    "type":        "string",
                    "description": "Working directory relative to project root. Default: project root.",
                },
                "env_vars": {
                    "type":        "string",
                    "description": "Extra env vars as 'KEY=VALUE,KEY2=VALUE2'. Optional.",
                },
            },
            "required": ["name", "command"],
        },
        handler=start_process,
        is_destructive=True,   # executes arbitrary shell commands — requires user approval
    )

    register_tool(
        name="stop_process",
        description=(
            "Terminate a background process started with start_process. "
            "Use force=true to send SIGKILL if the process ignores normal termination. "
            "Returns: {success, name, exit_code}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type":        "string",
                    "description": "Process name as given to start_process.",
                },
                "force": {
                    "type":        "boolean",
                    "description": "If true, send SIGKILL instead of SIGTERM. Default: false.",
                },
            },
            "required": ["name"],
        },
        handler=stop_process,
        is_destructive=True,   # terminates a running process
    )

    register_tool(
        name="read_process_output",
        description=(
            "Read buffered stdout lines from a background process. "
            "Non-blocking — returns whatever lines are available up to max_lines. "
            "Returns: {success, name, lines, is_running}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type":        "string",
                    "description": "Process name.",
                },
                "max_lines": {
                    "type":        "integer",
                    "description": "Maximum lines to return (default 50, max 200).",
                },
            },
            "required": ["name"],
        },
        handler=read_process_output,
        is_destructive=False,
    )

    register_tool(
        name="send_process_input",
        description=(
            "Write a line of text to a background process's stdin. "
            "Useful for interactive CLI tools or REPLs. "
            "A newline is appended automatically. "
            "Returns: {success, name, sent}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type":        "string",
                    "description": "Process name.",
                },
                "text": {
                    "type":        "string",
                    "description": "Text to send to stdin (newline appended automatically).",
                },
            },
            "required": ["name", "text"],
        },
        handler=send_process_input,
        is_destructive=True,   # sends commands to a running process
    )

    register_tool(
        name="list_processes",
        description=(
            "List all tracked background processes and their current status. "
            "Auto-removes entries for processes that have already exited. "
            "Returns: {success, processes: [{name, pid, is_running, exit_code}]}."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=list_processes,
        is_destructive=False,
    )

    logger.info(
        "[startup] Registered tools: process_manager "
        "(start_process, stop_process, read_process_output, send_process_input, list_processes)"
    )
