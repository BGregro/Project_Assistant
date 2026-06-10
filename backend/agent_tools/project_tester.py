"""
project_tester.py  —  Phase 4b/4c: Integration Testing
                    —  Phase 4d: Project Memory (inline)

Registers one tool:

  run_project_test  — execute the project's test_command (from scaffold.json)
                      inside the project directory, capture stdout/stderr,
                      and record the outcome in progress.json.

Phase 4d is wired inline: when a test passes, the project is automatically
logged to long-term memory via long_term.log_project().  If there have been
3+ failed attempts, the failure patterns are included as lessons so future
projects can benefit from what went wrong.

Security note: this tool executes arbitrary code in the project directory and
is therefore marked is_destructive=True.  The user must confirm before it runs.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent_tools import register_tool

logger = logging.getLogger(__name__)

# Project root — backend/agent_tools/ → backend/ → project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_OUTPUTS_DIR  = _PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict | None:
    """Read a JSON file, return None on any failure."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[project_tester] Could not read {path.name}: {e}")
        return None


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[project_tester] Could not write {path.name}: {e}")


def _truncate(text: str, limit: int = 3000) -> str:
    """Truncate a string to `limit` chars, appending a note if cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated, {len(text) - limit} chars omitted]"


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _build_command(command: str) -> list[str]:
    """
    On Windows, wrap python/shell commands in  cmd /c  so they resolve
    correctly through the system PATH.  On POSIX, split naively via shell=False
    is fine for simple commands.

    We always return a list so subprocess.run receives an explicit argv rather
    than a shell string — this avoids shell-injection issues.
    """
    if _is_windows():
        # cmd /c handles PATH lookup and built-ins (pip, python, pytest, etc.)
        return ["cmd", "/c"] + command.split()
    # POSIX: simple whitespace split is sufficient for the commands we generate
    return command.split()


def _log_to_long_term(
    scaffold: dict,
    outcome: str,
    failed_attempts: int,
    progress: dict,
) -> None:
    """
    Save a completed project to long-term memory (Phase 4d).
    Imported lazily so a missing long_term module never breaks the tester.
    """
    try:
        from memory import long_term  # local import to avoid circular deps

        # Collect failure patterns as lessons (if there were retries)
        lessons = ""
        if failed_attempts >= 3:
            failed_tests = [
                t for t in progress.get("test_history", [])
                if not t.get("passed")
            ]
            if failed_tests:
                snippets = "; ".join(
                    (t.get("stderr") or t.get("stdout") or "unknown error")[:120]
                    for t in failed_tests[-3:]
                )
                lessons = (
                    f"Required {failed_attempts} test attempts. "
                    f"Recurring errors: {snippets}"
                )

        long_term.log_project(
            name=scaffold.get("name", "unknown"),
            description=scaffold.get("description", ""),
            structure=[f["path"] for f in scaffold.get("structure", [])],
            dependencies=scaffold.get("dependencies", []),
            entry_point=scaffold.get("entry_point", ""),
            outcome=outcome,
            lessons=lessons,
        )
        logger.info(f"[project_tester] Project '{scaffold.get('name')}' logged to long-term memory.")
    except Exception as e:
        # Phase 4d failure is non-fatal — the test result is still returned
        logger.warning(f"[project_tester] Could not log project to long-term memory: {e}")


# ---------------------------------------------------------------------------
# Tool: run_project_test
# ---------------------------------------------------------------------------

async def run_project_test(
    project_name: str,
    command: str = "",
    timeout: int = 30,
) -> dict:
    """
    Run the project's test/entry-point command inside its output directory.

    The command defaults to the test_command field in scaffold.json.
    The working directory is set to outputs/{project_name}/ so relative
    imports and file references resolve correctly.

    Phase 4d: on a passing test, the project is automatically logged to
    long-term memory.  On 3+ failures, failure patterns are included.

    Args:
        project_name: Project identifier (matches outputs/{project_name}/).
        command:      Override the scaffold test_command if provided.
        timeout:      Max seconds before the subprocess is killed (default 30).

    Returns:
        {
            "success":     bool,   # False on timeout/exception, True otherwise
            "exit_code":   int,
            "stdout":      str,    # truncated to 3000 chars
            "stderr":      str,    # truncated to 3000 chars
            "passed":      bool,   # True iff exit_code == 0
            "project_dir": str,
        }
        On timeout:
        {
            "success": False,
            "error":   "Test timed out after {timeout}s",
        }
    """
    project_dir  = _OUTPUTS_DIR / project_name.strip()
    scaffold_path = project_dir / "scaffold.json"
    progress_path = project_dir / "progress.json"

    # ── Resolve command ───────────────────────────────────────────────────────
    scaffold = _load_json(scaffold_path)
    if not command:
        if scaffold is None:
            return {
                "success": False,
                "error":   f"No scaffold found for '{project_name}' and no command supplied.",
            }
        command = scaffold.get("test_command", "python main.py")
        if not command:
            command = "python main.py"

    # ── Strip outputs/{project_name}/ prefix if agent passed a full path ──────
    # The subprocess runs with cwd=project_dir, so only the filename or a short
    # project-relative path is needed. Strip any leading outputs/{name}/ prefix
    # the agent may have accidentally included.
    for prefix in [f"outputs/{project_name}/", f"outputs\\{project_name}\\"]:
        if command.startswith(prefix):
            command = command[len(prefix):]
            logger.debug(f"[project_tester] Stripped path prefix \u2192 {command!r}")
            break

    # ── Load / initialise progress ────────────────────────────────────────────
    progress: dict = _load_json(progress_path) or {
        "project_name":    project_name,
        "created_at":      _now_iso(),
        "completed_files": [],
        "last_test":       None,
        "test_attempts":   0,
        "test_history":    [],
    }
    progress.setdefault("test_attempts", 0)
    progress.setdefault("test_history",  [])

    logger.info(
        f"[project_tester] Running test for '{project_name}': {command!r} "
        f"(cwd={project_dir}, timeout={timeout}s)"
    )

    # ── Run subprocess ────────────────────────────────────────────────────────
    cmd_list = _build_command(command)

    try:
        result = subprocess.run(
            cmd_list,
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"[project_tester] Test timed out after {timeout}s for '{project_name}'.")
        return {
            "success": False,
            "error":   f"Test timed out after {timeout}s",
        }
    except Exception as e:
        logger.error(f"[project_tester] Subprocess error for '{project_name}': {e}")
        return {
            "success": False,
            "error":   f"Failed to run command: {e}",
        }

    exit_code = result.returncode
    stdout    = _truncate(result.stdout or "")
    stderr    = _truncate(result.stderr or "")
    passed    = exit_code == 0

    # ── Update progress ───────────────────────────────────────────────────────
    test_record = {
        "command":   command,
        "exit_code": exit_code,
        "stdout":    stdout[:500],   # keep brief copies in history to save space
        "stderr":    stderr[:500],
        "timestamp": _now_iso(),
        "passed":    passed,
    }

    progress["last_test"]     = test_record
    progress["test_attempts"] = progress["test_attempts"] + 1
    progress["test_history"].append(test_record)
    # Keep only the last 20 test records to avoid unbounded growth
    progress["test_history"] = progress["test_history"][-20:]

    _save_json(progress_path, progress)

    logger.info(
        f"[project_tester] Test result for '{project_name}': "
        f"exit={exit_code}, passed={passed}, attempt={progress['test_attempts']}"
    )

    # ── Phase 4d: log to long-term memory on success ──────────────────────────
    if passed and scaffold:
        _log_to_long_term(
            scaffold=scaffold,
            outcome="success",
            failed_attempts=progress["test_attempts"] - 1,  # -1 for the passing run
            progress=progress,
        )
        # ── Phase 5a: flag that this project hasn't been pushed to GitHub yet ──
        # The agent can check this flag and offer to push the project.
        # Only set it if not already True (don't reset after a re-test).
        if not progress.get("github_pushed"):
            progress["github_pushed"] = False
            _save_json(progress_path, progress)
            logger.debug(
                f"[project_tester] Set github_pushed=false for '{project_name}' — "
                "agent can offer to push to GitHub."
            )

    return {
        "success":     True,
        "exit_code":   exit_code,
        "stdout":      stdout,
        "stderr":      stderr,
        "passed":      passed,
        "project_dir": str(project_dir),
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_project_tester_tools() -> None:
    """Register run_project_test with the live tool registry."""

    register_tool(
        name="run_project_test",
        description=(
            "Run the project's test command (from scaffold.json) inside its output directory. "
            "Use this after all files in implementation_order have been written. "
            "Captures stdout and stderr; returns passed=True if exit code is 0. "
            "If the test fails, read stderr to identify which file to fix, then call again. "
            "On success, the project is automatically saved to long-term memory."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_name": {
                    "type":        "string",
                    "description": "Project identifier matching the outputs/{project_name}/ directory.",
                },
                "command": {
                    "type":        "string",
                    "description": (
                        "Command to run (e.g. 'python main.py', 'pytest'). "
                        "Defaults to the test_command from scaffold.json."
                    ),
                },
                "timeout": {
                    "type":        "integer",
                    "description": "Max seconds before the subprocess is killed. Default: 30.",
                },
            },
            "required": ["project_name"],
        },
        handler=run_project_test,
        is_destructive=True,   # executes code — requires user confirmation
    )

    logger.info("[startup] Registered tool: run_project_test")
