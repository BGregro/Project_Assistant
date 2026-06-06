"""
code_executor.py  —  Code Execution Tool (Phase 3a)

Registers one tool:
    execute_code(code, language)  —  runs a Python or Bash snippet in a subprocess
                                     and returns stdout, stderr, and exit code.

⚠️  SECURITY NOTICE:
    This tool runs code DIRECTLY on the host machine with the same permissions
    as the Python process running the agent.  There is NO network restriction,
    NO filesystem sandboxing, and NO resource capping beyond the configurable
    timeout.  The permission layer marks this tool as destructive so the user
    is prompted to approve each execution before it runs.  Never allow the
    agent to call this tool without that approval step.
"""

import json
import logging
import subprocess
import sys
import tempfile
import os
from pathlib import Path

from agent_tools import register_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _get_timeout() -> int:
    """
    Read tools.code.timeout_seconds from config.json.
    Falls back to 30 seconds if the key is absent or the file cannot be read.
    The config path is resolved relative to this file's location so it works
    regardless of the current working directory.
    """
    try:
        config_path = Path(__file__).parent.parent.parent / "config.json"
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg.get("tools", {}).get("code", {}).get("timeout_seconds", 30))
    except Exception:
        return 30


# ---------------------------------------------------------------------------
# Core executor
# ---------------------------------------------------------------------------

async def execute_code(code: str, language: str = "python") -> dict:
    """
    Execute a code snippet in a subprocess and return structured output.

    Args:
        code:     The source code to run.
        language: "python" or "bash".  Any other value returns an error.

    Returns:
        {
            "success":   bool,   # True if exit_code == 0
            "stdout":    str,    # captured standard output (may be empty)
            "stderr":    str,    # captured standard error  (may be empty)
            "exit_code": int,    # process exit code; -1 on timeout
            "language":  str,    # echoes back the requested language
        }
    """
    timeout = _get_timeout()
    lang    = language.lower().strip()

    # ------------------------------------------------------------------
    # Validate language
    # ------------------------------------------------------------------
    if lang not in ("python", "bash"):
        return {
            "success":   False,
            "stdout":    "",
            "stderr":    f"Unsupported language: '{language}'. Supported: python, bash.",
            "exit_code": -1,
            "language":  language,
        }

    logger.info(f"[code_executor] Executing {lang} snippet ({len(code)} chars, timeout={timeout}s)")

    # ------------------------------------------------------------------
    # Build the subprocess command
    # ------------------------------------------------------------------

    tmp_path = None  # used for Python only; cleaned up in finally

    try:
        if lang == "python":
            # Write code to a temp .py file so Python can import __file__,
            # produce proper tracebacks, etc.
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            cmd = [sys.executable, tmp_path]

        else:  # bash
            # On Windows: route through cmd /c so the shell built-ins work.
            # On Linux/macOS: use bash -c.
            if sys.platform == "win32":
                cmd = ["cmd", "/c", code]
            else:
                cmd = ["bash", "-c", code]

        # ------------------------------------------------------------------
        # Run the subprocess
        # ------------------------------------------------------------------
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        logger.info(
            f"[code_executor] Finished — exit_code={proc.returncode}, "
            f"stdout={len(proc.stdout)} chars, stderr={len(proc.stderr)} chars"
        )

        return {
            "success":   proc.returncode == 0,
            "stdout":    proc.stdout,
            "stderr":    proc.stderr,
            "exit_code": proc.returncode,
            "language":  lang,
        }

    except subprocess.TimeoutExpired:
        logger.warning(f"[code_executor] Timed out after {timeout}s")
        return {
            "success":   False,
            "stdout":    "",
            "stderr":    f"Execution timed out after {timeout} seconds.",
            "exit_code": -1,
            "language":  lang,
        }

    except FileNotFoundError as e:
        # Interpreter / shell binary not found on PATH
        logger.error(f"[code_executor] Interpreter not found: {e}")
        return {
            "success":   False,
            "stdout":    "",
            "stderr":    f"Interpreter not found: {e}",
            "exit_code": -1,
            "language":  lang,
        }

    except Exception as e:
        logger.exception("[code_executor] Unexpected error during execution")
        return {
            "success":   False,
            "stdout":    "",
            "stderr":    str(e),
            "exit_code": -1,
            "language":  lang,
        }

    finally:
        # Clean up the temp Python file (if one was created)
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass  # non-fatal; OS will clean temp dir eventually


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_code_executor_tools() -> None:
    """
    Register execute_code with the tool registry.

    The tool is marked destructive=True so the permission layer will ask the
    user to confirm before the agent runs any code.
    """
    register_tool(
        name="execute_code",
        description=(
            "Execute a code snippet in a subprocess and return the output. "
            "Supported languages: 'python' and 'bash'. "
            "Returns stdout, stderr, exit_code, and a success flag. "
            "Use this to compute results, test logic, process files, or "
            "verify anything that can be confirmed by running code. "
            "Always read stdout AND stderr before judging success. "
            "If the run fails, fix the code and try again."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The source code to execute.",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "bash"],
                    "description": (
                        "The language to run. 'python' uses the current interpreter; "
                        "'bash' uses bash on Linux/macOS or cmd on Windows."
                    ),
                },
            },
            "required": ["code"],
        },
        handler=execute_code,
        # is_destructive=True triggers the permission modal before every execution.
        # The user must explicitly approve before any code reaches the host shell.
        is_destructive=True,
    )
    logger.info("[code_executor] Registered tool: execute_code (destructive)")
