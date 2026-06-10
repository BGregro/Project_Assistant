"""
code_executor.py  —  Code Execution Tool (Phase 3a / Phase 4.5)

Registers two tools:
    execute_code(code, language)  —  runs a Python or Bash snippet in a subprocess
                                     and returns stdout, stderr, and exit code.
                                     Phase 4.5: streams stdout line-by-line to the
                                     frontend via _send_event_callback so long-running
                                     scripts feel live rather than frozen.

    install_package(package, version)  —  pip-installs a package into the current
                                          Python environment.  Flagged DESTRUCTIVE.

⚠️  SECURITY NOTICE:
    execute_code runs code DIRECTLY on the host machine with the same permissions
    as the Python process running the agent.  There is NO network restriction,
    NO filesystem sandboxing, and NO resource capping beyond the configurable
    timeout.  The permission layer marks this tool as destructive so the user
    is prompted to approve each execution before it runs.  Never allow the
    agent to call this tool without that approval step.

    install_package modifies the Python environment — installing packages from
    untrusted sources can introduce malicious code.  It is also flagged as
    destructive and requires user confirmation before running.
"""

import json
import logging
import re
import subprocess
import sys
import tempfile
import os
import threading
from pathlib import Path
from typing import Callable

from agent_tools import register_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streaming callback — set by main.py at startup
# ---------------------------------------------------------------------------

# Set via set_send_event_callback() once the WebSocket send_event function is
# available.  execute_code calls this with ("execution_output", {...}) for each
# stdout line so the frontend can update the tool block in real time.
_send_event_callback: Callable | None = None


def set_send_event_callback(cb: Callable) -> None:
    """
    Register the async send_event function so execute_code can stream output.

    Called from main.py after the agent is initialised.  The callback must
    accept (event_type: str, data: dict) and schedule the coroutine safely
    across thread boundaries (main.py wraps it with asyncio.run_coroutine_threadsafe).
    """
    global _send_event_callback
    _send_event_callback = cb
    logger.info("[code_executor] Streaming callback registered.")


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
        # Run the subprocess with Popen so we can stream stdout line-by-line
        # ------------------------------------------------------------------
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        # --- Stream stdout -----------------------------------------------
        # Read line-by-line; fire the callback for each line so the frontend
        # can append it to the active tool block in real time.
        # proc.stdout is None only if stdout wasn't piped — it always is here.
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                stdout_lines.append(line)
                if _send_event_callback is not None:
                    try:
                        _send_event_callback(
                            "execution_output",
                            {"line": line.rstrip("\r\n"), "stream": "stdout"},
                        )
                    except Exception as cb_err:
                        logger.debug(f"[code_executor] Streaming callback error (non-fatal): {cb_err}")

            # Wait for process to finish; enforce the timeout here.
            _, stderr_raw = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()  # drain pipes to avoid zombie
            raise  # re-raised → caught by the outer TimeoutExpired handler

        stderr_lines = stderr_raw.splitlines(keepends=True) if stderr_raw else []

        stdout_str = "".join(stdout_lines)
        stderr_str = "".join(stderr_lines)
        returncode  = proc.returncode

        logger.info(
            f"[code_executor] Finished — exit_code={returncode}, "
            f"stdout={len(stdout_str)} chars, stderr={len(stderr_str)} chars"
        )

        return {
            "success":   returncode == 0,
            "stdout":    stdout_str,
            "stderr":    stderr_str,
            "exit_code": returncode,
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
# install_package
# ---------------------------------------------------------------------------

# Allowlist pattern: only plain package names are accepted.
# Rejects anything with shell metacharacters, paths, or injection attempts.
_PACKAGE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


async def install_package(package: str, version: str = "") -> dict:
    """
    Install a Python package into the current environment using pip.

    ⚠️  SECURITY NOTICE: Installing packages from untrusted sources can
    introduce malicious code into the environment.  Always confirm the package
    name and source before approving this tool call.

    Args:
        package:  The package name, e.g. "requests" or "numpy".
                  Must match ^[a-zA-Z0-9_\\-.]+$ to prevent injection.
        version:  Optional exact version string, e.g. "2.28.1".
                  If provided, pip is called as: pip install package==version.
                  Leave empty to install the latest compatible version.

    Returns:
        {
            "success":   bool,
            "package":   str,
            "version":   str,          # echoes version arg (may be "")
            "stdout":    str,          # last 500 chars of pip stdout
            "stderr":    str,          # last 500 chars of pip stderr
            "exit_code": int,
        }

    DESTRUCTIVE: modifies the Python environment — requires user approval.
    """
    # ---- Validate package name ------------------------------------------
    pkg = package.strip()
    if not pkg:
        return {
            "success":   False,
            "package":   package,
            "version":   version,
            "stdout":    "",
            "stderr":    "Package name cannot be empty.",
            "exit_code": -1,
        }
    if not _PACKAGE_NAME_RE.match(pkg):
        return {
            "success":   False,
            "package":   package,
            "version":   version,
            "stdout":    "",
            "stderr":    (
                f"Invalid package name: {package!r}. "
                "Only letters, digits, hyphens, underscores, and dots are allowed."
            ),
            "exit_code": -1,
        }

    # ---- Build pip spec -------------------------------------------------
    ver = version.strip()
    spec = f"{pkg}=={ver}" if ver else pkg

    logger.info(f"[code_executor] install_package: pip install {spec}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", spec],
            capture_output=True,
            text=True,
            timeout=120,
        )
        success    = result.returncode == 0
        stdout_out = result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
        stderr_out = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr

        if success:
            logger.info(f"[code_executor] install_package OK: {spec}")
        else:
            logger.warning(f"[code_executor] install_package FAILED: {spec} — {stderr_out[:200]}")

        return {
            "success":   success,
            "package":   pkg,
            "version":   ver,
            "stdout":    stdout_out,
            "stderr":    stderr_out,
            "exit_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        logger.warning(f"[code_executor] install_package timed out after 120s: {spec}")
        return {
            "success":   False,
            "package":   pkg,
            "version":   ver,
            "stdout":    "",
            "stderr":    "pip install timed out after 120 seconds.",
            "exit_code": -1,
        }
    except Exception as e:
        logger.exception(f"[code_executor] install_package unexpected error: {spec}")
        return {
            "success":   False,
            "package":   pkg,
            "version":   ver,
            "stdout":    "",
            "stderr":    str(e),
            "exit_code": -1,
        }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_code_executor_tools() -> None:
    """
    Register execute_code and install_package with the tool registry.

    Both tools are marked destructive=True so the permission layer will ask
    the user to confirm before execution.
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

    register_tool(
        name="install_package",
        description=(
            "Install a Python package into the current environment using pip. "
            "Use this before running code that requires a package not yet installed. "
            "Check scaffold_project dependencies first to know what packages are needed. "
            "Accepts an optional exact version string (e.g. '2.28.1'); "
            "omit version to install the latest compatible release. "
            "⚠️ Only install packages from trusted sources — pip install runs "
            "arbitrary code from PyPI."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "package": {
                    "type": "string",
                    "description": (
                        "Package name to install, e.g. 'requests' or 'numpy'. "
                        "Must be a plain name — no shell operators, paths, or extras."
                    ),
                },
                "version": {
                    "type": "string",
                    "description": (
                        "Optional exact version, e.g. '2.28.1'. "
                        "Leave empty to install the latest compatible version."
                    ),
                },
            },
            "required": ["package"],
        },
        handler=install_package,
        is_destructive=True,  # modifies the Python environment
    )
    logger.info("[code_executor] Registered tool: install_package (destructive)")

