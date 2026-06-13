"""
agent_tools/tool_writer.py  —  Agent Self-Modification Tools

Provides two tools that let the agent create and register its own tools:

  write_tool(filename, code)   — Validate a filename, write code to
                                  agent_tools/generated/{filename}, then run the
                                  static validator. Returns success/failure.

  reload_tool(filename)        — Dynamically import an already-written file from
                                  agent_tools/generated/{filename} and register its
                                  tools into the live registry.

Security:
  - write_tool is marked destructive (requires user approval before writing).
  - reload_tool is marked destructive (registering arbitrary code is high-risk).
  - Filename validation prevents path traversal: only plain alphanumeric + underscore
    names ending in .py are accepted.  No slashes, backslashes, or ".." allowed.
  - The agent can ONLY write to agent_tools/generated/.  Built-in tools in
    agent_tools/ are never touched.

Workflow the agent should follow:
  1. write_tool("my_tool.py", code)   → validates syntax, writes file
  2. reload_tool("my_tool.py")        → imports file, calls register_* function
  3. The new tool is now live in the registry for this session and all future
     sessions (it is auto-loaded on startup).
"""

import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import register_tool
from .hot_reload import validate_tool_file, hot_reload_tool, GENERATED_DIR

logger = logging.getLogger(__name__)

# Only allow plain identifiers as filenames — no path components, no special chars.
_SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_]+\.py$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_filename(filename: str) -> tuple[bool, str]:
    """
    Return (True, "") if the filename is safe, or (False, reason) if not.

    Rules:
      - Must end with .py
      - Must consist only of [a-zA-Z0-9_] before the .py suffix
      - Must NOT contain /, \\, or .. (path traversal guard)
    """
    if not filename.endswith(".py"):
        return False, "filename must end with .py"

    if "/" in filename or "\\" in filename or ".." in filename:
        return False, "path traversal detected — filename must be a plain name, not a path"

    if not _SAFE_FILENAME_RE.match(filename):
        return False, (
            "filename contains invalid characters. "
            "Only letters, digits, and underscores are allowed before the .py extension."
        )

    return True, ""


# ---------------------------------------------------------------------------
# Runtime validation helper
# ---------------------------------------------------------------------------

def _test_tool_file(path: Path) -> tuple[bool, bool, str]:
    """
    Run two lightweight subprocess checks on the written tool file.

    Check 1 (syntax):
        Run 'python -c "import ast; ast.parse(open(path).read())"' — fast,
        safe, catches all syntax errors without executing any user code.

    Check 2 (import):
        Attempt to load the file as a module via importlib in a subprocess
        sandbox.  This catches NameError/ImportError at module level without
        touching the live registry.

    Returns:
        (syntax_valid: bool, import_valid: bool, error_message: str)
        error_message is empty when both checks pass.
    """
    # ── Check 1: Syntax ───────────────────────────────────────────────────────
    syntax_cmd = [
        sys.executable, "-c",
        f"import ast; ast.parse(open({str(path)!r}).read())",
    ]
    try:
        result = subprocess.run(
            syntax_cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        syntax_ok = result.returncode == 0
        if not syntax_ok:
            err = (result.stderr or result.stdout).strip()
            return False, False, f"Syntax check failed: {err}"
    except subprocess.TimeoutExpired:
        return False, False, "Syntax check timed out."
    except Exception as e:
        return False, False, f"Syntax check error: {e}"

    # ── Check 2: Import / module-level execution ──────────────────────────────
    import_cmd = [
        sys.executable, "-c",
        (
            "import importlib.util, sys; "
            f"spec = importlib.util.spec_from_file_location('_test_tool', {str(path)!r}); "
            "m = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(m)"
        ),
    ]
    try:
        result = subprocess.run(
            import_cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        import_ok = result.returncode == 0
        if not import_ok:
            err = (result.stderr or result.stdout).strip()
            return True, False, f"Import check failed: {err}"
    except subprocess.TimeoutExpired:
        return True, False, "Import check timed out."
    except Exception as e:
        return True, False, f"Import check error: {e}"

    return True, True, ""


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def write_tool(filename: str, code: str) -> dict[str, Any]:
    """
    Write agent-generated tool code to agent_tools/generated/{filename}.

    Steps:
      1. Validate the filename (no path traversal, .py only, safe chars).
      2. Write the code to the generated directory.
      3. Run static validation (syntax, async def, register_ function present).
      4. Return a structured result so the agent knows what to do next.

    The agent should follow a write_tool → reload_tool sequence to activate.

    Args:
        filename:  Plain filename, e.g. "weather_tool.py".  No path components.
        code:      Full Python source code for the tool module.

    Returns:
        {
          "success":    bool,
          "path":       str  (absolute path where the file was written),
          "validation": "OK" | error message,
        }
    """
    # --- Filename safety check ---
    ok, reason = _check_filename(filename)
    if not ok:
        logger.warning(f"[tool_writer] Rejected unsafe filename: {filename!r} — {reason}")
        return {"success": False, "path": "", "validation": f"Invalid filename: {reason}"}

    # --- Ensure the generated directory exists ---
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    dest = GENERATED_DIR / filename

    # --- Write the code ---
    try:
        dest.write_text(code, encoding="utf-8")
        logger.info(f"[tool_writer] Wrote tool file: {dest}")
    except OSError as e:
        return {"success": False, "path": str(dest), "validation": f"Write error: {e}"}

    # --- Static validation ---
    valid, msg = validate_tool_file(dest)
    if not valid:
        logger.warning(f"[tool_writer] Validation failed for {filename}: {msg}")
        # Leave the file on disk so the agent can read it back and fix it,
        # but report the failure clearly.
        return {"success": False, "path": str(dest), "validation": msg}

    logger.info(f"[tool_writer] {filename} passed validation — call reload_tool to activate.")

    # ── Runtime validation (syntax + import in subprocess sandbox) ────────────
    syntax_valid, import_valid, test_error = _test_tool_file(dest)
    ready_to_reload = syntax_valid and import_valid

    if not ready_to_reload:
        logger.warning(
            f"[tool_writer] Runtime test failed for {filename}: {test_error}"
        )
        return {
            "success":         False,
            "path":            str(dest),
            "validation":      "OK",           # static validation passed …
            "syntax_valid":    syntax_valid,
            "import_valid":    import_valid,
            "ready_to_reload": False,
            "error":           test_error,     # … but runtime test failed
        }

    return {
        "success":         True,
        "path":            str(dest),
        "validation":      "OK",
        "syntax_valid":    True,
        "import_valid":    True,
        "ready_to_reload": True,
    }


async def reload_tool(filename: str) -> dict[str, Any]:
    """
    Dynamically import and register a tool file from agent_tools/generated/.

    The file must already exist (written by write_tool or manually placed there).
    Calls hot_reload_tool() without a WebSocket send_event — the result is returned
    directly to the agent as a tool result instead.

    Args:
        filename:  Plain filename, e.g. "weather_tool.py".

    Returns:
        {
          "success": bool,
          "message": str,
        }
    """
    ok, reason = _check_filename(filename)
    if not ok:
        return {"success": False, "message": f"Invalid filename: {reason}"}

    path = GENERATED_DIR / filename

    if not path.exists():
        return {
            "success": False,
            "message": (
                f"File not found: {path}. "
                "Use write_tool first to create the file before calling reload_tool."
            ),
        }

    # hot_reload_tool handles validation + importlib loading + register_* call.
    # send_event=None because we have no WebSocket handle here;
    # main.py's auto-loader passes a real send_event on startup.
    success, message = await hot_reload_tool(path, send_event=None)

    if success:
        logger.info(f"[tool_writer] reload_tool: {filename} registered successfully.")
    else:
        logger.warning(f"[tool_writer] reload_tool: {filename} failed — {message}")

    return {"success": success, "message": message}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_tool_writer_tools() -> None:
    """Register write_tool and reload_tool. Call once at startup from main.py."""

    register_tool(
        name="write_tool",
        description=(
            "Write a new Python tool to agent_tools/generated/{filename}. "
            "The code must define at least one async tool handler function and "
            "a register_<name>_tools() function that calls register_tool() from agent_tools. "
            "After writing, call reload_tool(filename) to activate the new tool. "
            "Returns: {success, path, validation, syntax_valid, import_valid, ready_to_reload}. "
            "If ready_to_reload is false, the error field explains what to fix before retrying. "
            "Example filename: 'calculator_tool.py'. "
            "Filename must be a plain name (letters, digits, underscores) ending in .py — "
            "no paths, no slashes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "filename": {
                    "type":        "string",
                    "description": "Plain .py filename, e.g. 'my_tool.py'. No path components.",
                },
                "code": {
                    "type":        "string",
                    "description": "Full Python source code for the tool module.",
                },
            },
            "required": ["filename", "code"],
        },
        handler=write_tool,
        is_destructive=True,   # Writes executable code to disk — requires user approval
    )

    register_tool(
        name="reload_tool",
        description=(
            "Dynamically import and register a tool file from agent_tools/generated/. "
            "The file must already exist (created by write_tool). "
            "After this call succeeds, the new tool is live in the registry and can be "
            "used immediately. The tool also persists across server restarts. "
            "Returns: {success, message}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "filename": {
                    "type":        "string",
                    "description": "Plain .py filename to load, e.g. 'my_tool.py'.",
                },
            },
            "required": ["filename"],
        },
        handler=reload_tool,
        is_destructive=True,   # Registers executable code into the live runtime — user approval required
    )
