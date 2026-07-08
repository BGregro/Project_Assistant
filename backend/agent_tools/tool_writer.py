"""
agent_tools/tool_writer.py  —  Agent Self-Modification Tools

Provides four tools that let the agent create and register its own tools:

  write_tool(filename, code)   — Validate a filename, write code to
                                  agent_tools/generated/{filename}, then run the
                                  static validator. Returns success/failure.

  reload_tool(filename)        — Dynamically import an already-written file from
                                  agent_tools/generated/{filename} and register its
                                  tools into the live registry.

  design_tool(gap_description) — Phase 15b: use the local qwen3:14b model to turn
                                  a capability gap into a structured tool spec,
                                  saved to outputs/tool_designs/.

  implement_tool_from_design(spec_path) — Phase 15b: use the local model to write
                                  the full implementation from a saved spec, then
                                  validate and hot-reload it automatically.

Security:
  - write_tool is marked destructive (requires user approval before writing).
  - reload_tool is marked destructive (registering arbitrary code is high-risk).
  - implement_tool_from_design is marked destructive (writes a file and
    registers code, same as write_tool + reload_tool combined).
  - design_tool is non-destructive — it only writes a JSON spec to outputs/.
  - Filename validation prevents path traversal: only plain alphanumeric + underscore
    names ending in .py are accepted.  No slashes, backslashes, or ".." allowed.
  - The agent can ONLY write to agent_tools/generated/.  Built-in tools in
    agent_tools/ are never touched.

Workflow the agent should follow:
  1. write_tool("my_tool.py", code)   → validates syntax, writes file
  2. reload_tool("my_tool.py")        → imports file, calls register_* function
  3. The new tool is now live in the registry for this session and all future
     sessions (it is auto-loaded on startup).

Design-first alternative (Phase 15b, produces better tools for non-trivial gaps):
  1. design_tool(gap_description)              → generates + saves a spec
  2. implement_tool_from_design(spec_path)      → writes, validates, hot-reloads
"""

import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import register_tool
from .hot_reload import validate_tool_file, hot_reload_tool, GENERATED_DIR

logger = logging.getLogger(__name__)

# backend/agent_tools/tool_writer.py -> parents: agent_tools, backend, <project root>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TOOL_DESIGNS_DIR = _PROJECT_ROOT / "outputs" / "tool_designs"


def _now_iso() -> str:
    """Current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _get_ollama_url() -> str:
    """Read ollama_base_url from config.json, fall back to default. Never raises."""
    config_path = _PROJECT_ROOT / "config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f).get("ollama_base_url", "http://localhost:11434")
    except Exception:
        return "http://localhost:11434"

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
    try:
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

    except TypeError as e:
        return {
            "success": False,
            "error": f"Serialization error: {e}",
            "hint": (
                "The tool code likely passed a Python function or non-serializable object "
                "as a return value or to register_tool(). Ensure all return values are plain "
                "dicts/lists/strings/numbers, and register_tool() receives description as a "
                "string (second argument), not the handler function."
            ),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


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
    try:
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

    except TypeError as e:
        return {
            "success": False,
            "error": f"Serialization error: {e}",
            "hint": (
                "The tool code likely passed a Python function or non-serializable object "
                "as a return value or to register_tool(). Ensure all return values are plain "
                "dicts/lists/strings/numbers, and register_tool() receives description as a "
                "string (second argument), not the handler function."
            ),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Phase 15b — Design-first tool pipeline
# ---------------------------------------------------------------------------

async def design_tool(gap_description: str) -> dict[str, Any]:
    """
    Use the local qwen3:14b model to turn a capability gap description into a
    structured tool specification, saved to outputs/tool_designs/.

    This is the first step of the design-first pipeline. It never writes or
    registers executable code — only a JSON spec describing what the tool
    should do, so the agent (or the user) can review the plan before any
    code is generated.

    Args:
        gap_description: Free-text description of the missing capability,
                          typically taken from analyze_capability_gap()'s
                          "gaps" list.

    Returns:
        {"success": bool, "spec": {...}, "spec_path": str} on success,
        {"success": False, "error": str} on failure.
    """
    from .local_llm import local_llm_call, strip_think_tags

    prompt = (
        f'Design a Python tool to fill this capability gap: "{gap_description}"\n\n'
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "filename": "descriptive_tool_name.py",\n'
        '  "function_name": "snake_case_name",\n'
        '  "description": "one sentence describing what it does",\n'
        '  "parameters": [{"name": "param", "type": "str", "description": "..."}],\n'
        '  "returns": "description of return dict",\n'
        '  "implementation_notes": "key logic to implement",\n'
        '  "test_input": {"param": "test_value"}\n'
        "}"
    )

    try:
        response = await local_llm_call(
            prompt, model="qwen3:14b", base_url=_get_ollama_url()
        )
        if not response:
            return {
                "success": False,
                "error": "Local LLM unavailable — cannot design tool. Is Ollama running?",
            }

        cleaned = strip_think_tags(response).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if "\n" in cleaned:
                cleaned = cleaned.split("\n", 1)[1]

        try:
            spec = json.loads(cleaned)
        except Exception as e:
            logger.warning(f"[tool_writer] design_tool: could not parse spec JSON: {e}")
            return {
                "success": False,
                "error": f"Could not parse tool spec from local LLM output: {e}",
                "raw_response": cleaned[:500],
            }

        filename = str(spec.get("filename", "")).strip()
        if not filename:
            return {"success": False, "error": "Spec is missing a 'filename' field."}
        # Reuse the same filename-safety guard used by write_tool.
        ok, reason = _check_filename(filename)
        if not ok:
            return {"success": False, "error": f"Spec has an invalid filename: {reason}"}

        # ── Save the spec to outputs/tool_designs/ ─────────────────────────
        _TOOL_DESIGNS_DIR.mkdir(parents=True, exist_ok=True)
        spec_path = _TOOL_DESIGNS_DIR / f"{filename}_spec.json"
        try:
            with open(spec_path, "w", encoding="utf-8") as f:
                json.dump(spec, f, ensure_ascii=False, indent=2)
            logger.info(f"[tool_writer] design_tool: spec saved to {spec_path}")
        except OSError as e:
            return {"success": False, "error": f"Could not save spec: {e}"}

        return {"success": True, "spec": spec, "spec_path": str(spec_path)}

    except Exception as e:
        logger.warning(f"[tool_writer] design_tool failed: {e}")
        return {"success": False, "error": str(e)}


async def implement_tool_from_design(spec_path: str) -> dict[str, Any]:
    """
    Read a tool spec saved by design_tool() and use the local qwen3:14b model
    to write the full implementation, then validate and hot-reload it.

    Steps:
      1. Load the spec JSON from spec_path.
      2. Ask qwen3:14b for the complete Python file content.
      3. Write it to agent_tools/generated/{filename}.
      4. Run validate_tool_file() (syntax, async def, register_ function).
      5. If valid: call hot_reload_tool() to register it immediately.

    Args:
        spec_path: Path to the *_spec.json file produced by design_tool().

    Returns:
        {"success": bool, "filename": str, "validation": str, "registered": bool}
    """
    from .local_llm import local_llm_call, strip_think_tags

    # ── Load the spec ───────────────────────────────────────────────────────
    path = Path(spec_path)
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        return {"success": False, "filename": "", "validation": f"Could not read spec: {e}", "registered": False}
    except json.JSONDecodeError as e:
        return {"success": False, "filename": "", "validation": f"Spec is not valid JSON: {e}", "registered": False}

    filename = str(spec.get("filename", "")).strip()
    function_name = str(spec.get("function_name", "")).strip()

    ok, reason = _check_filename(filename)
    if not ok:
        return {"success": False, "filename": filename, "validation": f"Invalid filename in spec: {reason}", "registered": False}
    if not function_name:
        return {"success": False, "filename": filename, "validation": "Spec is missing 'function_name'.", "registered": False}

    params = spec.get("parameters", [])
    params_str = ", ".join(
        f"{p.get('name', 'arg')}: {p.get('type', 'str')}" for p in params if isinstance(p, dict)
    )

    prompt = (
        "Write a Python tool file based on this spec:\n"
        f"{json.dumps(spec, ensure_ascii=False, indent=2)}\n\n"
        "Requirements:\n"
        f"- Async function: async def {function_name}({params_str}) -> dict\n"
        "- Always return a dict with at least \"success\" key\n"
        "- Import register_tool from agent_tools at the top\n"
        f"- Include a register_{function_name}_tools() function that calls register_tool()\n"
        "- Use is_destructive=False unless the tool writes files or makes external changes\n"
        "- Add a module docstring explaining the tool\n\n"
        "Return ONLY the complete Python file content, no markdown fences."
    )

    try:
        response = await local_llm_call(
            prompt, model="qwen3:14b", base_url=_get_ollama_url()
        )
        if not response:
            return {
                "success": False,
                "filename": filename,
                "validation": "Local LLM unavailable — cannot implement tool. Is Ollama running?",
                "registered": False,
            }

        code = strip_think_tags(response).strip()
        if code.startswith("```"):
            code = code.strip("`")
            if code.startswith("python\n"):
                code = code[len("python\n"):]
            elif "\n" in code:
                code = code.split("\n", 1)[1]

        # ── Write to agent_tools/generated/ ─────────────────────────────────
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        dest = GENERATED_DIR / filename
        try:
            dest.write_text(code, encoding="utf-8")
            logger.info(f"[tool_writer] implement_tool_from_design: wrote {dest}")
        except OSError as e:
            return {"success": False, "filename": filename, "validation": f"Write error: {e}", "registered": False}

        # ── Static validation ────────────────────────────────────────────────
        valid, msg = validate_tool_file(dest)
        if not valid:
            logger.warning(f"[tool_writer] implement_tool_from_design: validation failed — {msg}")
            return {"success": False, "filename": filename, "validation": msg, "registered": False}

        # ── Hot-reload (validate + import + call register_*) ───────────────
        registered, reload_msg = await hot_reload_tool(dest, send_event=None)
        if registered:
            logger.info(f"[tool_writer] implement_tool_from_design: {filename} registered successfully.")
        else:
            logger.warning(f"[tool_writer] implement_tool_from_design: hot-reload failed — {reload_msg}")

        return {
            "success": registered,
            "filename": filename,
            "validation": "OK" if valid else msg,
            "registered": registered,
            "reload_message": reload_msg,
        }

    except TypeError as e:
        return {
            "success": False,
            "filename": filename,
            "validation": f"Serialization error: {e}",
            "registered": False,
            "hint": (
                "The generated tool code likely passed a Python function or non-serializable "
                "object as a return value or to register_tool(). Ensure all return values are "
                "plain dicts/lists/strings/numbers, and register_tool() receives description as "
                "a string (second argument), not the handler function."
            ),
        }
    except Exception as e:
        logger.warning(f"[tool_writer] implement_tool_from_design failed: {e}")
        return {"success": False, "filename": filename, "validation": str(e), "registered": False}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_tool_writer_tools() -> None:
    """Register write_tool, reload_tool, design_tool, implement_tool_from_design.
    Call once at startup from main.py."""

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

    register_tool(
        name="design_tool",
        description=(
            "Design a new tool from a capability gap description, without writing any code. "
            "Uses the local model to produce a structured spec (filename, function_name, "
            "description, parameters, returns, implementation_notes, test_input) and saves it "
            "to outputs/tool_designs/{filename}_spec.json. "
            "Follow with implement_tool_from_design(spec_path) to write and register the tool. "
            "This design-first approach produces better tools than write_tool alone for "
            "non-trivial capability gaps. Returns: {success, spec, spec_path}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "gap_description": {
                    "type": "string",
                    "description": (
                        "Description of the missing capability, e.g. from "
                        "analyze_capability_gap()'s 'gaps' list."
                    ),
                },
            },
            "required": ["gap_description"],
        },
        handler=design_tool,
        is_destructive=False,  # Only writes a JSON spec, no executable code
    )

    register_tool(
        name="implement_tool_from_design",
        description=(
            "Write and register a tool from a spec previously saved by design_tool(). "
            "Reads the spec JSON, uses the local model to generate the full Python "
            "implementation, writes it to agent_tools/generated/{filename}, validates it, "
            "and hot-reloads it into the live registry automatically. "
            "Returns: {success, filename, validation, registered}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "spec_path": {
                    "type": "string",
                    "description": "Path to the *_spec.json file returned by design_tool().",
                },
            },
            "required": ["spec_path"],
        },
        handler=implement_tool_from_design,
        is_destructive=True,   # Writes a file and registers code — same risk as write_tool + reload_tool
    )
