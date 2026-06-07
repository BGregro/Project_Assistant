"""
agent_tools/hot_reload.py  —  Tool Validation and Hot-Reload

Handles the two-step process of bringing agent-written tools to life:
  1. validate_tool_file()   — static checks before any code runs.
  2. hot_reload_tool()      — dynamic import + registration into the live registry.

Security notes:
  - Validation catches syntax errors and enforces the async + register_ convention
    BEFORE any code is executed.
  - importlib loads the file directly — it runs module-level code in the file, so
    only trusted/validated paths should ever reach hot_reload_tool().
  - Agent self-modification is restricted to agent_tools/generated/ by tool_writer.py.
    hot_reload_tool() itself doesn't enforce the directory — that's the caller's job.
"""

import ast
import importlib.util
import logging
from pathlib import Path
from typing import Callable, Awaitable, Optional

logger = logging.getLogger(__name__)

# The one directory the agent is allowed to populate.
GENERATED_DIR = Path(__file__).parent / "generated"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_tool_file(path: Path) -> tuple[bool, str]:
    """
    Run static checks on a Python file before attempting to import it.

    Checks (in order):
      1. File is readable and parseable (AST parse — catches syntax errors).
      2. Contains at least one async def (tools must be async).
      3. Contains a register_* function (registration entry point convention).

    Returns:
        (True,  "OK")            if all checks pass.
        (False, error_message)   if any check fails.
    """
    # --- Read the file ---
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"Cannot read file: {e}"

    # --- Check 1: Syntax ---
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"

    # --- Walk the AST once for both remaining checks ---
    has_async_def   = False
    has_register_fn = False

    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            has_async_def = True
        # Top-level sync or async function whose name starts with "register_"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("register_"):
                has_register_fn = True

    # --- Check 2: At least one async def ---
    if not has_async_def:
        return False, (
            "No async def found. Tool handler functions must be declared with "
            "'async def' so they can be awaited inside the agent loop."
        )

    # --- Check 3: A register_* function ---
    if not has_register_fn:
        return False, (
            "No register_ function found. Every tool file must include a "
            "register_<name>_tools() function that calls register_tool() from "
            "agent_tools to add the tool(s) to the live registry."
        )

    return True, "OK"


# ---------------------------------------------------------------------------
# Hot-reload
# ---------------------------------------------------------------------------

async def hot_reload_tool(
    path: Path,
    send_event: Optional[Callable[[str, dict], Awaitable[None]]] = None,
) -> tuple[bool, str]:
    """
    Dynamically import a validated tool file and call its register_* function.

    Steps:
      1. Validate the file (syntax + async + register_ checks).
      2. Load the module via importlib using a unique module name based on the filename.
      3. Find the register_* function and call it.
      4. Optionally emit a "tool_registered" WebSocket event.

    Args:
        path:        Absolute path to the .py file to load.
        send_event:  Optional WebSocket event sender (main.py passes this in;
                     tool_writer.py passes None when called without a live WebSocket).

    Returns:
        (True,  "Registered successfully")  on success.
        (False, error_message)              if validation or import fails.
    """
    # --- Validate first — never import unvalidated code ---
    valid, reason = validate_tool_file(path)
    if not valid:
        logger.warning(f"[hot_reload] Validation failed for {path.name}: {reason}")
        return False, f"Validation failed: {reason}"

    try:
        # Build a unique module name so re-loading the same file doesn't collide
        # with a previous import in sys.modules.
        module_name = f"agent_tools.generated.{path.stem}"

        spec   = importlib.util.spec_from_file_location(module_name, str(path))
        module = importlib.util.module_from_spec(spec)

        # Execute the module (runs top-level code, defines functions/classes).
        spec.loader.exec_module(module)

        # Find the register_* function — there should be exactly one.
        register_fn = None
        for attr_name in dir(module):
            if attr_name.startswith("register_") and callable(getattr(module, attr_name)):
                register_fn = getattr(module, attr_name)
                break  # use the first one found

        if register_fn is None:
            # Shouldn't happen after validation, but be safe.
            return False, "register_ function disappeared after import — this is a bug."

        # Call the registration function (sync or async).
        import asyncio, inspect
        if inspect.iscoroutinefunction(register_fn):
            await register_fn()
        else:
            register_fn()

        logger.info(f"[hot_reload] Successfully registered tools from: {path.name}")

        # Emit WebSocket event if a sender was provided.
        if send_event is not None:
            await send_event("tool_registered", {
                "tool_file": path.name,
                "status":    "registered",
            })

        return True, "Registered successfully"

    except Exception as e:
        logger.error(f"[hot_reload] Error loading {path.name}: {e}", exc_info=True)
        return False, str(e)


# ---------------------------------------------------------------------------
# Listing generated tools
# ---------------------------------------------------------------------------

def list_generated_tools() -> list[str]:
    """
    Return the filenames of all .py files in agent_tools/generated/,
    excluding __init__.py.

    Used by capabilities.py to show the agent what it has already written.
    """
    if not GENERATED_DIR.exists():
        return []
    return sorted(
        p.name
        for p in GENERATED_DIR.glob("*.py")
        if p.name != "__init__.py"
    )
