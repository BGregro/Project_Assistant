"""
project_manager.py  —  Phase 4b: Project Progress Tracker

Registers two tools that maintain lightweight progress state for software
projects scaffolded via scaffold_project:

  get_project_status   — read current completion state from progress.json
  mark_file_complete   — record that a file has been written

Progress state lives in outputs/{project_name}/progress.json alongside the
scaffold.json produced by Phase 4a.  These tools are the agent's
checkpoint mechanism: they replace "mental bookkeeping" with explicit,
persistent state so long tasks survive interruptions and context reloads.

Both tools are NON-DESTRUCTIVE (no approval required).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from agent_tools import register_tool

logger = logging.getLogger(__name__)

# Project root: backend/agent_tools/ → backend/ → project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_OUTPUTS_DIR  = _PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_scaffold(project_dir: Path) -> dict | None:
    """Read scaffold.json, return None if missing or unparseable."""
    scaffold_path = project_dir / "scaffold.json"
    if not scaffold_path.exists():
        return None
    try:
        with open(scaffold_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[project_manager] Could not read scaffold.json: {e}")
        return None


def _load_progress(project_dir: Path) -> dict:
    """
    Read progress.json.  If the file is missing or corrupt, return an empty
    progress structure compatible with the expected schema.
    """
    progress_path = project_dir / "progress.json"
    empty = {
        "project_name":    project_dir.name,
        "created_at":      _now_iso(),
        "completed_files": [],
        "last_test":       None,
        "test_attempts":   0,
    }
    if not progress_path.exists():
        return empty
    try:
        with open(progress_path, encoding="utf-8") as f:
            data = json.load(f)
        # Forward-compat: ensure newer keys exist in older files
        data.setdefault("test_attempts", 0)
        return data
    except Exception as e:
        logger.warning(f"[project_manager] Could not read progress.json: {e}")
        return empty


def _save_progress(project_dir: Path, progress: dict) -> None:
    """Write progress.json (creates parents if needed)."""
    project_dir.mkdir(parents=True, exist_ok=True)
    progress_path = project_dir / "progress.json"
    try:
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[project_manager] Could not write progress.json: {e}")


# ---------------------------------------------------------------------------
# Tool: get_project_status
# ---------------------------------------------------------------------------

async def get_project_status(project_name: str) -> dict:
    """
    Return the current completion state for a scaffolded project.

    Reads scaffold.json to get the full implementation plan, and
    progress.json to determine which files are done.  Also checks the
    filesystem directly — a file is considered 'completed' if it actually
    exists on disk, regardless of whether mark_file_complete was called.

    Args:
        project_name: The project identifier (matches outputs/{project_name}/).

    Returns:
        {
            "project_name":   str,
            "total_files":    int,
            "completed":      [str, ...],   # filenames that exist on disk
            "pending":        [str, ...],   # filenames not yet written
            "last_test_result": str | null, # "passed" / "failed" / null
            "next_file":      str | null,   # first pending file (to implement next)
            "ready_to_test":  bool,         # True when all files are present
        }
    """
    project_dir = _OUTPUTS_DIR / project_name.strip()

    # ── Scaffold ──────────────────────────────────────────────────────────────
    scaffold = _load_scaffold(project_dir)
    if scaffold is None:
        return {
            "success": False,
            "error": (
                f"No scaffold found for project '{project_name}'. "
                f"Call scaffold_project first."
            ),
        }

    impl_order: list[str] = scaffold.get("implementation_order", [])

    # ── Progress ──────────────────────────────────────────────────────────────
    progress = _load_progress(project_dir)
    last_test = progress.get("last_test")
    last_test_result: str | None = None
    if last_test:
        last_test_result = "passed" if last_test.get("passed") else "failed"

    # ── File existence check (ground truth) ───────────────────────────────────
    # A file counts as done if it physically exists in the project directory.
    # This is more reliable than relying solely on mark_file_complete calls.
    completed: list[str] = []
    pending:   list[str] = []

    for filename in impl_order:
        file_path = project_dir / filename
        # Handle nested paths like "src/utils.py" correctly
        if file_path.exists():
            completed.append(filename)
        else:
            pending.append(filename)

    next_file     = pending[0] if pending else None
    ready_to_test = len(pending) == 0 and len(impl_order) > 0

    return {
        "success":          True,
        "project_name":     project_name,
        "total_files":      len(impl_order),
        "completed":        completed,
        "pending":          pending,
        "last_test_result": last_test_result,
        "next_file":        next_file,
        "ready_to_test":    ready_to_test,
    }


# ---------------------------------------------------------------------------
# Tool: mark_file_complete
# ---------------------------------------------------------------------------

async def mark_file_complete(
    project_name: str,
    filename: str,
    notes: str = "",
) -> dict:
    """
    Record that a file has been successfully written.

    Appends an entry to the completed_files list in progress.json.
    Calling this multiple times for the same filename is safe — duplicate
    entries are ignored so the list stays clean.

    Args:
        project_name: Project identifier (matches outputs/{project_name}/).
        filename:     The file that was just written (e.g. "main.py", "utils.py").
        notes:        Optional note about implementation decisions or caveats.

    Returns:
        {
            "success":        bool,
            "completed_count": int,  # total completed files after this call
            "remaining":      [str], # files still pending in implementation_order
        }
    """
    project_dir = _OUTPUTS_DIR / project_name.strip()
    scaffold    = _load_scaffold(project_dir)
    impl_order: list[str] = scaffold.get("implementation_order", []) if scaffold else []

    progress = _load_progress(project_dir)
    completed_files: list = progress.get("completed_files", [])

    # Avoid duplicate entries for the same filename
    already_recorded = any(e.get("file") == filename for e in completed_files)
    if not already_recorded:
        completed_files.append({
            "file":         filename,
            "completed_at": _now_iso(),
            "notes":        notes,
        })
        progress["completed_files"] = completed_files
        _save_progress(project_dir, progress)
        logger.info(f"[project_manager] Marked complete: {project_name}/{filename}")
    else:
        logger.debug(f"[project_manager] {filename} already in completed list — skipped duplicate.")

    # Compute remaining from implementation_order vs recorded completions
    recorded_names = {e.get("file") for e in completed_files}
    remaining = [f for f in impl_order if f not in recorded_names]

    return {
        "success":         True,
        "completed_count": len(completed_files),
        "remaining":       remaining,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_project_manager_tools() -> None:
    """Register get_project_status and mark_file_complete."""

    register_tool(
        name="get_project_status",
        description=(
            "Get the current implementation progress for a scaffolded project. "
            "Shows which files are done (exist on disk), which are still pending, "
            "the last test result, and the next file to implement. "
            "Call this after scaffold approval and after each file is written to "
            "stay oriented in a multi-file project."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_name": {
                    "type":        "string",
                    "description": "Project identifier matching the outputs/{project_name}/ directory.",
                },
            },
            "required": ["project_name"],
        },
        handler=get_project_status,
        is_destructive=False,
    )

    register_tool(
        name="mark_file_complete",
        description=(
            "Record that a project file has been successfully written. "
            "Call this immediately after writing each file during project implementation. "
            "Maintains a progress log in progress.json so the agent can resume "
            "if interrupted. Calling this multiple times for the same file is safe."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_name": {
                    "type":        "string",
                    "description": "Project identifier matching the outputs/{project_name}/ directory.",
                },
                "filename": {
                    "type":        "string",
                    "description": "The filename just written (e.g. 'main.py', 'utils/helpers.py').",
                },
                "notes": {
                    "type":        "string",
                    "description": "Optional notes about implementation decisions or caveats for this file.",
                },
            },
            "required": ["project_name", "filename"],
        },
        handler=mark_file_complete,
        is_destructive=False,
    )

    logger.info("[startup] Registered tools: get_project_status, mark_file_complete")
