"""
project_manager.py  —  Phase 4b: Project Progress Tracker
                     —  Phase 6b: Rich State Snapshots

Registers tools that maintain lightweight progress state for software
projects scaffolded via scaffold_project:

  get_project_status   — read current completion state from progress.json
  mark_file_complete   — record that a file has been written
  read_project_state   — read the rich state.json snapshot (Phase 6b)

Progress state lives in outputs/{project_name}/progress.json alongside the
scaffold.json produced by Phase 4a.  These tools are the agent's
checkpoint mechanism: they replace "mental bookkeeping" with explicit,
persistent state so long tasks survive interruptions and context reloads.

Phase 6b adds a second state file — outputs/{project_name}/state.json —
that is updated after every meaningful action.  It contains everything the
agent needs to resume a project in a single call:
  - completed_files, pending_files, last_action, next_step
  - key_decisions (architectural choices logged during the run)
  - test_status, entry_point, dependencies, notes

All tools are NON-DESTRUCTIVE (no approval required).
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


def _resolve_project_dir(project_name: str) -> Path:
    return _OUTPUTS_DIR / project_name.strip()


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
# Phase 6b: State snapshot helpers
# ---------------------------------------------------------------------------

def _compute_status(scaffold: dict | None, progress: dict) -> str:
    """
    Compute a high-level project status string from scaffold + progress.

    Returns one of: "in_progress", "complete", "blocked", "unknown".
    """
    if scaffold is None:
        return "unknown"

    impl_order: list[str] = scaffold.get("implementation_order", [])
    if not impl_order:
        return "unknown"

    completed_names = {e.get("file") for e in progress.get("completed_files", [])}
    pending = [f for f in impl_order if f not in completed_names]

    last_test = progress.get("last_test")
    if last_test and not last_test.get("passed") and not pending:
        # All files written but test failing — blocked
        return "blocked"

    if not pending:
        # All files written
        if last_test and last_test.get("passed"):
            return "complete"
        return "in_progress"  # files done but not yet tested

    return "in_progress"


def _pending_files(scaffold: dict | None, progress: dict) -> list[str]:
    """Return the list of implementation_order files not yet completed."""
    if scaffold is None:
        return []
    impl_order: list[str] = scaffold.get("implementation_order", [])
    completed_names = {e.get("file") for e in progress.get("completed_files", [])}
    return [f for f in impl_order if f not in completed_names]


def _update_state_snapshot(
    project_name: str,
    scaffold: dict | None,
    progress: dict,
    last_action: str,
    next_step: str,
    key_decisions: list[str] | None = None,
) -> None:
    """
    Write a rich state snapshot after every meaningful project action.

    This is the single file the agent reads to resume a project —
    it contains everything needed without re-reading all source files.

    Called by:
      - mark_file_complete()  after recording a file as done
      - get_project_status()  so a status check always refreshes the snapshot
      - project_tester.py     after every test run (imported there)

    The snapshot is intentionally denormalised: the agent should never
    need to open both progress.json and scaffold.json just to answer
    "what do I do next?".
    """
    project_dir = _resolve_project_dir(project_name)
    state = {
        "project_name":   project_name,
        "updated_at":     _now_iso(),
        "status":         _compute_status(scaffold, progress),
        "completed_files": [e["file"] for e in progress.get("completed_files", [])],
        "pending_files":   _pending_files(scaffold, progress),
        "last_action":     last_action,
        "next_step":       next_step,
        "key_decisions":   key_decisions or [],
        "test_status":     progress.get("last_test", {}).get("passed") if progress.get("last_test") else None,
        "entry_point":     scaffold.get("entry_point", "") if scaffold else "",
        "dependencies":    scaffold.get("dependencies", []) if scaffold else [],
        "notes":           progress.get("notes", []),
    }
    state_path = project_dir / "state.json"
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.debug(
            f"[project_manager] State snapshot updated for '{project_name}': "
            f"status={state['status']}, pending={len(state['pending_files'])} files"
        )
    except Exception as e:
        logger.warning(f"[project_manager] Could not write state.json: {e}")


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

    Phase 6b: also refreshes the state.json snapshot on every call so
    read_project_state always has current data.

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
    project_dir = _resolve_project_dir(project_name)

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

    # Phase 6b: refresh state snapshot so read_project_state stays current
    _update_state_snapshot(
        project_name=project_name,
        scaffold=scaffold,
        progress=progress,
        last_action="get_project_status called",
        next_step=(
            f"Implement {next_file}"
            if next_file
            else ("Run tests" if not last_test else "Project complete")
        ),
    )

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

    Phase 6b: updates state.json after every call so the agent can resume
    from a precise checkpoint.

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
    project_dir = _resolve_project_dir(project_name)
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

    # Phase 6b: refresh the state snapshot
    _update_state_snapshot(
        project_name=project_name,
        scaffold=scaffold,
        progress=progress,
        last_action=f"Implemented {filename}" + (f" — {notes}" if notes else ""),
        next_step=(
            f"Implement {remaining[0]}" if remaining else "All files done — run tests"
        ),
    )

    return {
        "success":         True,
        "completed_count": len(completed_files),
        "remaining":       remaining,
    }


# ---------------------------------------------------------------------------
# Phase 6b — Tool: read_project_state
# ---------------------------------------------------------------------------

async def read_project_state(project_name: str) -> dict:
    """
    Read the rich state snapshot for a project (outputs/{project_name}/state.json).

    Use this at the start of any project resumption instead of calling
    get_project_status + reading individual files.  The snapshot contains
    everything needed to pick up exactly where the last session left off:
    completed files, pending files, the last action taken, the recommended
    next step, key architectural decisions, and test status.

    If state.json does not yet exist (first session), falls back to
    get_project_status to generate it.

    Args:
        project_name: Project identifier (matches outputs/{project_name}/).

    Returns:
        The full state dict, or {"success": False, "error": "..."} if the
        project does not exist.
    """
    project_dir = _resolve_project_dir(project_name)
    state_path  = project_dir / "state.json"

    if state_path.exists():
        try:
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
            state["success"] = True
            logger.info(
                f"[project_manager] read_project_state '{project_name}': "
                f"status={state.get('status')}, "
                f"pending={len(state.get('pending_files', []))} files"
            )
            return state
        except Exception as e:
            logger.warning(f"[project_manager] Could not read state.json: {e}")
            # Fall through to rebuild from scratch

    # state.json missing or unreadable — regenerate it via get_project_status
    logger.info(
        f"[project_manager] state.json not found for '{project_name}' — "
        "falling back to get_project_status to rebuild."
    )
    status = await get_project_status(project_name)
    if not status.get("success"):
        return status

    # get_project_status wrote state.json — read it back now
    if state_path.exists():
        try:
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
            state["success"] = True
            return state
        except Exception:
            pass

    # Absolute fallback — return the status dict directly
    return status


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_project_manager_tools() -> None:
    """Register get_project_status, mark_file_complete, and read_project_state."""

    register_tool(
        name="get_project_status",
        description=(
            "Get the current implementation progress for a scaffolded project. "
            "Shows which files are done (exist on disk), which are still pending, "
            "the last test result, and the next file to implement. "
            "Call this after scaffold approval and after each file is written to "
            "stay oriented in a multi-file project. "
            "Also refreshes the state.json snapshot (Phase 6b)."
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
            "if interrupted. Calling this multiple times for the same file is safe. "
            "Also updates the state.json snapshot (Phase 6b)."
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

    register_tool(
        name="read_project_state",
        description=(
            "Read the rich state snapshot for a project — use this at the start "
            "of any resumption task instead of re-reading all project files. "
            "Contains: completed files, pending files, last action, next step, "
            "key decisions, test status, entry point, and dependencies. "
            "Falls back to get_project_status automatically if state.json does "
            "not yet exist. Always call this first when resuming a project."
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
        handler=read_project_state,
        is_destructive=False,
    )

    logger.info(
        "[startup] Registered tools: get_project_status, mark_file_complete, read_project_state"
    )
