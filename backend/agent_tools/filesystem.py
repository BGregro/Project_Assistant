"""
agent_tools/filesystem.py  —  Filesystem Tools

Provides three tools the agent can use:
  - read_file:       Read the text content of a file.
  - write_file:      Write or append text to a file. Flagged DESTRUCTIVE.
  - list_directory:  List files and subdirectories at a path.

All paths go through _safe_path() which handles Windows drive letters, ~ expansion,
and environment variables. Every operation is logged for audit purposes.

Phase 2 addition — folder tree broadcast:
  After a successful write_file or list_directory call, a compact folder tree
  is generated from the configured tree_root (config.json → "tree_root", default ".")
  and returned as an extra "tree" key in the result dict. agent_core.py detects
  this key and broadcasts a tree_update WebSocket event to the frontend.
"""

import json
import os
import logging
import pathlib
from pathlib import Path
from typing import Any

from . import register_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config access — loaded lazily so registration order doesn't matter
# ---------------------------------------------------------------------------

def _get_tree_root() -> Path:
    """
    Read tree_root from config.json. Defaults to the current working directory.
    Called at tree-generation time (not at import time) so config changes are
    picked up without a restart.
    """
    config_path = Path(__file__).parent.parent.parent / "config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        raw = cfg.get("tree_root", ".")
    except Exception:
        raw = "."
    # Resolve relative to the project root (where config.json lives), not CWD
    project_root = config_path.parent
    p = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


# ---------------------------------------------------------------------------
# Folder tree generator
# ---------------------------------------------------------------------------

def _build_tree(root: Path, max_depth: int = 3) -> str:
    """
    Build a compact text tree of the directory structure, like the `tree` command.

    Only directories and files up to `max_depth` levels deep are shown.
    Hidden entries (starting with '.') and __pycache__ directories are skipped
    to keep the output clean and readable.

    Example output:
        project/
        ├── backend/
        │   ├── agent_core.py
        │   └── main.py
        └── frontend/
            └── index.html
    """
    lines: list[str] = []

    def _walk(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            # Sort: directories first, then files, both alphabetically
            entries = sorted(
                directory.iterdir(),
                key=lambda x: (x.is_file(), x.name.lower()),
            )
            # Filter hidden files and __pycache__ clutter
            entries = [
                e for e in entries
                if not e.name.startswith(".") and e.name != "__pycache__"
            ]
        except PermissionError:
            return

        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            suffix    = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")

            if entry.is_dir():
                extension = "    " if is_last else "│   "
                _walk(entry, prefix + extension, depth + 1)

    root_resolved = root.resolve()
    lines.append(f"{root_resolved.name}/")
    _walk(root_resolved, "", 1)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def _safe_path(path_str: str) -> pathlib.Path:
    """
    Normalise a path string into an absolute pathlib.Path.
    Handles: '~', '%USERPROFILE%', relative paths, forward/backslashes.
    Using os.path.expandvars first handles Windows %ENV_VAR% syntax.
    """
    expanded = os.path.expandvars(os.path.expanduser(str(path_str)))
    return pathlib.Path(expanded).resolve()


# ---------------------------------------------------------------------------
# Tool handlers (all async to work with FastAPI's async event loop)
# ---------------------------------------------------------------------------

async def read_file(path: str) -> dict[str, Any]:
    """
    Read and return the UTF-8 text contents of a file.
    Returns a structured result dict so Claude can reason about success/failure.
    """
    p = _safe_path(path)
    logger.info(f"[filesystem] read_file: {p}")

    if not p.exists():
        return {"success": False, "error": f"File not found: {p}"}
    if not p.is_file():
        return {"success": False, "error": f"Path exists but is not a file: {p}"}

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        size = p.stat().st_size
        logger.info(f"[filesystem] read_file OK: {size} bytes from {p}")
        return {
            "success": True,
            "path": str(p),
            "content": content,
            "size_bytes": size,
        }
    except PermissionError:
        return {"success": False, "error": f"Permission denied: {p}"}
    except Exception as e:
        logger.exception(f"[filesystem] read_file error: {p}")
        return {"success": False, "error": str(e)}


async def write_file(path: str, content: str, mode: str = "overwrite") -> dict[str, Any]:
    """
    Write text content to a file.

    mode="overwrite"  replaces any existing file (or creates new).
    mode="append"     adds content after existing content.

    DESTRUCTIVE: this tool is flagged in the registry and the permission layer
    will ask the user to confirm before this handler is actually called.
    Parent directories are created automatically.

    After a successful write, a folder tree is appended as the "tree" key so
    agent_core.py can broadcast a tree_update event to the frontend sidebar.
    """
    p = _safe_path(path)
    logger.info(f"[filesystem] write_file: {p} (mode={mode!r})")

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return {"success": False, "error": f"Could not create parent directories: {e}"}

    try:
        if mode == "append":
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            p.write_text(content, encoding="utf-8")

        bytes_written = len(content.encode("utf-8"))
        logger.info(f"[filesystem] write_file OK: {bytes_written} bytes to {p}")

        result: dict[str, Any] = {
            "success": True,
            "path": str(p),
            "mode": mode,
            "bytes_written": bytes_written,
        }

        # --- Folder tree broadcast (Feature 2) ---
        # Generate and attach the tree so agent_core can send a tree_update event.
        try:
            tree_root = _get_tree_root()
            result["tree"] = _build_tree(tree_root)
        except Exception as te:
            logger.warning(f"[filesystem] Tree generation failed (non-fatal): {te}")

        return result

    except PermissionError:
        return {"success": False, "error": f"Permission denied: {p}"}
    except Exception as e:
        logger.exception(f"[filesystem] write_file error: {p}")
        return {"success": False, "error": str(e)}


async def list_directory(path: str = ".") -> dict[str, Any]:
    """
    List the contents of a directory.
    Returns type (file/dir), name, and size for each entry, sorted alphabetically.

    Also returns a "tree" key with a compact folder tree from the configured
    tree_root so the frontend sidebar stays in sync (Feature 2).
    """
    p = _safe_path(path)
    logger.info(f"[filesystem] list_directory: {p}")

    if not p.exists():
        return {"success": False, "error": f"Directory not found: {p}"}
    if not p.is_dir():
        return {"success": False, "error": f"Path exists but is not a directory: {p}"}

    try:
        entries = []
        for item in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            entry = {
                "name": item.name,
                "type": "file" if item.is_file() else "directory",
            }
            if item.is_file():
                try:
                    entry["size_bytes"] = item.stat().st_size
                except OSError:
                    entry["size_bytes"] = None
            entries.append(entry)

        logger.info(f"[filesystem] list_directory OK: {len(entries)} entries in {p}")

        result: dict[str, Any] = {
            "success": True,
            "path": str(p),
            "entries": entries,
            "count": len(entries),
        }

        # --- Folder tree broadcast (Feature 2) ---
        try:
            tree_root = _get_tree_root()
            result["tree"] = _build_tree(tree_root)
        except Exception as te:
            logger.warning(f"[filesystem] Tree generation failed (non-fatal): {te}")

        return result

    except PermissionError:
        return {"success": False, "error": f"Permission denied: {p}"}
    except Exception as e:
        logger.exception(f"[filesystem] list_directory error: {p}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Registration — call this once at startup from main.py
# ---------------------------------------------------------------------------

def register_all() -> None:
    """Register all filesystem tools into the global tool registry."""

    register_tool(
        name="read_file",
        description=(
            "Read the text content of a file at the given path. "
            "Returns the content as a string along with file size."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to read.",
                }
            },
            "required": ["path"],
        },
        handler=read_file,
        is_destructive=False,
    )

    register_tool(
        name="write_file",
        description=(
            "Write text content to a file. Creates the file and any parent directories "
            "if they do not exist. Use mode='overwrite' to replace (default) or "
            "mode='append' to add to the end of an existing file."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write to the file.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["overwrite", "append"],
                    "description": "Write mode: 'overwrite' (default) replaces the file, 'append' adds to it.",
                },
            },
            "required": ["path", "content"],
        },
        handler=write_file,
        is_destructive=True,   # ← triggers permission check in agent_core.py
    )

    register_tool(
        name="list_directory",
        description=(
            "List the files and subdirectories inside a directory. "
            "Returns name, type (file/directory), and size for each entry."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list. Defaults to the current working directory.",
                }
            },
            "required": [],
        },
        handler=list_directory,
        is_destructive=False,
    )
