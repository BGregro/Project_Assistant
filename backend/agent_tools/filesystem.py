"""
tools/filesystem.py  —  Filesystem Tools

Provides three tools the agent can use:
  - read_file:       Read the text content of a file.
  - write_file:      Write or append text to a file. Flagged DESTRUCTIVE.
  - list_directory:  List files and subdirectories at a path.

All paths go through _safe_path() which handles Windows drive letters, ~ expansion,
and environment variables. Every operation is logged for audit purposes.
"""

import os
import logging
import pathlib
from typing import Any

from . import register_tool

logger = logging.getLogger(__name__)


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
    """
    p = _safe_path(path)
    logger.info(f"[filesystem] write_file: {p} (mode={mode!r})")

    # Create parent directories (e.g. agent/notes/deep/file.txt → make notes/deep/)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return {"success": False, "error": f"Could not create parent directories: {e}"}

    try:
        if mode == "append":
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            # Default: overwrite (or create)
            p.write_text(content, encoding="utf-8")

        bytes_written = len(content.encode("utf-8"))
        logger.info(f"[filesystem] write_file OK: {bytes_written} bytes to {p}")
        return {
            "success": True,
            "path": str(p),
            "mode": mode,
            "bytes_written": bytes_written,
        }
    except PermissionError:
        return {"success": False, "error": f"Permission denied: {p}"}
    except Exception as e:
        logger.exception(f"[filesystem] write_file error: {p}")
        return {"success": False, "error": str(e)}


async def list_directory(path: str = ".") -> dict[str, Any]:
    """
    List the contents of a directory.
    Returns type (file/dir), name, and size for each entry, sorted alphabetically.
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
            # Include file size; directories don't have a meaningful size here
            if item.is_file():
                try:
                    entry["size_bytes"] = item.stat().st_size
                except OSError:
                    entry["size_bytes"] = None
            entries.append(entry)

        logger.info(f"[filesystem] list_directory OK: {len(entries)} entries in {p}")
        return {
            "success": True,
            "path": str(p),
            "entries": entries,
            "count": len(entries),
        }
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
