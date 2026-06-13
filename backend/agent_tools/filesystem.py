"""
agent_tools/filesystem.py  —  Filesystem Tools

Provides four tools the agent can use:
  - read_file:       Read the text content of a file.
  - write_file:      Write or append text to a file. Flagged DESTRUCTIVE.
  - list_directory:  List files and subdirectories at a path.
  - list_outputs:    List files the agent has previously generated in outputs/.
                     Useful for avoiding duplicate work across sessions.

All paths go through _safe_path() which handles Windows drive letters, ~ expansion,
and environment variables. Every operation is logged for audit purposes.

Phase 2 addition — folder tree broadcast:
  After a successful write_file or list_directory call, a compact folder tree
  is generated from the configured tree_root (config.json → "tree_root", default ".")
  and returned as an extra "tree" key in the result dict. agent_core.py detects
  this key and broadcasts a tree_update WebSocket event to the frontend.

Improvement 2 addition — list_outputs:
  New non-destructive tool that reads the outputs/ directory and returns a
  structured list of generated files. The agent should call this at the start
  of any task that might produce files to avoid regenerating existing reports.
"""

import json
import os
import logging
import pathlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import register_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Improvement 2: File content cache
# Keyed by absolute path string → (mtime: float, content: str).
# Avoids re-reading unchanged files during long multi-file tasks — the most
# common case being agent_core reading the same source file twice in a row
# (once to understand it, once to verify the patch).
# ---------------------------------------------------------------------------

_file_cache: dict[str, tuple[float, str]] = {}  # path → (mtime, content)
_FILE_CACHE_MAX = 20  # max entries; evict oldest (insertion-order) on overflow

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


def _get_outputs_dir() -> Path:
    """
    Read outputs_dir from config.json (default "outputs") and resolve it
    relative to the project root — same pattern as browser.py uses.
    Called lazily so runtime config changes take effect without a restart.
    """
    config_path = Path(__file__).parent.parent.parent / "config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        raw = cfg.get("outputs_dir", "outputs")
    except Exception:
        raw = "outputs"
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
    Handles: '~', '%USERPROFILE%', environment variables, forward/backslashes.

    IMPORTANT: relative paths are resolved relative to the PROJECT ROOT
    (the directory containing config.json), NOT the CWD. The server runs
    from backend/, so resolving against CWD would place files like
    'outputs/foo.py' inside backend/outputs/ instead of the intended
    project-root outputs/.
    """
    expanded = os.path.expandvars(os.path.expanduser(str(path_str)))
    p = pathlib.Path(expanded)
    if not p.is_absolute():
        # Anchor to project root: this file lives at backend/agent_tools/filesystem.py
        # so three .parent calls reach the project root.
        project_root = pathlib.Path(__file__).parent.parent.parent
        p = project_root / p
    return p.resolve()


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
        mtime = p.stat().st_mtime

        # ── Cache lookup ──────────────────────────────────────────────────
        cached = _file_cache.get(str(p))
        if cached and cached[0] == mtime:
            logger.debug(f"[filesystem] Cache hit: {p.name}")
            return {
                "success":    True,
                "path":       str(p),
                "content":    cached[1],
                "size_bytes": len(cached[1].encode("utf-8")),
                "cached":     True,
            }

        # ── Cache miss: read from disk ────────────────────────────────────
        content = p.read_text(encoding="utf-8", errors="replace")
        size = p.stat().st_size
        logger.info(f"[filesystem] read_file OK: {size} bytes from {p}")

        # Store in cache — evict oldest entry (dict preserves insertion order
        # in Python 3.7+, so next(iter(...)) is the oldest key).
        if len(_file_cache) >= _FILE_CACHE_MAX:
            oldest_key = next(iter(_file_cache))
            del _file_cache[oldest_key]
            logger.debug(f"[filesystem] Cache evicted oldest entry: {oldest_key}")
        _file_cache[str(p)] = (mtime, content)

        return {
            "success":    True,
            "path":       str(p),
            "content":    content,
            "size_bytes": size,
            "cached":     False,
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

        # Invalidate cache entry — file content has changed.
        _file_cache.pop(str(p), None)

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


async def list_outputs() -> dict[str, Any]:
    """
    List files previously generated by the agent in the outputs/ directory.

    Walks up to 2 levels deep and returns metadata for every file found.
    The agent should call this at the start of any task that might produce
    files to check what already exists and avoid duplicating work.

    If a report was already generated in a previous session, read it with
    read_file before generating a new one.

    Returns:
        {
            "success":     True,
            "outputs_dir": str,          # absolute path to the outputs directory
            "files": [
                {
                    "name":         str,   # filename only
                    "path":         str,   # path relative to project root
                    "size_bytes":   int,
                    "modified":     str,   # ISO 8601 UTC timestamp
                },
                ...
            ],
            "count": int,
            "note": str | None,          # set if the directory is missing or empty
        }
    """
    outputs_dir = _get_outputs_dir()
    logger.info(f"[filesystem] list_outputs: {outputs_dir}")

    if not outputs_dir.exists():
        logger.info("[filesystem] list_outputs: outputs directory does not exist yet.")
        return {
            "success":     True,
            "outputs_dir": str(outputs_dir),
            "files":       [],
            "count":       0,
            "note":        "Outputs directory does not exist yet — no files generated.",
        }

    # Project root is needed to compute relative paths for display
    project_root = Path(__file__).parent.parent.parent

    files: list[dict[str, Any]] = []
    try:
        # Walk up to 2 levels deep: outputs/ and outputs/subdir/
        for item in sorted(outputs_dir.rglob("*"), key=lambda x: x.name.lower()):
            # Skip hidden files and directories
            if any(part.startswith(".") for part in item.parts):
                continue

            # Depth check: count parts relative to outputs_dir
            try:
                rel_to_outputs = item.relative_to(outputs_dir)
            except ValueError:
                continue
            depth = len(rel_to_outputs.parts)
            if depth > 2:
                continue

            if not item.is_file():
                continue  # only return files, not subdirectory entries

            try:
                stat = item.stat()
                modified_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                modified_iso = modified_dt.isoformat()
                size_bytes = stat.st_size
            except OSError:
                modified_iso = ""
                size_bytes   = 0

            # Relative path from project root (friendlier for the agent to use
            # as the `path` argument in read_file)
            try:
                display_path = str(item.relative_to(project_root))
            except ValueError:
                display_path = str(item)

            files.append({
                "name":       item.name,
                "path":       display_path,
                "size_bytes": size_bytes,
                "modified":   modified_iso,
            })

    except Exception as e:
        logger.exception(f"[filesystem] list_outputs error scanning {outputs_dir}")
        return {"success": False, "error": str(e)}

    note: str | None = None
    if not files:
        note = "Outputs directory exists but contains no files."

    logger.info(f"[filesystem] list_outputs OK: {len(files)} files in {outputs_dir}")
    return {
        "success":     True,
        "outputs_dir": str(outputs_dir),
        "files":       files,
        "count":       len(files),
        "note":        note,
    }


async def patch_file(path: str, start_line: int, end_line: int, new_content: str) -> dict[str, Any]:
    """
    Apply a targeted edit to a specific line range in a file.

    Replaces lines start_line..end_line (1-indexed, inclusive) with new_content.
    All lines outside the specified range are preserved exactly.

    This is the preferred tool for modifying an existing file — it avoids
    rewriting lines you don't intend to change, which reduces errors and makes
    diffs easy to review.

    Workflow:
        1. Call analyze_file to see the total line count.
        2. Call read_file to identify the exact lines you want to change.
        3. Call patch_file with the line range and the replacement text.

    Args:
        path:        Path to the file to edit.
        start_line:  First line to replace (1-indexed, inclusive).
        end_line:    Last line to replace (1-indexed, inclusive).
                     Must be >= start_line and <= total line count.
        new_content: Replacement text. May contain any number of lines.
                     Does not need to contain a trailing newline — one will
                     be added automatically if the original file used them.

    Returns:
        {
            "success":       bool,
            "path":          str,
            "lines_replaced": int,   # end_line - start_line + 1
            "lines_written":  int,   # number of lines in new_content
        }

    DESTRUCTIVE: modifies the file in place.
    """
    p = _safe_path(path)
    logger.info(f"[filesystem] patch_file: {p} lines {start_line}-{end_line}")

    if not p.exists():
        return {"success": False, "error": f"File not found: {p}"}
    if not p.is_file():
        return {"success": False, "error": f"Path exists but is not a file: {p}"}

    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return {"success": False, "error": f"Permission denied reading: {p}"}
    except Exception as e:
        return {"success": False, "error": f"Could not read file: {e}"}

    # Split preserving line endings so we can reconstruct faithfully.
    # splitlines(keepends=True) keeps "\n", "\r\n", "\r" attached to each line.
    lines = raw.splitlines(keepends=True)
    total = len(lines)

    # ---- Validate range -------------------------------------------------
    if start_line < 1:
        return {"success": False, "error": f"start_line must be >= 1, got {start_line}"}
    if end_line < start_line:
        return {
            "success": False,
            "error": f"end_line ({end_line}) must be >= start_line ({start_line})",
        }
    if end_line > total:
        return {
            "success": False,
            "error": (
                f"end_line ({end_line}) exceeds file length ({total} lines). "
                f"Use analyze_file to check the line count first."
            ),
        }

    # ---- Determine the line ending used by the file (default \n) --------
    # Peek at the first line that has an ending; fall back to \n.
    file_eol = "\n"
    for ln in lines:
        if ln.endswith("\r\n"):
            file_eol = "\r\n"
            break
        if ln.endswith("\n"):
            file_eol = "\n"
            break
        if ln.endswith("\r"):
            file_eol = "\r"
            break

    # ---- Build replacement lines ----------------------------------------
    # Split new_content into lines, then ensure every line ends with the
    # file's native line ending (unless it's the very last line of the file,
    # which we handle below).
    replacement_raw = new_content.splitlines()
    replacement_lines = [ln + file_eol for ln in replacement_raw]

    # If the patch covers the last line of the file AND the original last line
    # had no trailing newline, strip the EOL from the last replacement line too.
    if end_line == total and not lines[-1].endswith(("\n", "\r")):
        if replacement_lines:
            replacement_lines[-1] = replacement_lines[-1].rstrip("\r\n")

    # ---- Splice ---------------------------------------------------------
    # lines is 0-indexed; start_line/end_line are 1-indexed.
    before = lines[: start_line - 1]
    after  = lines[end_line:]
    new_lines = before + replacement_lines + after

    try:
        p.write_text("".join(new_lines), encoding="utf-8")
    except PermissionError:
        return {"success": False, "error": f"Permission denied writing: {p}"}
    except Exception as e:
        return {"success": False, "error": f"Could not write file: {e}"}

    # Invalidate cache entry — file content has changed.
    _file_cache.pop(str(p), None)

    lines_replaced = end_line - start_line + 1
    lines_written  = len(replacement_raw)
    logger.info(
        f"[filesystem] patch_file OK: replaced {lines_replaced} lines "
        f"with {lines_written} lines in {p}"
    )
    return {
        "success":        True,
        "path":           str(p),
        "lines_replaced": lines_replaced,
        "lines_written":  lines_written,
    }


# ---------------------------------------------------------------------------
# Registration — call this once at startup from main.py
# ---------------------------------------------------------------------------

def register_all() -> None:
    """Register all filesystem tools into the global tool registry."""

    register_tool(
        name="patch_file",
        description=(
            "Apply a targeted edit to a specific line range in a file. "
            "Use this instead of write_file when modifying part of an existing file — "
            "it preserves all other content. "
            "start_line and end_line are 1-indexed and inclusive. "
            "Use analyze_file first to see line counts, then read_file to find the exact "
            "lines to change, then patch_file to apply the edit."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to replace (1-indexed, inclusive).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to replace (1-indexed, inclusive). Must be >= start_line.",
                },
                "new_content": {
                    "type": "string",
                    "description": (
                        "Replacement text for the specified line range. "
                        "May contain multiple lines. No trailing newline needed."
                    ),
                },
            },
            "required": ["path", "start_line", "end_line", "new_content"],
        },
        handler=patch_file,
        is_destructive=True,  # modifies an existing file in place
    )

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

    register_tool(
        name="list_outputs",
        description=(
            "Lists files previously generated by the agent in the outputs/ directory, "
            "useful for avoiding duplicate work across sessions. "
            "Returns name, relative path, size, and last-modified timestamp for each file. "
            "Call this at the start of any task that might produce files — if a report or "
            "script already exists, read it with read_file before generating a new one."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=list_outputs,
        is_destructive=False,
    )
