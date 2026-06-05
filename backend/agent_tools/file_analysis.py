"""
agent_tools/file_analysis.py  —  File Analysis Tool (Phase 2)

Provides one tool the agent can use:
  - analyze_file: Inspect a file and return size, line/word/char counts,
                  estimated token count, encoding, extension, and last-modified time.

This tool is entirely read-only (non-destructive).  It is useful for:
  - Understanding how large a file is before reading it into context
  - Estimating whether a file will fit in the Claude context window
  - Detecting encoding issues before attempting to read

Optional dependency: chardet
  If chardet is installed (`pip install chardet`) it is used for encoding
  detection.  If it is not installed the tool falls back to assuming UTF-8.
  chardet is intentionally NOT listed in requirements.txt — it is an optional
  enhancement, not a hard requirement.
"""

import logging
import os
import pathlib
from datetime import datetime, timezone
from typing import Any

from . import register_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional chardet import
# ---------------------------------------------------------------------------

try:
    import chardet as _chardet          # noqa: F401  (used via the name below)
    _CHARDET_AVAILABLE = True
except ImportError:
    _CHARDET_AVAILABLE = False


# ---------------------------------------------------------------------------
# Path helper (mirrors filesystem.py for consistency)
# ---------------------------------------------------------------------------

def _safe_path(path_str: str) -> pathlib.Path:
    """
    Normalise a path string into an absolute pathlib.Path.
    Handles '~', Windows %ENV_VAR% syntax, and relative paths.
    """
    expanded = os.path.expandvars(os.path.expanduser(str(path_str)))
    return pathlib.Path(expanded).resolve()


# ---------------------------------------------------------------------------
# Tool: analyze_file
# ---------------------------------------------------------------------------

async def analyze_file(path: str) -> dict[str, Any]:
    """
    Inspect a file and return detailed metadata about its content.

    Returned fields
    ---------------
    path            — resolved absolute path (string)
    size_bytes      — raw file size in bytes
    size_kb         — size in kilobytes (rounded to 2 dp)
    size_mb         — size in megabytes (rounded to 2 dp)
    line_count      — number of newline-delimited lines
    word_count      — number of whitespace-delimited words
    char_count      — number of Unicode characters
    estimated_tokens— char_count // 4  [approximate — real tokenizers vary;
                       Claude's tokenizer averages ~4 chars/token for English
                       prose, but code and non-Latin text can differ widely]
    encoding        — detected encoding string, or "utf-8 (assumed)" if chardet
                       is not installed
    extension       — file extension including the leading dot (e.g. ".py")
    last_modified   — ISO 8601 timestamp of the last modification time (UTC)
    """
    p = _safe_path(path)
    logger.info(f"[file_analysis] analyze_file: {p}")

    # --- Existence checks ---
    if not p.exists():
        return {"success": False, "error": f"File not found: {p}"}
    if not p.is_file():
        return {"success": False, "error": f"Path exists but is not a file: {p}"}

    try:
        stat = p.stat()
    except PermissionError:
        return {"success": False, "error": f"Permission denied: {p}"}

    size_bytes = stat.st_size

    # --- Encoding detection ---
    # Read raw bytes for chardet; we'll decode them as text afterwards.
    try:
        raw_bytes = p.read_bytes()
    except PermissionError:
        return {"success": False, "error": f"Permission denied reading: {p}"}
    except Exception as e:
        return {"success": False, "error": f"Could not read file: {e}"}

    if _CHARDET_AVAILABLE:
        import chardet
        detection  = chardet.detect(raw_bytes)
        encoding   = detection.get("encoding") or "utf-8"
        # chardet confidence reported for debugging
        confidence = detection.get("confidence", 0.0)
        logger.debug(f"[file_analysis] chardet: {encoding!r} ({confidence:.0%})")
    else:
        encoding = "utf-8 (assumed)"

    # --- Text decoding ---
    # Use the detected encoding; fall back to UTF-8 with replacement on error.
    decode_enc = encoding.replace(" (assumed)", "")  # strip our suffix if present
    try:
        text = raw_bytes.decode(decode_enc, errors="replace")
    except (LookupError, UnicodeDecodeError):
        # Unknown or broken encoding — fall back to UTF-8 with replacement
        text     = raw_bytes.decode("utf-8", errors="replace")
        encoding = "utf-8 (fallback)"

    # --- Text statistics ---
    line_count  = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    word_count  = len(text.split())
    char_count  = len(text)

    # Token estimation:  char_count // 4 is a rough heuristic widely used for
    # English text with Claude's tokenizer.  Code, JSON, and non-Latin scripts
    # can be significantly different — always treat this as an order-of-magnitude
    # guide rather than an exact count.
    estimated_tokens = char_count // 4

    # --- Last modified time ---
    mtime_utc = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    last_modified = mtime_utc.isoformat()

    logger.info(
        f"[file_analysis] analyze_file OK: {size_bytes} bytes, "
        f"{line_count} lines, {estimated_tokens} est. tokens — {p.name}"
    )

    return {
        "success":          True,
        "path":             str(p),
        "size_bytes":       size_bytes,
        "size_kb":          round(size_bytes / 1024,          2),
        "size_mb":          round(size_bytes / (1024 ** 2),   2),
        "line_count":       line_count,
        "word_count":       word_count,
        "char_count":       char_count,
        "estimated_tokens": estimated_tokens,
        "encoding":         encoding,
        "extension":        p.suffix,          # e.g. ".py", ".txt", "" if none
        "last_modified":    last_modified,
    }


# ---------------------------------------------------------------------------
# Registration — call this once at startup from main.py
# ---------------------------------------------------------------------------

def register_file_analysis_tools() -> None:
    """Register analyze_file into the global tool registry."""

    register_tool(
        name="analyze_file",
        description=(
            "Inspect a file and return its size (bytes/KB/MB), "
            "line count, word count, character count, estimated token count, "
            "detected encoding, file extension, and last-modified timestamp. "
            "Use this before reading a large file to check whether it will fit "
            "in the context window."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to analyse.",
                },
            },
            "required": ["path"],
        },
        handler=analyze_file,
        is_destructive=False,
    )
