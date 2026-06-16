"""
agent_tools/media_tool.py  —  Phase 9a: ffmpeg Media Processing Tools

Wraps common ffmpeg/ffprobe operations for video and audio processing.
All output goes to outputs/media/ relative to the project root.

Tools registered:
    get_media_info(input_path)                        — non-destructive metadata read
    convert_video(input_path, output_format, quality) — destructive: re-encode video
    extract_audio(input_path, output_format, bitrate) — destructive: extract audio track
    trim_clip(input_path, start_seconds, end_seconds) — destructive: cut clip
    merge_clips(input_paths, output_filename)         — destructive: concatenate clips

All tools return {"success": False, "error": "..."} if ffmpeg is not found in PATH.
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path

from agent_tools import register_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Project root is three levels up from this file:
# backend/agent_tools/media_tool.py → backend/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MEDIA_OUT = _PROJECT_ROOT / "outputs" / "media"


def _ffmpeg_available() -> bool:
    """Return True if ffmpeg is on PATH."""
    return shutil.which("ffmpeg") is not None


def _ffprobe_available() -> bool:
    """Return True if ffprobe is on PATH."""
    return shutil.which("ffprobe") is not None


def _ensure_media_dir() -> None:
    """Create outputs/media/ if it does not already exist."""
    _MEDIA_OUT.mkdir(parents=True, exist_ok=True)


def _missing_ffmpeg() -> dict:
    return {
        "success": False,
        "error": (
            "ffmpeg not found. Install from https://ffmpeg.org/download.html "
            "and ensure it is in PATH."
        ),
    }


def _run(cmd: list[str]) -> tuple[int, str, str]:
    """
    Run a subprocess command and return (returncode, stdout, stderr).
    Timeout is 300 s — large files may need time to encode.
    """
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def get_media_info(input_path: str) -> dict:
    """
    Get metadata for a video or audio file using ffprobe.

    Returns duration, resolution, framerate, codec, and file size.
    Non-destructive — no output file is created.
    """
    if not _ffprobe_available():
        return {
            "success": False,
            "error": (
                "ffprobe not found. Install ffmpeg (ffprobe is bundled with it) "
                "from https://ffmpeg.org/download.html."
            ),
        }

    src = Path(input_path)
    if not src.exists():
        return {"success": False, "error": f"File not found: {input_path}"}

    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(src),
    ]

    try:
        rc, stdout, stderr = _run(cmd)
        if rc != 0:
            return {"success": False, "error": f"ffprobe failed: {stderr.strip()}"}

        data = json.loads(stdout)
        fmt = data.get("format", {})
        streams = data.get("streams", [])

        # Pick the first video stream for width/height/fps
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        fps_raw = video_stream.get("r_frame_rate", "0/1")
        try:
            num, den = fps_raw.split("/")
            fps = round(int(num) / int(den), 2) if int(den) != 0 else 0.0
        except Exception:
            fps = 0.0

        size_bytes = int(fmt.get("size", 0))
        return {
            "success": True,
            "duration_seconds": round(float(fmt.get("duration", 0)), 2),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": fps,
            "codec": video_stream.get("codec_name", "unknown"),
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "format": fmt.get("format_name", "unknown"),
        }
    except json.JSONDecodeError:
        return {"success": False, "error": "ffprobe returned non-JSON output."}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "ffprobe timed out after 300 s."}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def convert_video(
    input_path: str,
    output_format: str = "mp4",
    quality: str = "medium",
) -> dict:
    """
    Re-encode a video file to the specified format using libx264 CRF encoding.

    quality levels:
        low    → CRF 28  (smaller file, lower quality)
        medium → CRF 23  (balanced, default)
        high   → CRF 18  (larger file, higher quality)

    Output is saved to outputs/media/{stem}.{output_format}.
    """
    if not _ffmpeg_available():
        return _missing_ffmpeg()

    src = Path(input_path)
    if not src.exists():
        return {"success": False, "error": f"File not found: {input_path}"}

    _ensure_media_dir()

    crf_map = {"low": 28, "medium": 23, "high": 18}
    crf = crf_map.get(quality.lower(), 23)

    output_path = _MEDIA_OUT / f"{src.stem}.{output_format.lstrip('.')}"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-crf", str(crf),
        str(output_path),
    ]

    try:
        # Run ffprobe first to get original duration for the return value
        dur_result = await get_media_info(input_path)
        duration = dur_result.get("duration_seconds", 0.0) if dur_result.get("success") else 0.0

        rc, stdout, stderr = _run(cmd)
        if rc != 0:
            return {"success": False, "error": f"ffmpeg failed: {stderr[-500:].strip()}"}

        return {
            "success": True,
            "output_path": str(output_path),
            "duration_seconds": duration,
            "quality": quality,
            "crf": crf,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "ffmpeg timed out after 300 s."}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def extract_audio(
    input_path: str,
    output_format: str = "mp3",
    bitrate: str = "192k",
) -> dict:
    """
    Extract the audio track from a video or audio file.

    Output is saved to outputs/media/{stem}.{output_format} with no video stream.
    """
    if not _ffmpeg_available():
        return _missing_ffmpeg()

    src = Path(input_path)
    if not src.exists():
        return {"success": False, "error": f"File not found: {input_path}"}

    _ensure_media_dir()

    output_path = _MEDIA_OUT / f"{src.stem}.{output_format.lstrip('.')}"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vn",                # no video
        "-ab", bitrate,
        str(output_path),
    ]

    try:
        rc, stdout, stderr = _run(cmd)
        if rc != 0:
            return {"success": False, "error": f"ffmpeg failed: {stderr[-500:].strip()}"}

        return {
            "success": True,
            "output_path": str(output_path),
            "bitrate": bitrate,
            "format": output_format,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "ffmpeg timed out after 300 s."}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def trim_clip(
    input_path: str,
    start_seconds: float,
    end_seconds: float,
) -> dict:
    """
    Cut a clip from start_seconds to end_seconds without re-encoding (stream copy).

    Output is saved to outputs/media/{stem}_trim_{start}_{end}.{ext}.
    Stream copy is fast but exact cut points depend on keyframes.
    """
    if not _ffmpeg_available():
        return _missing_ffmpeg()

    src = Path(input_path)
    if not src.exists():
        return {"success": False, "error": f"File not found: {input_path}"}

    if end_seconds <= start_seconds:
        return {
            "success": False,
            "error": f"end_seconds ({end_seconds}) must be greater than start_seconds ({start_seconds}).",
        }

    _ensure_media_dir()

    # Build output filename with trim bounds embedded
    s = str(start_seconds).replace(".", "_")
    e = str(end_seconds).replace(".", "_")
    output_path = _MEDIA_OUT / f"{src.stem}_trim_{s}_{e}{src.suffix}"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ss", str(start_seconds),
        "-to", str(end_seconds),
        "-c", "copy",
        str(output_path),
    ]

    try:
        rc, stdout, stderr = _run(cmd)
        if rc != 0:
            return {"success": False, "error": f"ffmpeg failed: {stderr[-500:].strip()}"}

        duration = round(end_seconds - start_seconds, 2)
        return {
            "success": True,
            "output_path": str(output_path),
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "duration_seconds": duration,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "ffmpeg timed out after 300 s."}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def merge_clips(
    input_paths: str,
    output_filename: str = "merged.mp4",
) -> dict:
    """
    Concatenate multiple video/audio files into one using the concat demuxer.

    input_paths: comma-separated list of file paths.
    All clips must have the same codec, resolution, and framerate for stream copy
    to work correctly.  Output is saved to outputs/media/{output_filename}.
    """
    if not _ffmpeg_available():
        return _missing_ffmpeg()

    paths = [p.strip() for p in input_paths.split(",") if p.strip()]
    if len(paths) < 2:
        return {
            "success": False,
            "error": "merge_clips requires at least 2 input paths (comma-separated).",
        }

    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        return {"success": False, "error": f"Files not found: {', '.join(missing)}"}

    _ensure_media_dir()

    # Write a temporary concat list file that ffmpeg can read
    concat_file = _MEDIA_OUT / "_concat_list.txt"
    try:
        concat_file.write_text(
            "\n".join(f"file '{Path(p).resolve()}'" for p in paths),
            encoding="utf-8",
        )

        output_path = _MEDIA_OUT / output_filename
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output_path),
        ]

        rc, stdout, stderr = _run(cmd)
        if rc != 0:
            return {"success": False, "error": f"ffmpeg failed: {stderr[-500:].strip()}"}

        return {
            "success": True,
            "output_path": str(output_path),
            "clip_count": len(paths),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "ffmpeg timed out after 300 s."}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        # Always clean up the temp concat list
        try:
            concat_file.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_media_tools() -> None:
    """Register all media tools into the agent tool registry."""

    register_tool(
        name="get_media_info",
        description=(
            "Get metadata for a video or audio file (duration, resolution, fps, codec, "
            "file size). Uses ffprobe — non-destructive, no output file created. "
            "Requires ffmpeg/ffprobe installed and in PATH."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to the video or audio file.",
                },
            },
            "required": ["input_path"],
        },
        handler=get_media_info,
        destructive=False,
    )

    register_tool(
        name="convert_video",
        description=(
            "Re-encode a video to a different format using libx264 CRF encoding. "
            "Output saved to outputs/media/. quality: 'low' (CRF 28), 'medium' (CRF 23), "
            "'high' (CRF 18). Requires ffmpeg in PATH."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to the input video file.",
                },
                "output_format": {
                    "type": "string",
                    "description": "Output container format, e.g. 'mp4', 'mkv', 'webm'. Default: 'mp4'.",
                    "default": "mp4",
                },
                "quality": {
                    "type": "string",
                    "description": "Encoding quality: 'low', 'medium', or 'high'. Default: 'medium'.",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
            },
            "required": ["input_path"],
        },
        handler=convert_video,
        destructive=True,
    )

    register_tool(
        name="extract_audio",
        description=(
            "Extract the audio track from a video or audio file. No video stream in output. "
            "Output saved to outputs/media/. Requires ffmpeg in PATH."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to the input video or audio file.",
                },
                "output_format": {
                    "type": "string",
                    "description": "Output audio format, e.g. 'mp3', 'aac', 'flac', 'wav'. Default: 'mp3'.",
                    "default": "mp3",
                },
                "bitrate": {
                    "type": "string",
                    "description": "Audio bitrate, e.g. '128k', '192k', '320k'. Default: '192k'.",
                    "default": "192k",
                },
            },
            "required": ["input_path"],
        },
        handler=extract_audio,
        destructive=True,
    )

    register_tool(
        name="trim_clip",
        description=(
            "Cut a clip from start_seconds to end_seconds using stream copy (no re-encoding, fast). "
            "Output saved to outputs/media/{name}_trim_{start}_{end}.{ext}. "
            "Exact cut points depend on keyframe positions. Requires ffmpeg in PATH."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to the input video file.",
                },
                "start_seconds": {
                    "type": "number",
                    "description": "Start time in seconds (float).",
                },
                "end_seconds": {
                    "type": "number",
                    "description": "End time in seconds (float). Must be > start_seconds.",
                },
            },
            "required": ["input_path", "start_seconds", "end_seconds"],
        },
        handler=trim_clip,
        destructive=True,
    )

    register_tool(
        name="merge_clips",
        description=(
            "Concatenate multiple video/audio clips into one file using stream copy (fast). "
            "All clips must share the same codec, resolution, and framerate. "
            "input_paths is a comma-separated list of file paths. "
            "Output saved to outputs/media/{output_filename}. Requires ffmpeg in PATH."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "input_paths": {
                    "type": "string",
                    "description": "Comma-separated list of input file paths, in order.",
                },
                "output_filename": {
                    "type": "string",
                    "description": "Output filename (e.g. 'merged.mp4'). Default: 'merged.mp4'.",
                    "default": "merged.mp4",
                },
            },
            "required": ["input_paths"],
        },
        handler=merge_clips,
        destructive=True,
    )

    logger.info(
        "[media] Registered tools: get_media_info, convert_video, "
        "extract_audio, trim_clip, merge_clips"
    )
