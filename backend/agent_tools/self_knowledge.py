"""
agent_tools/self_knowledge.py  —  Phase 3g: Deep Self-Knowledge Tools

Registers two tools that give the agent awareness of who it's working for
and what environment it's running in:

  read_user_profile  — read memory/user_profile.json (non-destructive)
  scan_system        — lightweight scan of installed tools, packages, projects,
                       disk space, and available Ollama models

Both tools are non-destructive and require no user approval.

Design intent:
  For self-directed tasks, agent_core.py injects the profile directly into the
  system prompt (zero round-trips).  These tools are exposed to Claude as well
  so it can re-read the profile mid-task or request a fresh system scan whenever
  it needs up-to-date environment information.
"""

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import psutil

from . import register_tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _profile_path() -> Path:
    """Absolute path to memory/user_profile.json (project root / memory /)."""
    return Path(__file__).resolve().parent.parent.parent / "memory" / "user_profile.json"


def _agent_root() -> Path:
    """Absolute path to the project root (two levels up from backend/)."""
    # backend/agent_tools/self_knowledge.py → backend/ → project root
    return Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Tool: read_user_profile
# ---------------------------------------------------------------------------

async def read_user_profile() -> dict[str, Any]:
    """
    Read and return the contents of memory/user_profile.json.

    The profile contains user-provided information: skills, goals, constraints,
    available hardware, current projects, and interests.  It is maintained
    manually by the user and injected automatically into the system prompt
    for self-directed tasks.

    Returns:
        The parsed profile dict on success, or a dict with an "error" key
        if the file is missing or unreadable.
    """
    path = _profile_path()

    if not path.exists():
        logger.info("[self_knowledge] user_profile.json not found.")
        return {
            "error": (
                "No user profile found. "
                "Ask the user to create memory/user_profile.json with their skills, "
                "goals, constraints, current projects, and other relevant context."
            )
        }

    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
        logger.info("[self_knowledge] User profile loaded successfully.")
        return profile
    except json.JSONDecodeError as e:
        logger.warning(f"[self_knowledge] Failed to parse user_profile.json: {e}")
        return {"error": f"user_profile.json exists but is not valid JSON: {e}"}
    except OSError as e:
        logger.warning(f"[self_knowledge] Could not read user_profile.json: {e}")
        return {"error": f"Could not read user_profile.json: {e}"}


# ---------------------------------------------------------------------------
# Tool: scan_system
# ---------------------------------------------------------------------------

# Common developer tools to probe with shutil.which()
_DEV_TOOLS = [
    "python", "python3", "node", "npm", "git",
    "docker", "pip", "cargo", "go", "java", "ffmpeg", "ollama",
]

# Project indicator filenames to search for (max 2 directory levels deep)
_PROJECT_INDICATORS = [
    "package.json", "requirements.txt", "Cargo.toml",
    "go.mod", "pom.xml",
]
# Glob patterns for Visual Studio project files
_GLOB_INDICATORS = ["*.sln", "*.csproj"]


async def _list_ollama_models(base_url: str) -> list[dict]:
    """
    Query Ollama /api/tags and return [{name, size_gb}].
    Returns [] if Ollama is offline — non-fatal.
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
        return [
            {
                "name":    m.get("name", "unknown"),
                "size_gb": round(m.get("size", 0) / (1024 ** 3), 2),
            }
            for m in data.get("models", [])
        ]
    except Exception as e:
        logger.debug(f"[self_knowledge] Ollama unreachable: {e}")
        return []


def _get_ollama_url() -> str:
    """Read ollama_base_url from config.json, fall back to default."""
    config_path = Path(__file__).parent.parent.parent / "config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f).get("ollama_base_url", "http://localhost:11434")
    except Exception:
        return "http://localhost:11434"


async def scan_system() -> dict[str, Any]:
    """
    Perform a lightweight scan of the user's environment and return a structured
    summary useful for understanding what the agent can work with.

    Fields returned:
        installed_tools  — developer tools found via shutil.which()
                           (only tools that are present on PATH are included)
        python_packages  — first 50 installed pip packages (alphabetical)
        project_files    — project indicator files found up to 2 levels above backend/
        disk_summary     — total and free space on the primary drive
        ollama_models    — available Ollama models (name, size_gb)
    """
    result: dict[str, Any] = {}

    # ── Installed dev tools ───────────────────────────────────────────────
    installed: dict[str, str] = {}
    for tool in _DEV_TOOLS:
        path = shutil.which(tool)
        if path:
            installed[tool] = path
    result["installed_tools"] = installed
    logger.info(f"[self_knowledge] Found {len(installed)} dev tool(s): {list(installed.keys())}")

    # ── Python packages ───────────────────────────────────────────────────
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode == 0:
            packages = json.loads(proc.stdout)
            # Sort alphabetically, then truncate to 50
            packages.sort(key=lambda p: p.get("name", "").lower())
            result["python_packages"] = packages[:50]
            if len(packages) > 50:
                result["python_packages_truncated"] = True
                result["python_packages_total"] = len(packages)
        else:
            result["python_packages"] = []
            result["python_packages_error"] = proc.stderr.strip()
    except Exception as e:
        logger.warning(f"[self_knowledge] pip list failed: {e}")
        result["python_packages"] = []
        result["python_packages_error"] = str(e)

    # ── Project files ─────────────────────────────────────────────────────
    # Scan up to 2 levels deep from the project root for common project indicators.
    project_root = _agent_root()
    found_projects: list[str] = []

    # Exact-name matches
    for depth in range(3):  # depth 0 = root, 1 = one level, 2 = two levels
        pattern = "/".join(["*"] * depth) if depth > 0 else "."
        for indicator in _PROJECT_INDICATORS:
            glob_pattern = f"{'*/' * depth}{indicator}"
            for match in project_root.glob(glob_pattern):
                found_projects.append(str(match.relative_to(project_root)))

    # Glob-pattern matches (*.sln, *.csproj)
    for glob_pat in _GLOB_INDICATORS:
        for match in project_root.glob(glob_pat):
            found_projects.append(str(match.relative_to(project_root)))
        for match in project_root.glob(f"*/{glob_pat}"):
            found_projects.append(str(match.relative_to(project_root)))
        for match in project_root.glob(f"*/*/{glob_pat}"):
            found_projects.append(str(match.relative_to(project_root)))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_projects: list[str] = []
    for p in found_projects:
        if p not in seen:
            seen.add(p)
            unique_projects.append(p)

    result["project_files"] = unique_projects
    logger.info(f"[self_knowledge] Found {len(unique_projects)} project indicator file(s).")

    # ── Disk summary (primary drive) ─────────────────────────────────────
    try:
        # Use the root of the project path as the query target — on Windows
        # this gives the correct drive; on Linux it's always /.
        usage = psutil.disk_usage(str(project_root.anchor))
        result["disk_summary"] = {
            "path":      str(project_root.anchor),
            "total_gb":  round(usage.total / (1024 ** 3), 2),
            "used_gb":   round(usage.used  / (1024 ** 3), 2),
            "free_gb":   round(usage.free  / (1024 ** 3), 2),
            "percent_used": usage.percent,
        }
    except Exception as e:
        logger.warning(f"[self_knowledge] Disk scan failed: {e}")
        result["disk_summary"] = {"error": str(e)}

    # ── Ollama models ─────────────────────────────────────────────────────
    ollama_url = _get_ollama_url()
    result["ollama_models"] = await _list_ollama_models(ollama_url)

    # ── Inference acceleration (iGPU / ipex-llm) ──────────────────────────────────────────────
    igpu_info: dict = {"acceleration": "none", "backend": "cpu"}
    try:
        cfg_path = Path(__file__).resolve().parent.parent.parent / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        igpu_cfg = cfg.get("igpu", {})

        if igpu_cfg.get("enabled"):
            igpu_info = {
                "acceleration": "Intel Arc iGPU via ipex-llm",
                "backend": "SYCL/XPU",
                "model_preprocessing": igpu_cfg.get("model_small", "unknown"),
                "model_agent_loop": igpu_cfg.get("model_large", "unknown"),
                "context_size_tokens": igpu_cfg.get("context_size", 2048),
                "max_models_in_memory": igpu_cfg.get("max_loaded_models", 1),
                "ollama_dir": igpu_cfg.get("ollama_dir", ""),
            }

        # Query Ollama /api/ps for live loaded models
        try:
            ps_resp = httpx.get(f"{ollama_url}/api/ps", timeout=3.0)
            if ps_resp.status_code == 200:
                ps_models = ps_resp.json().get("models", [])
                igpu_info["currently_loaded_models"] = [
                    {
                        "name": m.get("name", "unknown"),
                        "size_mb": round(m.get("size", 0) / 1_000_000, 1),
                        "expires_at": m.get("expires_at", "never (keep_alive=-1)"),
                    }
                    for m in ps_models
                ]
        except Exception:
            igpu_info["currently_loaded_models"] = "unavailable (Ollama not responding)"

    except Exception as exc:
        igpu_info["error"] = str(exc)

    result["inference_acceleration"] = igpu_info

    logger.info("[self_knowledge] System scan complete.")
    return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_self_knowledge_tools() -> None:
    """Register self-knowledge tools into the global tool registry.

    Phase 3g: read_user_profile, scan_system
    Phase 8:  get_context_usage, analyze_performance
    """

    register_tool(
        name="get_context_usage",
        description=(
            "Estimate how much of the Claude API context window is currently in use. "
            "Returns: model name, estimated tokens used, context limit, percent used, "
            "a warning level ('low' / 'medium' / 'high' / 'critical'), the current "
            "history turn count, and a recommendation for what to do. "
            "Use this periodically during very long tasks to check context pressure. "
            "If warning_level is 'high' or 'critical', summarize completed steps and "
            "ask the user if they want to clear old turns before continuing."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=get_context_usage,
        is_destructive=False,
    )

    register_tool(
        name="analyze_performance",
        description=(
            "Analyze the agent's task history from long_term.json to identify failure "
            "patterns, most-used tools, slowest tasks, and generate improvement "
            "suggestions using the local LLM (no Claude API cost). "
            "Returns: total task count, success rate, failed task count, "
            "average duration, top tools by usage, and a numbered list of "
            "AI-generated improvement suggestions. "
            "Requires at least 5 completed tasks in history. "
            "Use this periodically to identify what's failing or taking too long, "
            "and to get actionable suggestions for improving reliability."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=analyze_performance,
        is_destructive=False,
    )

    register_tool(
        name="read_user_profile",
        description=(
            "Read the user's profile from memory/user_profile.json. "
            "NOTE: The user profile is already injected into your system prompt at the "
            "start of every conversation — you do NOT need to call this tool to access it. "
            "Only call this tool if you specifically need to re-read the raw file, for "
            "example after the user has updated their profile mid-session. "
            "Returns the full profile as a dict, or an error message if the file is missing."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=read_user_profile,
        is_destructive=False,
    )

    register_tool(
        name="scan_system",
        description=(
            "Perform a lightweight scan of the user's development environment. "
            "Returns: installed developer tools (python, node, git, docker, etc.) with paths; "
            "installed Python packages (first 50, alphabetical); "
            "project indicator files found near the agent root (package.json, requirements.txt, etc.); "
            "disk usage summary for the primary drive; "
            "and available Ollama models with sizes. "
            "Use this when you need to know what tools are available, what projects exist, "
            "or before writing code that depends on specific software being installed."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=scan_system,
        is_destructive=False,
    )


# ---------------------------------------------------------------------------
# Tool: get_context_usage  (Phase 8)
# ---------------------------------------------------------------------------

async def get_context_usage() -> dict:
    """
    Estimate how much of the Claude API context window is currently in use.
    Uses a simple character-count heuristic (1 token ≈ 4 chars) against
    the known limits for each model.

    Returns a warning level so the agent can act before hitting hard limits:
      'low'      < 40% used  — plenty of room
      'medium'   40–70%      — consider summarizing older turns
      'high'     70–90%      — summarize soon
      'critical' > 90%       — summarize or the next call will fail
    """
    import json as _json
    from memory.context import load_history

    MODEL_LIMITS: dict[str, int] = {
        "claude-haiku-4-5":  200_000,
        "claude-sonnet-4-6": 200_000,
    }

    history = load_history()
    history_chars = sum(len(str(h.get("content", ""))) for h in history)

    # Read config for current model
    cfg_path = Path(__file__).resolve().parent.parent.parent / "config.json"
    try:
        config = _json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    except Exception:
        config = {}

    model = config.get("llm", {}).get("primary", "claude-haiku-4-5")
    limit_tokens = MODEL_LIMITS.get(model, 200_000)
    limit_chars  = limit_tokens * 4

    estimated_tokens = history_chars // 4
    pct = round(estimated_tokens / limit_tokens * 100, 1)

    if pct < 40:
        level = "low"
    elif pct < 70:
        level = "medium"
    elif pct < 90:
        level = "high"
    else:
        level = "critical"

    recommendations = {
        "low":      "Context is fine — no action needed.",
        "medium":   "Consider summarizing older conversation turns if this is a long session.",
        "high":     "Summarize old turns soon to avoid context limit errors.",
        "critical": "Summarize immediately — next API call may fail due to context overflow.",
    }

    logger.info(
        f"[self_knowledge] Context usage: {pct}% ({estimated_tokens} tokens), level={level}"
    )

    return {
        "success": True,
        "model": model,
        "estimated_tokens_used": estimated_tokens,
        "limit_tokens": limit_tokens,
        "percent_used": pct,
        "warning_level": level,
        "history_turns": len(history),
        "recommendation": recommendations[level],
    }


# ---------------------------------------------------------------------------
# Tool: analyze_performance  (Phase 8)
# ---------------------------------------------------------------------------

async def analyze_performance() -> dict:
    """
    Analyze the agent's task history to identify failure patterns,
    most-used tools, slowest tasks, and generate improvement suggestions.

    Uses only long_term.json data — no Claude API call needed.
    The suggestions are generated by the local LLM so this costs nothing.
    """
    from memory.long_term import load as load_lt
    from .local_llm import local_llm_call

    data = load_lt()
    tasks = data.get("tasks", [])
    if len(tasks) < 5:
        return {
            "success": False,
            "error": "Not enough task history yet (need at least 5 tasks).",
            "current_task_count": len(tasks),
        }

    failed  = [t for t in tasks if t.get("outcome") != "success"]
    slow    = sorted(tasks, key=lambda t: -t.get("duration_seconds", 0))[:3]
    tool_counts: dict[str, int] = {}
    for t in tasks:
        for tool in t.get("tools_used", []):
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
    top_tools = sorted(tool_counts.items(), key=lambda x: -x[1])[:8]

    summary = (
        f"Total tasks: {len(tasks)}\n"
        f"Failed tasks ({len(failed)}): {[t['goal'][:60] for t in failed[-5:]]}\n"
        f"Slowest tasks: {[(t['goal'][:40], t.get('duration_seconds', 0)) for t in slow]}\n"
        f"Most used tools: {top_tools}\n"
    )

    prompt = (
        f"An AI agent has this performance history:\n{summary}\n\n"
        "Based on this data, write 3-5 specific, actionable improvement suggestions "
        "for making this agent more reliable and efficient. "
        "Focus on patterns in failures and slow tasks. Be specific and practical. "
        "Format as a numbered list."
    )

    logger.info("[self_knowledge] Requesting improvement suggestions from local LLM…")
    suggestions = await local_llm_call(prompt, "qwen2.5:14b", base_url="http://localhost:11434")
    if not suggestions:
        suggestions = "Could not generate suggestions — local LLM unavailable."

    avg_duration = round(
        sum(t.get("duration_seconds", 0) for t in tasks) / len(tasks)
    )

    return {
        "success": True,
        "total_tasks": len(tasks),
        "success_rate": round((len(tasks) - len(failed)) / len(tasks) * 100, 1),
        "failed_count": len(failed),
        "avg_duration_seconds": avg_duration,
        "top_tools": [{"tool": k, "uses": v} for k, v in top_tools],
        "improvement_suggestions": suggestions,
    }
