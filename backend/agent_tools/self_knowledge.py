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
    return Path(__file__).parent.parent.parent / "memory" / "user_profile.json"


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

    logger.info("[self_knowledge] System scan complete.")
    return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_self_knowledge_tools() -> None:
    """Register read_user_profile and scan_system into the global tool registry."""

    register_tool(
        name="read_user_profile",
        description=(
            "Read the user's profile from memory/user_profile.json. "
            "The profile contains the user's name, skills, education, available time, "
            "hardware, constraints, goals, accounts, current projects, and interests. "
            "Use this at the start of any task that requires understanding what the user "
            "can do, what they want to achieve, or what resources they have available. "
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
