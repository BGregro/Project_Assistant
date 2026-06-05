"""
agent_tools/system_info.py  —  System Information Tool (Phase 2)

Provides one tool the agent can use:
  - get_system_info: Returns CPU, RAM, disk, Ollama models, and platform info.

This tool is entirely read-only (non-destructive).  It is useful for:
  - Diagnosing performance issues ("why is my agent slow?")
  - Checking which Ollama models are currently available before switching
  - Giving the agent situational awareness about the host machine

Requires: psutil>=5.9.0  (add to requirements.txt)
"""

import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import httpx
import psutil  # pip install psutil>=5.9.0

from . import register_tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config access — reads ollama_base_url from config.json at call time
# (same lazy-load pattern used by filesystem.py for tree_root)
# ---------------------------------------------------------------------------

def _get_ollama_url() -> str:
    """
    Read ollama_base_url from config.json.
    Falls back to the default local address if the file cannot be read.
    """
    config_path = Path(__file__).parent.parent.parent / "config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("ollama_base_url", "http://localhost:11434")
    except Exception:
        return "http://localhost:11434"


# ---------------------------------------------------------------------------
# Ollama model lister
# ---------------------------------------------------------------------------

async def _list_ollama_models(base_url: str) -> list[dict]:
    """
    Query the Ollama /api/tags endpoint and return a list of available models.

    Each item in the list has:
        name      — model identifier (e.g. "qwen2.5:7b")
        size_gb   — model size in GB (rounded to 2 decimal places)

    Returns an empty list if Ollama is offline or unreachable — this is
    treated as a non-fatal condition so the rest of get_system_info still works.
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
        models = []
        for m in data.get("models", []):
            size_bytes = m.get("size", 0)
            models.append({
                "name":    m.get("name", "unknown"),
                "size_gb": round(size_bytes / (1024 ** 3), 2),
            })
        return models
    except Exception as e:
        # Ollama offline or network error — return empty list, log at DEBUG
        logger.debug(f"[system_info] Ollama unreachable at {base_url}: {e}")
        return []


# ---------------------------------------------------------------------------
# Tool: get_system_info
# ---------------------------------------------------------------------------

async def get_system_info() -> dict[str, Any]:
    """
    Return a snapshot of the host machine's current resource usage and
    available Ollama models.

    CPU:
        percent_used  — current CPU utilisation across all cores (%)
        physical_cores — number of physical CPU cores
        logical_cores  — number of logical CPU cores (includes hyperthreading)

    RAM:
        total_gb, used_gb, free_gb, percent_used

    Disk:
        List of mounted drives, each with:
            path, total_gb, used_gb, free_gb, percent_used

    Ollama:
        List of currently available models: name, size_gb.
        Empty list if Ollama is offline.

    Platform:
        os      — OS description (e.g. "Windows 11", "Linux 6.x")
        python  — Python version string
    """
    ollama_url = _get_ollama_url()

    # --- CPU ---
    # interval=0.1 gives a short blocking sample; without an interval
    # psutil returns 0.0 on the first call (no reference point yet).
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_info = {
        "percent_used":   cpu_percent,
        "physical_cores": psutil.cpu_count(logical=False) or 0,
        "logical_cores":  psutil.cpu_count(logical=True)  or 0,
    }

    # --- RAM ---
    vm = psutil.virtual_memory()
    ram_info = {
        "total_gb":    round(vm.total   / (1024 ** 3), 2),
        "used_gb":     round(vm.used    / (1024 ** 3), 2),
        "free_gb":     round(vm.available / (1024 ** 3), 2),  # available ≈ usable free
        "percent_used": vm.percent,
    }

    # --- Disk ---
    disk_list = []
    for part in psutil.disk_partitions(all=False):
        # Skip pseudo-filesystems (e.g. /proc, /sys, /dev on Linux)
        # and optical drives that may have no media inserted
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except (PermissionError, OSError):
            continue
        disk_list.append({
            "path":         part.mountpoint,
            "total_gb":     round(usage.total / (1024 ** 3), 2),
            "used_gb":      round(usage.used  / (1024 ** 3), 2),
            "free_gb":      round(usage.free  / (1024 ** 3), 2),
            "percent_used": usage.percent,
        })

    # --- Ollama ---
    ollama_models = await _list_ollama_models(ollama_url)

    # --- Platform ---
    platform_info = {
        "os":     platform.platform(),
        "python": sys.version.split()[0],  # e.g. "3.12.3"
    }

    logger.info(
        f"[system_info] OK — CPU {cpu_percent}%, "
        f"RAM {ram_info['percent_used']}%, "
        f"{len(ollama_models)} Ollama model(s)"
    )

    return {
        "cpu":      cpu_info,
        "ram":      ram_info,
        "disk":     disk_list,
        "ollama":   ollama_models,
        "platform": platform_info,
    }


# ---------------------------------------------------------------------------
# Registration — call this once at startup from main.py
# ---------------------------------------------------------------------------

def register_system_tools() -> None:
    """Register get_system_info into the global tool registry."""

    register_tool(
        name="get_system_info",
        description=(
            "Return current system resource usage: CPU percent and core count, "
            "RAM total/used/free, disk usage per drive, available Ollama models "
            "(name and size), and OS/Python version. "
            "Useful for diagnosing performance or checking which models are loaded."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=get_system_info,
        is_destructive=False,
    )
