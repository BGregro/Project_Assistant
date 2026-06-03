"""
agent_tools/__init__.py  —  Tool Registry

Decouples tool *definitions* (what Claude sees) from tool *handlers* (what actually runs).
agent_core.py depends only on this registry interface, never on individual tool files.
Adding a new tool means: create agent_tools/mytool.py, call register_tool(), done.
"""

import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

# Internal store: tool_name -> {"definition": {...}, "handler": callable, "is_destructive": bool}
_registry: dict[str, dict] = {}


def register_tool(
    name: str,
    description: str,
    input_schema: dict,
    handler: Callable,
    is_destructive: bool = False,
) -> None:
    """
    Register a callable tool with the agent.

    Args:
        name:           Tool identifier (must match what Claude calls it).
        description:    Human-readable description sent to Claude in the API request.
        input_schema:   JSON Schema dict describing the tool's parameters.
        handler:        Async callable that executes the tool. Receives **input_schema props.
        is_destructive: Marks the tool for permission-layer interception before execution.
    """
    if name in _registry:
        logger.warning(f"[registry] Tool '{name}' is being re-registered (overwriting).")

    _registry[name] = {
        "definition": {
            "name": name,
            "description": description,
            "input_schema": input_schema,
        },
        "handler": handler,
        "is_destructive": is_destructive,
    }
    logger.info(f"[registry] Registered tool: '{name}' (destructive={is_destructive})")


def get_all_definitions() -> list[dict]:
    """
    Return all tool definitions in the format Anthropic's `tools=` parameter expects.
    Called once per Claude API request to tell Claude what it can use.
    """
    return [entry["definition"] for entry in _registry.values()]


def get_handler(name: str) -> Callable | None:
    """Return the async handler for a named tool, or None if unrecognised."""
    entry = _registry.get(name)
    return entry["handler"] if entry else None


def is_destructive(name: str) -> bool:
    """Return True if the tool requires user confirmation before running."""
    entry = _registry.get(name)
    return entry.get("is_destructive", False) if entry else False


def list_tools() -> list[str]:
    """Return names of all registered tools (useful for logging/debug)."""
    return list(_registry.keys())
