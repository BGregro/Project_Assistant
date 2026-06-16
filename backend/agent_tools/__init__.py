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
    description: str = "",
    input_schema: dict | None = None,
    handler: Callable | None = None,
    is_destructive: bool = False,
    # Accept alternate keyword names used by some agent-written and Phase 9 tools:
    parameters: dict | None = None,
    destructive: bool | None = None,
) -> None:
    """
    Register a callable tool with the agent.

    Accepts both naming conventions:
      input_schema / parameters    — both refer to the JSON schema
      is_destructive / destructive — both control the permission layer

    Safety guards:
      - If description is callable, the caller swapped description and handler.
        Log an error and skip registration — never store a function in the definition.
      - If input_schema is a list instead of a dict, convert to an empty object schema.

    Args:
        name:           Tool identifier (must match what Claude calls it).
        description:    Human-readable description sent to Claude in the API request.
        input_schema:   JSON Schema dict describing the tool's parameters.
        handler:        Async callable that executes the tool. Receives **input_schema props.
        is_destructive: Marks the tool for permission-layer interception before execution.
        parameters:     Alias for input_schema (accepted for compatibility).
        destructive:    Alias for is_destructive (accepted for compatibility).
    """
    # Guard: detect swapped positional args (agent wrote handler as 2nd arg)
    if callable(description):
        logger.error(
            f"[registry] register_tool('{name}') received a callable as 'description'. "
            "The handler was passed in the wrong position. "
            "Skipping registration to prevent JSON serialization crash. "
            "Correct call: register_tool(name, description_str, input_schema, handler)"
        )
        return

    # Resolve alternate keyword names
    if input_schema is None and parameters is not None:
        input_schema = parameters
    if destructive is not None:
        is_destructive = destructive

    # Ensure input_schema is a dict (agent sometimes writes a list)
    if not isinstance(input_schema, dict):
        input_schema = {"type": "object", "properties": {}}
    if "type" not in input_schema:
        input_schema["type"] = "object"
    if "properties" not in input_schema:
        input_schema["properties"] = {}

    if name in _registry:
        logger.debug(f"[registry] Tool '{name}' already registered — skipping.")
        return

    _registry[name] = {
        "definition": {
            "name":         name,
            "description":  description,
            "input_schema": input_schema,
        },
        "handler":        handler,
        "is_destructive": is_destructive,
    }
    logger.info(f"[registry] Registered tool: '{name}' (destructive={is_destructive})")


def get_all_definitions() -> list[dict]:
    """
    Return all tool definitions in the format Anthropic's `tools=` parameter expects.
    Called once per Claude API request to tell Claude what it can use.
    """
    return [entry["definition"] for entry in _registry.values()]


def get_all_definitions_for_prefilter() -> list[dict]:
    """
    Return tool definitions in a compact format for the local LLM pre-filter.

    Unlike get_all_definitions() which returns Anthropic-formatted dicts with
    'input_schema', this returns dicts with a 'parameters' key so that
    local_llm.select_relevant_tools() can describe each tool without sending
    the full Anthropic schema to a local model.

    Each entry has: name, description, parameters (the raw input_schema dict).
    Called by local_llm.select_relevant_tools() and capabilities.py.
    """
    return [
        {
            "name":        entry["definition"]["name"],
            "description": entry["definition"]["description"],
            "parameters":  entry["definition"]["input_schema"],
        }
        for entry in _registry.values()
    ]


def get_handler(name: str) -> Callable | None:
    """Return the async handler for a named tool, or None if unrecognised."""
    entry = _registry.get(name)
    return entry["handler"] if entry else None


def is_destructive(name: str) -> bool:
    """Return True if the tool requires user confirmation before running."""
    entry = _registry.get(name)
    return entry.get("is_destructive", False) if entry else False


def list_tools() -> list[str]:
    """Return names of all registered tools (useful for logging/debug and pre-filter)."""
    return list(_registry.keys())
