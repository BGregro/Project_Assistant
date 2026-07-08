"""
agent_tools/__init__.py  —  Tool Registry

Decouples tool *definitions* (what Claude sees) from tool *handlers* (what actually runs).
agent_core.py depends only on this registry interface, never on individual tool files.
Adding a new tool means: create agent_tools/mytool.py, call register_tool(), done.

Phase 15c adds per-tool performance metadata (call_count, success_count, last_used)
that accumulates in-memory over the session, plus get_tool_metadata() and
prune_unused_tools() tools for introspecting and cleaning up generated tools.
"""

import logging
from datetime import datetime, timezone
from typing import Callable, Any

logger = logging.getLogger(__name__)

# Internal store: tool_name -> {"definition": {...}, "handler": callable,
#                                "is_destructive": bool, "metadata": {...}}
_registry: dict[str, dict] = {}


def _now_iso() -> str:
    """Current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def register_tool(
    name: str,
    description: str = "",
    input_schema: dict | None = None,
    handler: Callable | None = None,
    is_destructive: bool = False,
    # Accept alternate keyword names used by some agent-written and Phase 9 tools:
    parameters: dict | None = None,
    destructive: bool | None = None,
    # Phase 15c: performance-tracking metadata. "system" for built-in tools
    # (the default); hot_reload.py overrides this to "agent" after loading
    # a file from agent_tools/generated/.
    created_by: str = "system",
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
        created_by:     "system" (built-in) or "agent" (self-written). Used by
                        get_tool_metadata()/prune_unused_tools() (Phase 15c).
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
        "metadata": {
            "created_date": _now_iso(),
            "created_by":   created_by,
            "call_count":   0,
            "success_count": 0,
            "last_used":    None,
        },
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


# ---------------------------------------------------------------------------
# Phase 15c — Tool performance tracking
# ---------------------------------------------------------------------------

def update_tool_stats(tool_name: str, success: bool) -> None:
    """
    Record the outcome of a tool invocation.

    Called from task_runner.py's dispatch loop after each tool call completes.
    Silently ignores unknown tool names — stats are a nice-to-have, never a
    reason to interrupt the agent loop.
    """
    entry = _registry.get(tool_name)
    if not entry:
        return
    meta = entry.setdefault("metadata", {
        "created_date": _now_iso(), "created_by": "system",
        "call_count": 0, "success_count": 0, "last_used": None,
    })
    meta["call_count"] = meta.get("call_count", 0) + 1
    if success:
        meta["success_count"] = meta.get("success_count", 0) + 1
    meta["last_used"] = _now_iso()


def mark_tool_created_by(name: str, created_by: str, source_file: str | None = None) -> None:
    """
    Update a tool's created_by (and optionally source_file) metadata after
    registration. Used by hot_reload.py to tag tools loaded from
    agent_tools/generated/ as created_by="agent" instead of the "system"
    default that register_tool() applies when no keyword is given.
    """
    entry = _registry.get(name)
    if not entry:
        return
    meta = entry.setdefault("metadata", {
        "created_date": _now_iso(), "created_by": "system",
        "call_count": 0, "success_count": 0, "last_used": None,
    })
    meta["created_by"] = created_by
    if source_file:
        meta["source_file"] = source_file


async def get_tool_metadata(tool_name: str = "") -> dict[str, Any]:
    """
    Return performance metadata for one tool, or all tools sorted by call_count
    descending when tool_name is omitted.

    Each entry includes success_rate_pct = success_count / call_count * 100
    when call_count > 0, else None.
    """
    def _with_rate(meta: dict) -> dict:
        out = dict(meta)
        calls = out.get("call_count", 0)
        out["success_rate_pct"] = (
            round(out.get("success_count", 0) / calls * 100, 1) if calls else None
        )
        return out

    if tool_name:
        entry = _registry.get(tool_name)
        if not entry:
            return {"success": False, "error": f"Unknown tool: '{tool_name}'"}
        return {
            "success": True,
            "tool_name": tool_name,
            "description": entry["definition"].get("description", ""),
            "is_destructive": entry.get("is_destructive", False),
            "metadata": _with_rate(entry.get("metadata", {})),
        }

    all_tools = [
        {"name": name, "metadata": _with_rate(entry.get("metadata", {}))}
        for name, entry in _registry.items()
    ]
    all_tools.sort(key=lambda t: t["metadata"].get("call_count", 0), reverse=True)
    return {"success": True, "tool_count": len(all_tools), "tools": all_tools}


async def prune_unused_tools() -> dict[str, Any]:
    """
    Find agent-generated tools (created_by=="agent") that look unused: never
    called and created more than 30 days ago, or not called in the last 30
    days. Returns candidates for removal — never deletes anything itself.
    The user or agent decides whether to actually delete the file.
    """
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    candidates: list[dict] = []

    for name, entry in _registry.items():
        meta = entry.get("metadata", {})
        if meta.get("created_by") != "agent":
            continue

        reason = ""
        created_date_str = meta.get("created_date")
        last_used_str = meta.get("last_used")

        if meta.get("call_count", 0) == 0:
            try:
                created_dt = datetime.fromisoformat(created_date_str) if created_date_str else None
            except Exception:
                created_dt = None
            if created_dt and created_dt < cutoff:
                reason = "never used, created over 30 days ago"
        elif last_used_str:
            try:
                last_used_dt = datetime.fromisoformat(last_used_str)
                if last_used_dt < cutoff:
                    reason = "not used in over 30 days"
            except Exception:
                pass

        if reason:
            candidates.append({
                "name": name,
                "filename": meta.get("source_file", ""),
                "created_date": created_date_str or "",
                "reason": reason,
            })

    logger.info(f"[registry] prune_unused_tools: {len(candidates)} candidate(s) found.")
    return {"success": True, "candidates": candidates, "count": len(candidates)}


def register_tool_management_tools() -> None:
    """Register get_tool_metadata and prune_unused_tools. Call once at startup from main.py."""

    register_tool(
        name="get_tool_metadata",
        description=(
            "Get performance metadata for a tool: created_date, created_by (system/agent), "
            "call_count, success_count, success_rate_pct, last_used. "
            "Pass tool_name to look up one tool, or omit it to get all tools sorted by "
            "call_count descending. Non-destructive."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to look up. Omit or leave empty for all tools.",
                },
            },
            "required": [],
        },
        handler=get_tool_metadata,
        is_destructive=False,
    )

    register_tool(
        name="prune_unused_tools",
        description=(
            "Find agent-generated tools that appear unused (never called and over 30 days old, "
            "or not called in the last 30 days). Returns a list of removal candidates — "
            "does NOT delete anything automatically. Non-destructive."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=prune_unused_tools,
        is_destructive=False,
    )
