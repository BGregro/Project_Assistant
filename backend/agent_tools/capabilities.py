"""
agent_tools/capabilities.py  —  Self-Aware Capabilities Tool

Provides the `list_capabilities` tool so the agent can answer questions like
"what can you do?" or "what tools do you have?" with an accurate, always
up-to-date answer derived directly from the live tool registry.

Unlike hard-coded descriptions, this tool introspects the registry at call time,
so it automatically reflects any new tools added in Phase 2 or 3 without any
manual updates.

Phase 3c addition: also lists files in agent_tools/generated/ so the agent
always knows what tool code it has already written (even across restarts).

Registration: call register_capabilities_tools() once at startup from main.py.
"""

import logging
from typing import Any

from . import register_tool, get_all_definitions, list_tools, is_destructive
from .hot_reload import list_generated_tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

async def list_capabilities() -> dict[str, Any]:
    """
    Introspect the live tool registry and return a structured description of
    every registered tool, plus top-level agent architecture information,
    plus a listing of agent-written tools in agent_tools/generated/.

    Returns a dict with:
      - "agent_info":       High-level description of the agent's architecture.
      - "tool_count":       Total number of registered tools.
      - "tools":            List of tool descriptors (one per registered tool).
      - "generated_tools":  List of .py filenames in agent_tools/generated/
                            (tools the agent has written itself).

    Each tool descriptor contains:
      - name:           The tool's identifier (what Claude calls it).
      - description:    Human-readable description passed to the LLM.
      - parameters:     The JSON Schema input_schema for the tool's parameters.
      - is_destructive: Whether the tool requires user confirmation before running.
    """
    logger.info("[capabilities] list_capabilities called — introspecting registry.")

    definitions = get_all_definitions()
    registered  = list_tools()

    # Build the per-tool descriptor list
    tools_info = []
    for defn in definitions:
        name = defn.get("name", "?")
        tools_info.append({
            "name":           name,
            "description":    defn.get("description", ""),
            "parameters":     defn.get("input_schema", {}),
            "is_destructive": is_destructive(name),
        })

    # List agent-generated tool files (Phase 3c)
    generated = list_generated_tools()

    return {
        "success": True,
        "agent_info": (
            "This is a two-tier personal AI agent. "
            "The primary LLM is Claude (Anthropic API), used for all tool use and complex reasoning. "
            "A secondary local LLM (Ollama / qwen2.5:7b) handles prompt optimisation, "
            "context compression, and acts as an offline fallback when the API is unreachable. "
            "Conversation history is persisted to disk (history.json) and embedded into a local "
            "ChromaDB vector store (nomic-embed-text) for semantic retrieval of relevant past context. "
            "Tools are registered in a central registry; new tools can be added without modifying "
            "the agent core. The agent can write new tools to agent_tools/generated/ using "
            "write_tool and activate them with reload_tool."
        ),
        "tool_count": len(tools_info),
        "tools": tools_info,
        # Phase 3c: agent-written tool files that persist across sessions
        "generated_tools": {
            "description": (
                "Python files in agent_tools/generated/ — tools written by the agent itself. "
                "These are auto-loaded on every server restart."
            ),
            "files": generated,
            "count": len(generated),
        },
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_capabilities_tools() -> None:
    """Register the list_capabilities tool. Call once at startup from main.py."""

    register_tool(
        name="list_capabilities",
        description=(
            "List all tools and capabilities available to this agent. "
            "Returns a structured description of every registered tool (name, description, "
            "parameters, whether it requires user confirmation) plus a summary of the agent's "
            "architecture and a list of agent-written tool files in agent_tools/generated/. "
            "Use this when the user asks what you can do, what tools you have, "
            "or how you work."
        ),
        input_schema={
            "type":       "object",
            "properties": {},
            "required":   [],
        },
        handler=list_capabilities,
        is_destructive=False,   # Read-only introspection — no side effects
    )
