"""
memory_tool.py  —  Phase 3f: Long-Term Memory Tools

Registers three tools that let the agent actively interact with the long-term
memory store:

  log_research   — save research findings for future sessions
  recall_memory  — query tasks, facts, and/or research by keyword
  log_fact       — store a key-value fact (e.g. a discovered setting, URL, version)

All three are non-destructive (no approval required).
"""

from agent_tools import register_tool
from memory import long_term


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def _log_research(topic: str, findings: str, sources: str) -> dict:
    """
    Save research findings to long-term memory so they can be recalled in
    future sessions.

    Args:
        topic:    Short label for what was researched.
        findings: Key findings or conclusions (plain text).
        sources:  Comma-separated list of URLs or tool names used as sources.
    """
    source_list = [s.strip() for s in sources.split(",") if s.strip()]
    long_term.log_research(
        topic=topic,
        findings=findings,
        sources=source_list,
    )
    return {"success": True, "message": f"Research on '{topic}' saved."}


async def _recall_memory(query: str, memory_type: str = "all") -> dict:
    """
    Recall stored memory matching a keyword or topic.

    Args:
        query:       Keyword or topic to search for.
        memory_type: One of "tasks", "facts", "research", or "all".
                     Controls which stores are searched.

    Returns a dict with the relevant results under their respective keys.
    """
    result: dict = {}
    mt = memory_type.lower().strip()

    if mt in ("tasks", "all"):
        result["tasks"] = long_term.query_tasks(keyword=query, last_n=10)

    if mt in ("facts", "all"):
        result["facts"] = long_term.query_facts(key=query)

    if mt in ("research", "all"):
        result["research"] = long_term.query_research(topic=query, last_n=10)

    if not result:
        return {
            "success": False,
            "error": (
                f"Unknown memory_type '{memory_type}'. "
                "Use 'tasks', 'facts', 'research', or 'all'."
            ),
        }

    result["success"] = True
    return result


async def _log_fact(key: str, value: str, source: str) -> dict:
    """
    Store a key-value fact in long-term memory.

    If a fact with the same key already exists it is updated rather than
    duplicated, so calling this multiple times with the same key is safe.

    Args:
        key:    Short identifier, e.g. "python_version" or "project_root".
        value:  The fact content.
        source: Where this information came from (URL, tool name, user statement).
    """
    long_term.log_fact(key=key, value=value, source=source)
    return {"success": True}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_memory_tools() -> None:
    """Register all Phase 3f memory tools into the live tool registry."""

    register_tool(
        name="log_research",
        description=(
            "Save research findings, conclusions, or discoveries to long-term memory "
            "so they can be recalled in future sessions. Use this whenever you complete "
            "a research sub-task and want the findings to persist beyond this conversation."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "topic": {
                    "type":        "string",
                    "description": "Short label for what was researched (e.g. 'Python packaging tools 2025').",
                },
                "findings": {
                    "type":        "string",
                    "description": "Key findings, conclusions, or a summary of what was discovered.",
                },
                "sources": {
                    "type":        "string",
                    "description": (
                        "Comma-separated list of sources used: URLs, tool names, "
                        "or descriptions of where the information came from."
                    ),
                },
            },
            "required": ["topic", "findings", "sources"],
        },
        handler=_log_research,
        is_destructive=False,
    )

    register_tool(
        name="recall_memory",
        description=(
            "Query long-term memory for past task outcomes, stored facts, or research "
            "findings that match a keyword or topic. Use this when you want to know "
            "what has been tried or discovered in previous sessions before starting a task."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": "Keyword or topic to search for in stored memory.",
                },
                "memory_type": {
                    "type":        "string",
                    "enum":        ["tasks", "facts", "research", "all"],
                    "description": (
                        "Which memory store to search: "
                        "'tasks' for past task outcomes, "
                        "'facts' for stored key-value facts, "
                        "'research' for saved research findings, "
                        "'all' to search everything (default)."
                    ),
                },
            },
            "required": ["query"],
        },
        handler=_recall_memory,
        is_destructive=False,
    )

    register_tool(
        name="log_fact",
        description=(
            "Store a key-value fact in long-term memory (e.g. a URL, version number, "
            "config value, or any piece of information worth remembering across sessions). "
            "If a fact with the same key already exists it will be updated, not duplicated."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "key": {
                    "type":        "string",
                    "description": "Short identifier for this fact (e.g. 'project_root', 'api_base_url').",
                },
                "value": {
                    "type":        "string",
                    "description": "The fact content.",
                },
                "source": {
                    "type":        "string",
                    "description": "Where this fact came from (URL, tool name, or user statement).",
                },
            },
            "required": ["key", "value", "source"],
        },
        handler=_log_fact,
        is_destructive=False,
    )
