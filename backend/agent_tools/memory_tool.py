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
from agent_tools.local_llm import local_llm_call, strip_think_tags
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


async def _recall_projects(query: str = "") -> dict:
    """
    Search long-term memory for past software projects that match a keyword.

    Call this at the start of any software development task to check whether
    a similar project has been built before — its structure, dependencies, and
    lessons learned can inform the new scaffold.

    Args:
        query: Keyword or project name fragment to search for.
                Pass an empty string to return the 5 most recent projects.
    """
    projects = long_term.query_projects(keyword=query, last_n=5)
    return {"success": True, "projects": projects}


async def correlate_memories(concept1: str, concept2: str) -> dict:
    """
    Phase 12d: find connections between two concepts across all memory layers.

    Searches tasks, research, facts, and the knowledge graph for overlaps
    between the two concepts, then asks the local model to describe the
    relationship in plain language.

    Args:
        concept1: First concept/topic to correlate.
        concept2: Second concept/topic to correlate.
    """
    from memory.long_term import load as _lt, _age_label
    data = _lt()
    c1, c2 = concept1.lower(), concept2.lower()

    matching_tasks = [
        {"goal": t["goal"][:80], "outcome": t["outcome"], "age": _age_label(t.get("timestamp", ""))}
        for t in data.get("tasks", [])
        if c1 in t.get("goal", "").lower() and c2 in t.get("goal", "").lower()
    ][-10:]

    matching_research = [
        {"topic": r["topic"], "snippet": r["findings"][:150], "age": _age_label(r.get("timestamp", ""))}
        for r in data.get("research", [])
        if c1 in (r.get("topic", "") + r.get("findings", "")).lower()
        and c2 in (r.get("topic", "") + r.get("findings", "")).lower()
    ][-5:]

    matching_facts = [
        {"key": f["key"], "value": f["value"][:100]}
        for f in data.get("facts", [])
        if c1 in (f.get("key", "") + f.get("value", "")).lower()
        and c2 in (f.get("key", "") + f.get("value", "")).lower()
    ]

    graph_connection = None
    try:
        from agent_tools.knowledge_graph import get_strongest_connections
        for edge in get_strongest_connections(concept1, top_n=20):
            if c2 in edge.get("to", "").lower() or c2 in edge.get("from", "").lower():
                graph_connection = edge
                break
    except Exception:
        pass

    total = len(matching_tasks) + len(matching_research) + len(matching_facts)
    if total == 0 and not graph_connection:
        return {"success": True, "concept1": concept1, "concept2": concept2,
                "connection_found": False,
                "summary": f"No direct connection found between '{concept1}' and '{concept2}' in memory."}

    evidence = (f"Tasks: {matching_tasks}\nResearch: {matching_research}\n"
                f"Facts: {matching_facts}\nGraph edge: {graph_connection}\n")
    prompt = (
        f"Describe how '{concept1}' and '{concept2}' are connected based on this evidence:\n"
        f"{evidence[:800]}\n\n"
        "Write 2-3 sentences describing the relationship. Be specific and factual."
    )
    raw = await local_llm_call(prompt, "qwen3:14b", "http://localhost:11434")
    summary = strip_think_tags(raw).strip()

    return {"success": True, "concept1": concept1, "concept2": concept2,
            "connection_found": True, "matching_tasks": matching_tasks,
            "matching_research": matching_research, "matching_facts": matching_facts,
            "graph_connection": graph_connection, "summary": summary}


async def timeline_memory(start_date: str, end_date: str = "") -> dict:
    """
    Phase 12d: return a chronological timeline of all agent activity
    (tasks, research, facts) within a date range.

    Args:
        start_date: Start of the range, YYYY-MM-DD or ISO datetime.
        end_date:   End of the range, YYYY-MM-DD or ISO datetime.
                    Defaults to now if omitted.
    """
    from memory.long_term import load as _lt, _parse_ts, _age_label
    from datetime import datetime, timezone

    try:
        s = start_date if "T" in start_date else start_date + "T00:00:00"
        start_dt = _parse_ts(s)
        end_dt = (_parse_ts(end_date + "T23:59:59" if end_date and "T" not in end_date else end_date)
                  if end_date else datetime.now(timezone.utc))
    except Exception:
        return {"success": False, "error": "Invalid date format. Use YYYY-MM-DD or ISO datetime."}

    data = _lt()
    events = []

    for t in data.get("tasks", []):
        ts = _parse_ts(t.get("timestamp", ""))
        if ts and start_dt <= ts <= end_dt:
            events.append({"type": "task", "timestamp": t["timestamp"],
                "age": _age_label(t["timestamp"]),
                "summary": f"[{t['outcome']}] {t['goal'][:80]}",
                "tools": t.get("tools_used", []),
                "reflection": t.get("reflection", "")[:100]})

    for r in data.get("research", []):
        ts = _parse_ts(r.get("timestamp", ""))
        if ts and start_dt <= ts <= end_dt:
            events.append({"type": "research", "timestamp": r["timestamp"],
                "age": _age_label(r["timestamp"]),
                "summary": f"Research: {r['topic'][:80]}"})

    for f in data.get("facts", []):
        ts = _parse_ts(f.get("timestamp", ""))
        if ts and start_dt <= ts <= end_dt:
            events.append({"type": "fact", "timestamp": f["timestamp"],
                "age": _age_label(f["timestamp"]),
                "summary": f"Fact: {f['key']} = {f['value'][:60]}"})

    events.sort(key=lambda e: e["timestamp"])
    return {"success": True, "start_date": start_date, "end_date": end_date or "now",
            "total_events": len(events),
            "task_count": sum(1 for e in events if e["type"] == "task"),
            "research_count": sum(1 for e in events if e["type"] == "research"),
            "fact_count": sum(1 for e in events if e["type"] == "fact"),
            "events": events}


async def query_memory(query: str, memory_types: str = "all", max_results: int = 10) -> dict:
    """
    Phase 12d: unified search across all memory layers — tasks, facts,
    research, projects, and knowledge graph.

    Args:
        query:        Keyword or topic to search for.
        memory_types: "all" or comma-separated list of: tasks, facts, research,
                      projects, graph.
        max_results:  Maximum number of results to return.
    """
    from memory.long_term import load as _lt, _age_label, semantic_query_research
    types = [t.strip() for t in memory_types.split(",")] if memory_types != "all" \
            else ["tasks", "facts", "research", "projects", "graph"]
    q = query.lower()
    data = _lt()
    results = []

    if "tasks" in types:
        for t in reversed(data.get("tasks", [])):
            if q in t.get("goal", "").lower() or q in t.get("reflection", "").lower():
                results.append({"type": "task",
                    "relevance": 2 if q in t.get("goal", "").lower() else 1,
                    "timestamp": t.get("timestamp", ""),
                    "age": _age_label(t.get("timestamp", "")),
                    "content": f"[{t['outcome']}] {t['goal'][:80]}",
                    "detail": t.get("reflection", "")[:100]})

    if "research" in types:
        try:
            for r in semantic_query_research(query, n_results=5):
                results.append({"type": "research", "relevance": 3,
                    "timestamp": r.get("timestamp", ""),
                    "age": _age_label(r.get("timestamp", "")),
                    "content": f"Research: {r['topic']}",
                    "detail": r.get("findings", "")[:150]})
        except Exception:
            for r in data.get("research", []):
                if q in (r.get("topic", "") + r.get("findings", "")).lower():
                    results.append({"type": "research", "relevance": 2,
                        "timestamp": r.get("timestamp", ""),
                        "age": _age_label(r.get("timestamp", "")),
                        "content": f"Research: {r['topic'][:80]}",
                        "detail": r.get("findings", "")[:150]})

    if "facts" in types:
        for f in data.get("facts", []):
            if q in (f.get("key", "") + f.get("value", "")).lower():
                results.append({"type": "fact",
                    "relevance": 1 if f.get("may_be_stale") else 2,
                    "timestamp": f.get("timestamp", ""),
                    "age": f.get("age", _age_label(f.get("timestamp", ""))),
                    "content": f"Fact: {f['key']} = {f['value'][:80]}",
                    "detail": "⚠ May be stale" if f.get("may_be_stale") else ""})

    if "projects" in types:
        for p in data.get("projects", []):
            if q in (p.get("name", "") + p.get("description", "")).lower():
                results.append({"type": "project", "relevance": 2,
                    "timestamp": p.get("timestamp", ""),
                    "age": _age_label(p.get("timestamp", "")),
                    "content": f"Project: {p['name']} ({p.get('outcome', '?')})",
                    "detail": p.get("description", "")[:100]})

    if "graph" in types:
        try:
            from agent_tools.knowledge_graph import get_strongest_connections
            for e in get_strongest_connections(query, top_n=5):
                results.append({"type": "graph_edge", "relevance": 2,
                    "timestamp": e.get("last_seen", ""),
                    "age": _age_label(e.get("last_seen", "")),
                    "content": f"Graph: {e['from']} —[{e['relationship']}]→ {e['to']}",
                    "detail": f"Strength: {e.get('strength', 1.0):.1f}"})
        except Exception:
            pass

    results.sort(key=lambda r: r["relevance"], reverse=True)
    return {"success": True, "query": query, "memory_types": memory_types,
            "total_found": len(results), "results": results[:max_results]}


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
        name="recall_projects",
        description=(
            "Search long-term memory for past software projects that match a keyword or name. "
            "Call this at the start of any development task to check whether something similar "
            "has been built before — reuse structure, dependencies, and lessons learned. "
            "Pass an empty query to see the 5 most recently built projects."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": (
                        "Keyword or project name to search for "
                        "(e.g. 'cli', 'flask', 'todo'). "
                        "Empty string returns the 5 most recent projects."
                    ),
                },
            },
            "required": [],
        },
        handler=_recall_projects,
        is_destructive=False,
    )

    register_tool(
        name="log_fact",
        description=(
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

    register_tool(
        name="correlate_memories",
        description=(
            "Find connections between two concepts across all memory layers — tasks, "
            "research, facts, and the knowledge graph. Uses local model to describe what "
            "the connection means."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "concept1": {
                    "type":        "string",
                    "description": "First concept or topic to correlate.",
                },
                "concept2": {
                    "type":        "string",
                    "description": "Second concept or topic to correlate.",
                },
            },
            "required": ["concept1", "concept2"],
        },
        handler=correlate_memories,
        is_destructive=False,
    )

    register_tool(
        name="timeline_memory",
        description=(
            "Return a chronological timeline of all agent activity (tasks, research, facts) "
            "within a date range. Use YYYY-MM-DD format for dates."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "start_date": {
                    "type":        "string",
                    "description": "Start of the range, YYYY-MM-DD or ISO datetime.",
                },
                "end_date": {
                    "type":        "string",
                    "description": "End of the range, YYYY-MM-DD or ISO datetime. Defaults to now if omitted.",
                },
            },
            "required": ["start_date"],
        },
        handler=timeline_memory,
        is_destructive=False,
    )

    register_tool(
        name="query_memory",
        description=(
            "Unified search across all memory layers — tasks, facts, research, projects, "
            "and knowledge graph. memory_types can be 'all' or comma-separated list of: "
            "tasks, facts, research, projects, graph."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": "Keyword or topic to search for across memory.",
                },
                "memory_types": {
                    "type":        "string",
                    "description": (
                        "'all' or comma-separated list of: tasks, facts, research, "
                        "projects, graph."
                    ),
                },
                "max_results": {
                    "type":        "integer",
                    "description": "Maximum number of results to return (default 10).",
                },
            },
            "required": ["query"],
        },
        handler=query_memory,
        is_destructive=False,
    )
