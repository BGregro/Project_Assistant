"""
knowledge_graph.py  —  Phase 12c: Semantic Knowledge Graph

Tracks relationships between concepts, tools, projects, research topics,
skills, services, and people. Nodes are named entities. Edges are typed,
weighted relationships between two nodes.

Built automatically as the agent works:
  - TaskRunner._update_knowledge_graph() (task_runner.py) adds edges between
    goal concepts and the tools used to accomplish them, and between tools
    that were used together in the same task.
  - Anything else (research, manual observations) can call add_graph_edge
    directly, either from Python or as a tool Claude calls itself.

Also queryable:
  - query_knowledge_graph(concept, depth) — BFS subgraph around a concept.
  - add_graph_edge(from_node, to_node, relationship) — manually note a link.

Data lives in  memory/semantic_graph.json  (relative to the project root).
Atomic writes: every save() writes to a .tmp file first, then replaces, so a
crash mid-write never corrupts the store.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GRAPH_FILE = Path(__file__).resolve().parent.parent.parent / "memory" / "semantic_graph.json"

_VALID_NODE_TYPES = {"concept", "tool", "project", "skill", "service", "person", "topic"}

_MAX_EDGE_STRENGTH = 5.0
_MAX_QUERY_NODES = 50

# Simple stopword list for extract_concepts_from_text — no LLM call, pure Python.
_STOPWORDS = {
    "about", "above", "after", "again", "against", "all", "and", "any",
    "are", "aren't", "because", "been", "before", "being", "below",
    "between", "both", "but", "cannot", "could", "couldn't", "did",
    "didn't", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "here", "hers", "herself", "himself",
    "his", "how", "into", "isn't", "it's", "its", "itself", "let's",
    "more", "most", "mustn't", "myself", "once", "only", "other", "ours",
    "ourselves", "over", "own", "same", "shan't", "she'd", "she'll",
    "she's", "should", "shouldn't", "some", "such", "than", "that",
    "that's", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "under", "until", "very",
    "wasn't", "we'd", "we'll", "we're", "we've", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while",
    "who's", "whom", "with", "won't", "would", "wouldn't", "your",
    "yours", "yourself", "yourselves", "please", "using", "using.",
}

_EMPTY_GRAPH: dict = {"nodes": {}, "edges": [], "last_updated": ""}


# ---------------------------------------------------------------------------
# Core I/O
# ---------------------------------------------------------------------------

def load() -> dict:
    """
    Read semantic_graph.json and return its contents as a dict.
    Returns the canonical empty structure on missing/corrupt file so callers
    never need to handle missing keys.
    """
    if not GRAPH_FILE.exists():
        return {"nodes": {}, "edges": [], "last_updated": ""}
    try:
        with open(GRAPH_FILE, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("nodes", {})
        data.setdefault("edges", [])
        data.setdefault("last_updated", "")
        return data
    except Exception as e:
        logger.warning(f"[knowledge_graph] Could not load {GRAPH_FILE}: {e} — returning empty graph.")
        return {"nodes": {}, "edges": [], "last_updated": ""}


def save(data: dict) -> None:
    """
    Write data to semantic_graph.json using an atomic rename so partial
    writes never corrupt the store.
    """
    GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    tmp = GRAPH_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, GRAPH_FILE)
    except Exception as e:
        logger.error(f"[knowledge_graph] Failed to save graph: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Node / edge mutation
# ---------------------------------------------------------------------------

def add_node(name: str, node_type: str, metadata: dict | None = None) -> None:
    """
    Add a node to the graph, or update it in place if it already exists.

    Args:
        name:      The node's identifier (e.g. "python", "search_web", "todo_cli").
        node_type: One of "concept", "tool", "project", "skill", "service",
                   "person", "topic". Unrecognised types are coerced to "concept".
        metadata:  Optional free-form dict merged into the node's metadata.
                   Existing keys are overwritten by new values; nothing else
                   is removed.

    Never duplicates — nodes are keyed by name in a dict, so re-adding the
    same name just refreshes last_seen and merges metadata.
    """
    if not name or not isinstance(name, str):
        return
    name = name.strip()
    if not name:
        return
    if node_type not in _VALID_NODE_TYPES:
        node_type = "concept"
    metadata = metadata or {}

    data = load()
    now = datetime.now(timezone.utc).isoformat()

    existing = data["nodes"].get(name)
    if existing:
        existing["last_seen"] = now
        existing["type"] = node_type or existing.get("type", "concept")
        existing.setdefault("metadata", {}).update(metadata)
    else:
        data["nodes"][name] = {
            "type": node_type,
            "metadata": metadata,
            "first_seen": now,
            "last_seen": now,
        }
    save(data)


def add_edge(from_node: str, to_node: str, relationship: str, strength: float = 1.0, source: str = "") -> None:
    """
    Add an edge between two nodes, or strengthen an existing identical edge.

    An edge is identified by the triple (from_node, to_node, relationship).
    If it already exists: strength is incremented by 0.1 (capped at 5.0) and
    last_seen is refreshed. Otherwise a new edge is created.

    Args:
        from_node:    Source node name (should already exist via add_node).
        to_node:      Target node name (should already exist via add_node).
        relationship: Short label for the relationship, e.g. "solved_with",
                       "used_together", "related_to".
        strength:     Initial weight for a new edge (default 1.0).
        source:       Optional task_id / research_id that produced this edge —
                       useful for tracing where a connection came from.
    """
    if not from_node or not to_node or not relationship:
        return
    if from_node == to_node:
        return  # no self-edges — not useful for graph traversal

    data = load()
    now = datetime.now(timezone.utc).isoformat()

    for edge in data["edges"]:
        if edge["from"] == from_node and edge["to"] == to_node and edge["relationship"] == relationship:
            edge["strength"] = min(_MAX_EDGE_STRENGTH, round(edge.get("strength", 1.0) + 0.1, 2))
            edge["last_seen"] = now
            if source and source not in edge.get("sources", []):
                edge.setdefault("sources", []).append(source)
            save(data)
            return

    data["edges"].append({
        "from": from_node,
        "to": to_node,
        "relationship": relationship,
        "strength": float(strength),
        "sources": [source] if source else [],
        "first_seen": now,
        "last_seen": now,
    })
    save(data)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def query_graph(concept: str, depth: int = 2) -> dict:
    """
    Breadth-first search from `concept` up to `depth` hops, following edges
    in either direction (the graph is treated as undirected for traversal
    purposes, even though each edge stores a directional relationship).

    Returns:
        {
          "center": concept,
          "nodes": [{"name": ..., "type": ..., ...}, ...],
          "edges": [{"from": ..., "to": ..., "relationship": ..., "strength": ...}, ...],
          "depth": depth,
        }

    Capped at _MAX_QUERY_NODES (50) total nodes to keep results manageable.
    Returns an empty subgraph (nodes=[], edges=[]) if concept is not present.
    """
    depth = max(1, min(5, int(depth)))
    data = load()
    nodes = data["nodes"]
    edges = data["edges"]

    if concept not in nodes:
        return {"center": concept, "nodes": [], "edges": [], "depth": depth}

    # Build adjacency: node -> list of (neighbor, edge)
    adjacency: dict[str, list[tuple[str, dict]]] = {}
    for edge in edges:
        adjacency.setdefault(edge["from"], []).append((edge["to"], edge))
        adjacency.setdefault(edge["to"], []).append((edge["from"], edge))

    visited: set[str] = {concept}
    frontier = [concept]
    found_edges: list[dict] = []
    found_edge_keys: set[tuple] = set()

    for _ in range(depth):
        next_frontier: list[str] = []
        for node_name in frontier:
            for neighbor, edge in adjacency.get(node_name, []):
                edge_key = (edge["from"], edge["to"], edge["relationship"])
                if edge_key not in found_edge_keys:
                    found_edge_keys.add(edge_key)
                    found_edges.append(edge)
                if neighbor not in visited:
                    if len(visited) >= _MAX_QUERY_NODES:
                        continue
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if len(visited) >= _MAX_QUERY_NODES:
            break

    result_nodes = [
        {"name": name, **nodes[name]}
        for name in visited
        if name in nodes
    ][:_MAX_QUERY_NODES]

    return {
        "center": concept,
        "nodes": result_nodes,
        "edges": found_edges,
        "depth": depth,
    }


def get_strongest_connections(concept: str, top_n: int = 10) -> list:
    """
    Return the top_n edges involving `concept` (either as from_node or
    to_node), sorted by strength descending.
    """
    top_n = max(1, min(50, int(top_n)))
    data = load()
    involved = [
        edge for edge in data["edges"]
        if edge["from"] == concept or edge["to"] == concept
    ]
    involved.sort(key=lambda e: e.get("strength", 0.0), reverse=True)
    return involved[:top_n]


def extract_concepts_from_text(text: str) -> list[str]:
    """
    Simple keyword extraction — no LLM call, pure Python.

    Splits text on non-alphanumeric characters, lowercases, filters out
    stopwords and short tokens, and de-duplicates while preserving order.
    Used to auto-build the graph from task goals and research topics.

    Returns a list of meaningful tokens longer than 4 characters.
    """
    if not text:
        return []
    tokens = re.split(r"[^a-zA-Z0-9']+", text.lower())
    seen: set[str] = set()
    concepts: list[str] = []
    for tok in tokens:
        tok = tok.strip("'")
        if len(tok) <= 4:
            continue
        if tok in _STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        concepts.append(tok)
    return concepts


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register_knowledge_graph_tools() -> None:
    """
    Register the knowledge graph tools with the global tool registry.

    Called once at startup from main.py, after the registry is initialised.
    Both tools are non-destructive — querying is read-only, and adding a
    node/edge is purely additive (building knowledge is not destructive).
    """
    from agent_tools import register_tool

    # ── query_knowledge_graph ────────────────────────────────────────────
    async def _query_knowledge_graph(concept: str, depth: int = 2) -> dict[str, Any]:
        try:
            result = query_graph(concept, depth)
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"[knowledge_graph] query_knowledge_graph failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="query_knowledge_graph",
        description=(
            "Query the semantic knowledge graph to find how concepts, tools, and projects "
            "are related. Returns connected nodes and edges up to the specified depth. "
            "Useful for understanding what the agent knows about a topic and how it "
            "connects to other things."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "concept": {
                    "type": "string",
                    "description": "The concept, tool, or project name to center the query on.",
                },
                "depth": {
                    "type": "integer",
                    "description": "How many hops to traverse from the concept (1-5). Default 2.",
                    "default": 2,
                },
            },
            "required": ["concept"],
        },
        handler=_query_knowledge_graph,
        is_destructive=False,
    )

    # ── add_graph_edge ────────────────────────────────────────────────────
    async def _add_graph_edge(
        from_node: str,
        to_node: str,
        relationship: str,
        from_type: str = "concept",
        to_type: str = "concept",
    ) -> dict[str, Any]:
        try:
            add_node(from_node, from_type)
            add_node(to_node, to_type)
            add_edge(from_node, to_node, relationship)
            return {
                "success": True,
                "edge": {"from": from_node, "to": to_node, "relationship": relationship},
            }
        except Exception as e:
            logger.error(f"[knowledge_graph] add_graph_edge failed: {e}")
            return {"success": False, "error": str(e)}

    register_tool(
        name="add_graph_edge",
        description=(
            "Manually add a relationship between two concepts, tools, projects, skills, "
            "services, or people to the semantic knowledge graph. Use this when you notice "
            "a connection worth remembering that the automatic tool-usage tracking wouldn't "
            "capture — e.g. 'ChromaDB' is 'used_by' 'long_term memory'. "
            "This is purely additive and never destructive."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "from_node": {"type": "string", "description": "Source node name."},
                "to_node": {"type": "string", "description": "Target node name."},
                "relationship": {
                    "type": "string",
                    "description": "Short relationship label, e.g. 'depends_on', 'related_to', 'part_of'.",
                },
                "from_type": {
                    "type": "string",
                    "description": "Node type for from_node: concept, tool, project, skill, service, person, topic. Default 'concept'.",
                    "default": "concept",
                },
                "to_type": {
                    "type": "string",
                    "description": "Node type for to_node. Default 'concept'.",
                    "default": "concept",
                },
            },
            "required": ["from_node", "to_node", "relationship"],
        },
        handler=_add_graph_edge,
        is_destructive=False,
    )

    logger.info("[knowledge_graph] Phase 12c: knowledge graph tools registered (query_knowledge_graph, add_graph_edge)")
