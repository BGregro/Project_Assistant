"""
memory/embeddings.py  —  Semantic Vector Memory

Provides persistent vector storage for conversation turns so the agent can
retrieve semantically relevant past context even when it's outside the
recent-turns window.

Stack:
  - Ollama (nomic-embed-text) for local embeddings — no external API or cost
  - ChromaDB as the local vector store — file-backed, zero infrastructure

Storage layout:
  project_root/memory/vectors/   ← ChromaDB persists its files here

Typical flow (called from main.py):
  1. After every turn  → store_turn()    embeds and upserts the turn
  2. Before every turn → search_similar() finds the top-K relevant past turns
  3. On history clear  → clear_all()     wipes the vector store

Failure policy: every public function logs and returns a safe default (None /
empty list / False) — a broken vector store must NEVER crash the agent.

Ollama embed endpoint: POST /api/embed  (requires Ollama ≥ 0.1.26)
Required model: ollama pull nomic-embed-text
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

import httpx
import chromadb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

# project_root/memory/vectors/
_PROJECT_ROOT = Path(__file__).parent.parent.parent
VECTOR_DIR    = _PROJECT_ROOT / "memory" / "vectors"

COLLECTION_NAME = "conversation_history"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
EMBED_TIMEOUT = 30.0   # local inference can be slow on CPU

# ---------------------------------------------------------------------------
# ChromaDB — lazy singleton so startup is instant
# ---------------------------------------------------------------------------

_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def _get_collection() -> chromadb.Collection:
    """
    Return the ChromaDB collection, creating client + collection on first call.
    Uses cosine distance (standard for text embeddings — value range 0-2,
    lower = more similar).
    """
    global _chroma_client, _collection
    if _collection is None:
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(VECTOR_DIR))
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"[embeddings] ChromaDB collection '{COLLECTION_NAME}' ready. "
            f"Stored turns: {_collection.count()}"
        )
    return _collection


# ---------------------------------------------------------------------------
# Ollama embedding
# ---------------------------------------------------------------------------

async def _embed(
    text: str,
    model: str,
    base_url: str,
) -> Optional[list[float]]:
    """
    Call Ollama /api/embed and return the embedding vector.
    Returns None on any failure so callers can skip storage gracefully.

    Ollama endpoint spec (v0.1.26+):
      POST /api/embed
      { "model": "nomic-embed-text", "input": "<text>" }
      → { "embeddings": [[...float list...]] }
    """
    try:
        async with httpx.AsyncClient(timeout=EMBED_TIMEOUT) as client:
            resp = await client.post(
                f"{base_url}/api/embed",
                json={"model": model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            if embeddings and isinstance(embeddings[0], list):
                return embeddings[0]
            logger.warning("[embeddings] Ollama returned unexpected embed shape.")
            return None

    except httpx.ConnectError:
        logger.warning("[embeddings] Ollama offline — skipping embed.")
        return None
    except httpx.TimeoutException:
        logger.warning(f"[embeddings] Embed request timed out after {EMBED_TIMEOUT}s.")
        return None
    except Exception as e:
        logger.warning(f"[embeddings] Embed failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Turn ID
# ---------------------------------------------------------------------------

def _turn_id(timestamp: str, user_content: str) -> str:
    """
    Deterministic, collision-resistant ID for a conversation turn.
    Uses a short SHA-1 prefix — enough for millions of turns with negligible
    collision probability.
    """
    raw = f"{timestamp}:{user_content[:60]}"
    return "turn_" + hashlib.sha1(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def store_turn(
    timestamp: str,
    user_content: str,
    assistant_content: str,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = "http://localhost:11434",
) -> bool:
    """
    Embed a completed conversation turn and upsert it into ChromaDB.

    The document text is the full turn concatenated — this lets semantic
    search find turns where either the question or the answer is relevant.

    Stored metadata keeps truncated copies of each side (ChromaDB metadata
    values are limited; full content is in the document field).

    Returns True on success, False on any failure.
    """
    # Concatenate turn as a single document for embedding
    document = f"User: {user_content}\nAssistant: {assistant_content}"
    embedding = await _embed(document, model=model, base_url=base_url)

    if embedding is None:
        # Embed failed (Ollama offline, model not pulled, etc.) — not fatal
        return False

    turn_id = _turn_id(timestamp, user_content)

    try:
        col = _get_collection()
        col.upsert(
            ids=[turn_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                # Store truncated copies for quick access in search results
                # (ChromaDB metadata is not for full text storage)
                "user_content":      user_content[:500],
                "assistant_content": assistant_content[:500],
                "timestamp":         timestamp,
            }],
        )
        logger.debug(f"[embeddings] Stored turn {turn_id[:20]}…")
        return True

    except Exception as e:
        logger.error(f"[embeddings] ChromaDB upsert failed: {e}")
        return False


async def search_similar(
    query: str,
    n_results: int = 3,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = "http://localhost:11434",
) -> list[dict]:
    """
    Find the n_results most semantically similar past turns to `query`.

    Returns a list of dicts:
      {
        "user_content":      str,   # truncated to 500 chars
        "assistant_content": str,   # truncated to 500 chars
        "timestamp":         str,   # ISO timestamp
        "distance":          float, # cosine distance — 0 = identical, 2 = opposite
      }

    Returns [] on any failure — callers must handle empty results gracefully.
    """
    embedding = await _embed(query, model=model, base_url=base_url)
    if embedding is None:
        return []

    try:
        col = _get_collection()
        count = col.count()

        if count == 0:
            return []  # Empty store — nothing to retrieve

        # Cap n_results to how many turns are actually stored
        actual_n = min(n_results, count)

        results = col.query(
            query_embeddings=[embedding],
            n_results=actual_n,
            include=["metadatas", "distances"],
        )

        output = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            output.append({
                "user_content":      meta.get("user_content", ""),
                "assistant_content": meta.get("assistant_content", ""),
                "timestamp":         meta.get("timestamp", ""),
                "distance":          dist,
            })

        return output

    except Exception as e:
        logger.error(f"[embeddings] ChromaDB query failed: {e}")
        return []


def clear_all() -> None:
    """
    Delete all stored embeddings (called when the user clears conversation history).
    Recreates the empty collection so the store is still usable after clearing.
    """
    global _collection
    try:
        client = _chroma_client or chromadb.PersistentClient(path=str(VECTOR_DIR))
        client.delete_collection(COLLECTION_NAME)
        _collection = None  # Force re-creation on next _get_collection() call
        logger.info("[embeddings] Vector store cleared.")
    except Exception as e:
        logger.error(f"[embeddings] Failed to clear vector store: {e}")


def count() -> int:
    """Return how many turns are stored. Returns 0 on failure."""
    try:
        return _get_collection().count()
    except Exception:
        return 0
