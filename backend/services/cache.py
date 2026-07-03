"""
Semantic response cache.

Before running the full RAG pipeline, check if a semantically similar question
was already answered recently and the source documents have not changed.

Cache lifecycle:
  1. Embed incoming question.
  2. Search {kb_namespace}_cache Qdrant collection at a high similarity threshold (0.92+).
  3. If hit: verify source documents have not been re-ingested since the cache entry was created.
  4. If valid: return cached answer immediately.
  5. If miss or stale: run full RAG, then store result in cache.
  6. Entries expire after cache_ttl_days (default 7).
  7. When a document is deleted or re-ingested, invalidate all cache entries
     that reference that document_id.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from models.tenant import Tenant
from models.document import Document
from services.qdrant import get_qdrant_client, ensure_cache_collection

logger = logging.getLogger(__name__)


def _cache_collection(kb_namespace: str) -> str:
    return f"{kb_namespace}_cache"


def _is_cache_active() -> bool:
    return settings.cache_enabled


async def get_cached_response(
    question_vector: List[float],
    tenant: Tenant,
    db: AsyncSession,
) -> Optional[Dict[str, Any]]:
    """
    Search for a cached response to a semantically similar question.

    Returns the cached answer dict (with cache_hit=True) if a valid, non-stale
    entry is found. Returns None otherwise.
    """
    if not _is_cache_active():
        return None

    # Also check per-tenant module flag
    enabled_modules = tenant.enabled_modules or []
    if "cache" not in enabled_modules or not settings.cache_enabled:
        return None

    cache_col = _cache_collection(tenant.kb_namespace)

    try:
        client = get_qdrant_client()
        results = await client.search(
            collection_name=cache_col,
            query_vector=question_vector,
            limit=1,
            score_threshold=settings.cache_similarity_threshold,
            with_payload=True,
        )
    except Exception as e:
        logger.warning("Cache search failed (will proceed with full RAG): %s", e)
        return None

    if not results:
        return None

    hit = results[0]
    payload = hit.payload or {}

    created_at_str = payload.get("created_at")
    if not created_at_str:
        return None

    # Check TTL
    try:
        created_at = datetime.fromisoformat(created_at_str)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        ttl_days = getattr(tenant, "cache_ttl_days", None) or settings.cache_ttl_days
        if datetime.now(timezone.utc) - created_at > timedelta(days=ttl_days):
            logger.info("Cache entry expired for tenant=%s", tenant.slug)
            return None
    except Exception:
        return None

    # Validate source documents have not been re-ingested since cache was created
    source_doc_ids: List[str] = payload.get("source_document_ids", [])
    if source_doc_ids:
        try:
            result = await db.execute(
                select(Document).where(
                    Document.id.in_([
                        __import__("uuid").UUID(did) for did in source_doc_ids
                    ]),
                    Document.processed_at > created_at,
                )
            )
            stale_docs = result.scalars().all()
            if stale_docs:
                logger.info(
                    "Cache stale: %d source doc(s) re-ingested since cache entry for tenant=%s",
                    len(stale_docs),
                    tenant.slug,
                )
                return None
        except Exception as e:
            logger.warning("Cache staleness check failed: %s", e)
            return None

    logger.info(
        "Cache HIT for tenant=%s score=%.3f", tenant.slug, hit.score
    )
    return {
        "answer": payload.get("answer", ""),
        "citations": payload.get("citations", []),
        "escalated": payload.get("escalated", False),
        "escalation_pending": payload.get("escalation_pending", False),
        "escalation_summary": payload.get("escalation_summary"),
        "suggestions": payload.get("suggestions", []),
        "top_score": payload.get("top_score", 0.0),
        "cache_hit": True,
    }


async def store_cached_response(
    question_vector: List[float],
    question: str,
    result: Dict[str, Any],
    tenant: Tenant,
) -> None:
    """
    Store a successful RAG result in the cache collection.
    Only non-escalated answers are cached.
    """
    if not _is_cache_active():
        return

    if result.get("escalated", False) or result.get("escalation_pending", False):
        return  # Never cache escalations or pending escalations

    import uuid
    cache_col = _cache_collection(tenant.kb_namespace)

    source_doc_ids = list({
        c["document_id"] for c in result.get("citations", []) if "document_id" in c
    })

    payload = {
        "question": question,
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "escalated": result.get("escalated", False),
        "escalation_pending": result.get("escalation_pending", False),
        "escalation_summary": result.get("escalation_summary"),
        "suggestions": result.get("suggestions", []),
        "top_score": result.get("top_score", 0.0),
        "source_document_ids": source_doc_ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tenant_id": str(tenant.id),
    }

    try:
        from qdrant_client.models import PointStruct
        client = get_qdrant_client()
        await ensure_cache_collection(cache_col)
        await client.upsert(
            collection_name=cache_col,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=question_vector,
                    payload=payload,
                )
            ],
        )
        logger.info("Cached response for tenant=%s", tenant.slug)
    except Exception as e:
        logger.warning("Failed to store cache entry: %s", e)


async def invalidate_cache_for_document(document_id: str, tenant: Tenant) -> None:
    """
    Delete all cache entries that reference the given document_id.
    Called when a document is deleted or re-ingested.
    """
    if not _is_cache_active():
        return

    cache_col = _cache_collection(tenant.kb_namespace)
    try:
        await delete_cache_entries_by_document(cache_col, document_id)
        logger.info(
            "Invalidated cache entries for document_id=%s tenant=%s",
            document_id,
            tenant.slug,
        )
    except Exception as e:
        logger.warning("Cache invalidation failed for document_id=%s: %s", document_id, e)


# Import here to avoid circular at module level
from services.qdrant import delete_cache_entries_by_document  # noqa: E402
