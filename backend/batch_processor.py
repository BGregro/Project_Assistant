"""
batch_processor.py  —  Phase 11.5b: Batch Processing Infrastructure

Thin wrapper around the Anthropic Message Batches API, used for any
background/overnight work where latency doesn't matter but cost does
(batch requests are billed at 50% of the standard per-token price).

Used by:
  - agent_tools/batch_tools.py (backfill_reflections, Phase 11.5c)
  - Phase 16b memory summarization
  - Phase 17a strategy extraction
  - Phase 17d knowledge synthesis reports

All functions are async (except the small sync JSON helpers) and use the
same anthropic.AsyncAnthropic client pattern as agent_core.py — the API
key is read from the ANTHROPIC_API_KEY environment variable automatically,
never hardcoded.

Batch lifecycle:
  1. submit_batch(requests, job_name) -> batch_id
     Persists batch metadata (id, job_name, status, submitted_at) to
     memory/pending_batches.json so it survives a server restart.
  2. poll_batch(batch_id) -> status dict
     Cheap, can be called frequently (e.g. every 30 min by APScheduler).
  3. get_results(batch_id) -> list of per-request results
     Only meaningful once poll_batch() reports status == "ended".
  4. cancel_batch(batch_id) -> None
     Best-effort cancellation of an in-flight batch.

pending_batches.json schema (list of dicts):
    {
        "batch_id":     str,
        "job_name":     str,
        "status":       "submitted" | "ended" | "retrieved" | "cancelled" | "error",
        "submitted_at": ISO timestamp,
        "retrieved_at": ISO timestamp | None,
    }
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

BATCHES_FILE = Path(__file__).resolve().parent.parent / "memory" / "pending_batches.json"

# Module-level client — created lazily so importing this module never
# requires ANTHROPIC_API_KEY to already be set (e.g. during tests).
_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        # AsyncAnthropic reads ANTHROPIC_API_KEY from the environment automatically.
        _client = anthropic.AsyncAnthropic()
    return _client


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_batches() -> list[dict]:
    """Read pending_batches.json, tolerating a missing or corrupt file."""
    if not BATCHES_FILE.exists():
        return []
    try:
        with open(BATCHES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        logger.warning("[batch_processor] pending_batches.json is not a list — resetting.")
        return []
    except Exception as e:
        logger.warning(f"[batch_processor] Failed to read pending_batches.json (non-fatal): {e}")
        return []


def _save_batches(batches: list[dict]) -> None:
    """Atomically write pending_batches.json via a .tmp rename."""
    try:
        BATCHES_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = BATCHES_FILE.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(batches, f, ensure_ascii=False, indent=2)
        tmp_path.replace(BATCHES_FILE)
    except Exception as e:
        logger.error(f"[batch_processor] Failed to save pending_batches.json (non-fatal): {e}")


def _save_batch(batch_id: str, job_name: str, status: str = "submitted") -> None:
    """Persist batch metadata to BATCHES_FILE. Appends a new entry."""
    batches = _load_batches()
    batches.append({
        "batch_id":     batch_id,
        "job_name":     job_name,
        "status":       status,
        "submitted_at": _now_iso(),
        "retrieved_at": None,
    })
    _save_batches(batches)
    logger.info(f"[batch_processor] Batch persisted: id={batch_id[:12]}..., job={job_name!r}")


def _update_batch_status(batch_id: str, status: str) -> None:
    """Update the status field of an existing persisted batch entry."""
    batches = _load_batches()
    for b in batches:
        if b.get("batch_id") == batch_id:
            b["status"] = status
            _save_batches(batches)
            return
    logger.warning(f"[batch_processor] _update_batch_status: batch {batch_id} not found.")


def _mark_retrieved(batch_id: str) -> None:
    """Mark a batch as retrieved in BATCHES_FILE."""
    batches = _load_batches()
    for b in batches:
        if b.get("batch_id") == batch_id:
            b["status"] = "retrieved"
            b["retrieved_at"] = _now_iso()
            _save_batches(batches)
            logger.info(f"[batch_processor] Batch marked retrieved: id={batch_id[:12]}...")
            return
    logger.warning(f"[batch_processor] _mark_retrieved: batch {batch_id} not found.")


def list_pending_batches() -> list[dict]:
    """Return all batches from BATCHES_FILE not yet marked 'retrieved'."""
    batches = _load_batches()
    return [b for b in batches if b.get("status") != "retrieved"]


# ---------------------------------------------------------------------------
# Anthropic Batches API wrappers
# ---------------------------------------------------------------------------

async def submit_batch(requests: list[dict], job_name: str) -> str:
    """
    Submit a list of requests as a batch. Returns batch_id.

    Args:
        requests: list of {"custom_id": str, "params": {model, max_tokens, messages, ...}}
                  "params" is passed straight through to the Anthropic Messages
                  API shape (model / max_tokens / messages / system / etc.).
        job_name: human-readable label stored alongside the batch for
                  list_batch_jobs() / debugging (e.g. "reflection_backfill").

    Raises:
        ValueError if requests is empty.
        anthropic.APIError subclasses on submission failure (not caught here —
        callers should wrap this in try/except, as batch_tools.py does).
    """
    if not requests:
        raise ValueError("submit_batch: requests list must not be empty.")

    client = _get_client()

    # Map our simplified request shape to the Anthropic batch request format.
    batch_requests = [
        {
            "custom_id": req["custom_id"],
            "params":    req["params"],
        }
        for req in requests
    ]

    batch = await client.messages.batches.create(requests=batch_requests)
    batch_id = batch.id

    _save_batch(batch_id, job_name, status="submitted")
    logger.info(
        f"[batch_processor] Submitted batch '{job_name}' with {len(requests)} "
        f"requests -> batch_id={batch_id}"
    )
    return batch_id


async def poll_batch(batch_id: str) -> dict:
    """
    Check the status of a batch.

    Returns:
        {
            "status":     "in_progress" | "ended" | "canceling" | "cancelled" | "error",
            "succeeded":  int,
            "errored":    int,
            "processing": int,
            "canceled":   int,
            "total":      int,
        }
    """
    client = _get_client()
    try:
        batch = await client.messages.batches.retrieve(batch_id)
    except Exception as e:
        logger.error(f"[batch_processor] poll_batch failed for {batch_id}: {e}")
        return {
            "status": "error", "succeeded": 0, "errored": 0,
            "processing": 0, "canceled": 0, "total": 0, "error": str(e),
        }

    counts = getattr(batch, "request_counts", None)
    succeeded  = getattr(counts, "succeeded", 0)  if counts else 0
    errored    = getattr(counts, "errored", 0)    if counts else 0
    processing = getattr(counts, "processing", 0) if counts else 0
    canceled   = getattr(counts, "canceled", 0)   if counts else 0
    total = succeeded + errored + processing + canceled

    status = batch.processing_status  # "in_progress" | "ended" | "canceling"

    if status == "ended":
        _update_batch_status(batch_id, "ended")

    return {
        "status":     status,
        "succeeded":  succeeded,
        "errored":    errored,
        "processing": processing,
        "canceled":   canceled,
        "total":      total,
    }


async def get_results(batch_id: str) -> list[dict]:
    """
    Retrieve results for a completed batch.

    Returns:
        [{"custom_id": str, "type": "success"|"error", "content": Any|None,
          "error": str|None}, ...]

    Should only be called once poll_batch() reports status == "ended" — the
    Anthropic API will raise if results aren't ready yet.
    """
    client = _get_client()
    results: list[dict] = []
    try:
        async for entry in await client.messages.batches.results(batch_id):
            custom_id = entry.custom_id
            result = entry.result
            result_type = getattr(result, "type", "error")

            if result_type == "succeeded":
                message = getattr(result, "message", None)
                text_parts = []
                if message is not None:
                    for block in getattr(message, "content", []) or []:
                        if getattr(block, "type", None) == "text":
                            text_parts.append(block.text)
                results.append({
                    "custom_id": custom_id,
                    "type":      "success",
                    "content":   "\n".join(text_parts),
                    "error":     None,
                })
            else:
                # errored / canceled / expired
                err_msg = getattr(result, "error", None)
                results.append({
                    "custom_id": custom_id,
                    "type":      "error",
                    "content":   None,
                    "error":     str(err_msg) if err_msg else f"Result type: {result_type}",
                })
    except Exception as e:
        logger.error(f"[batch_processor] get_results failed for {batch_id}: {e}")
        raise

    return results


async def cancel_batch(batch_id: str) -> None:
    """Best-effort cancellation of an in-flight batch."""
    client = _get_client()
    try:
        await client.messages.batches.cancel(batch_id)
        _update_batch_status(batch_id, "cancelled")
        logger.info(f"[batch_processor] Batch cancelled: {batch_id}")
    except Exception as e:
        logger.error(f"[batch_processor] cancel_batch failed for {batch_id}: {e}")
        raise
