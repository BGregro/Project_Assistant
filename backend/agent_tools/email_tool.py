"""
agent_tools/email_tool.py  —  Phase 9d: IMAP Email Inbox Management Tools

Connects to an IMAP server to scan, classify, and delete old emails.
Uses only the Python standard library (imaplib, email) — no extra dependencies.

Workflow:
    1. email_connect(host, username)             — connect and authenticate
    2. email_scan_inbox(max_emails, older_than_days)  — find old emails, categorize by header
    3. email_classify_and_plan(email_ids)        — local LLM decides DELETE or KEEP per ID
    4. SHOW user the plan; get explicit approval
    5. email_delete_batch(email_ids, dry_run=False)   — only after user approval
    6. email_disconnect()                        — clean up

SAFETY:
    - email_delete_batch defaults to dry_run=True — it NEVER deletes without explicit
      dry_run=False AND the user must approve the tool call (it is marked destructive).
    - Headers only are fetched — the full message body is never read or sent to any LLM.
    - Passwords are retrieved from the credential manager, never hardcoded.

Tools registered:
    email_connect(host, username, credential_service)   — destructive (auth session)
    email_scan_inbox(max_emails, older_than_days)       — non-destructive
    email_classify_and_plan(email_ids, category_hint)  — non-destructive
    email_delete_batch(email_ids, dry_run)              — destructive
    email_disconnect()                                  — non-destructive
"""

import email as _email_module
import imaplib
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional

from agent_tools import register_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level IMAP connection state
# ---------------------------------------------------------------------------

_imap_connection: Optional[imaplib.IMAP4_SSL] = None
_connected_host:  Optional[str]  = None
_connected_user:  Optional[str]  = None


def _require_connection() -> dict | None:
    """Return an error dict if not connected, else None."""
    if _imap_connection is None:
        return {
            "success": False,
            "error": (
                "No active IMAP connection. "
                "Call email_connect(host, username) first."
            ),
        }
    return None


# ---------------------------------------------------------------------------
# Categorisation helpers (no LLM — pure keyword/domain heuristics)
# ---------------------------------------------------------------------------

_NEWSLETTER_KEYWORDS  = {"newsletter", "unsubscribe", "weekly digest", "monthly digest",
                          "update from", "news from", "dispatch", "roundup"}
_NOTIFICATION_KEYWORDS = {"notification", "alert", "action required", "your account",
                           "password", "verify", "confirmation", "code:", "otp"}
_PROMOTION_KEYWORDS   = {"sale", "% off", "discount", "offer", "deal", "promo",
                          "coupon", "limited time", "shop now", "buy now", "free shipping",
                          "subscribe", "special offer"}
_RECEIPT_KEYWORDS     = {"receipt", "invoice", "order #", "order confirmation",
                          "payment", "your order", "shipped", "delivered", "tracking",
                          "refund", "billing"}

def _categorise_email(subject: str, sender: str) -> str:
    """
    Categorise an email into one of: newsletters, notifications, promotions,
    receipts, unknown — based on subject and sender keywords alone.
    No LLM call needed; this is just for the overview bucketing.
    """
    text = (subject + " " + sender).lower()

    for kw in _NEWSLETTER_KEYWORDS:
        if kw in text:
            return "newsletters"
    for kw in _RECEIPT_KEYWORDS:
        if kw in text:
            return "receipts"
    for kw in _PROMOTION_KEYWORDS:
        if kw in text:
            return "promotions"
    for kw in _NOTIFICATION_KEYWORDS:
        if kw in text:
            return "notifications"
    return "unknown"


def _parse_date_header(date_str: str) -> Optional[datetime]:
    """Parse an email Date header into a UTC datetime.  Returns None on failure."""
    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def email_connect(
    host: str,
    username: str,
    credential_service: str = "email_password",
) -> dict:
    """
    Connect to an IMAP server over SSL and authenticate.

    For Gmail: host = "imap.gmail.com"
    Password must be stored via store_credential(credential_service, value).
    For Gmail, use an App Password (not your regular password).
    """
    global _imap_connection, _connected_host, _connected_user

    # Disconnect any existing session first
    if _imap_connection is not None:
        try:
            _imap_connection.logout()
        except Exception:
            pass
        _imap_connection = None

    # Retrieve password from credential manager
    try:
        from agent_tools.credentials import get_credential
        cred_result = await get_credential(credential_service)
        if not cred_result.get("success"):
            return {
                "success": False,
                "error": (
                    f"Could not retrieve credential '{credential_service}': "
                    f"{cred_result.get('error', 'unknown error')}. "
                    f"Store it first with store_credential('{credential_service}', '<password>')."
                ),
            }
        password = cred_result.get("value", "")
        if not password:
            return {
                "success": False,
                "error": f"Credential '{credential_service}' is empty.",
            }
    except Exception as e:
        return {"success": False, "error": f"Credential retrieval failed: {e}"}

    try:
        conn = imaplib.IMAP4_SSL(host)
        conn.login(username, password)

        # List mailboxes to return a count
        status, mailbox_list = conn.list()
        mailbox_count = len(mailbox_list) if status == "OK" and mailbox_list else 0

        _imap_connection = conn
        _connected_host  = host
        _connected_user  = username

        logger.info(f"[email] Connected to {host} as {username} ({mailbox_count} mailboxes)")
        return {
            "success":       True,
            "host":          host,
            "username":      username,
            "mailbox_count": mailbox_count,
        }
    except imaplib.IMAP4.error as e:
        return {"success": False, "error": f"IMAP authentication failed: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def email_scan_inbox(
    max_emails: int = 500,
    older_than_days: int = 30,
) -> dict:
    """
    Scan INBOX for emails older than older_than_days.

    Fetches headers only (never full body).
    Returns a categorised summary with sample subjects and IDs per category.
    """
    err = _require_connection()
    if err:
        return err

    try:
        # Select INBOX
        status, _ = _imap_connection.select("INBOX", readonly=True)
        if status != "OK":
            return {"success": False, "error": "Could not select INBOX."}

        # Build BEFORE date string in DD-Mon-YYYY format
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        before_str = cutoff.strftime("%d-%b-%Y")

        # Search for old emails
        status, data = _imap_connection.search(None, f'BEFORE "{before_str}"')
        if status != "OK":
            return {"success": False, "error": "IMAP SEARCH failed."}

        all_ids = data[0].split() if data[0] else []
        # Apply max_emails cap from the end (most recent within the old bucket)
        ids_to_scan = all_ids[-max_emails:] if len(all_ids) > max_emails else all_ids

        if not ids_to_scan:
            return {
                "success":     True,
                "total_found": 0,
                "categories":  {"newsletters": 0, "notifications": 0,
                                 "promotions": 0, "receipts": 0, "unknown": 0},
                "sample_subjects":      {},
                "email_ids_by_category": {},
            }

        # Fetch headers in batch
        id_str = b",".join(ids_to_scan)
        status, messages = _imap_connection.fetch(
            id_str, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"
        )
        if status != "OK":
            return {"success": False, "error": "IMAP FETCH failed."}

        # Parse and categorise
        categories: dict[str, list[str]] = {
            "newsletters": [], "notifications": [],
            "promotions": [], "receipts": [], "unknown": [],
        }
        email_ids_by_category: dict[str, list[str]] = {k: [] for k in categories}
        sample_subjects: dict[str, list[str]] = {k: [] for k in categories}

        # messages alternates between header data tuples and b')' separators
        for item in messages:
            if not isinstance(item, tuple):
                continue
            raw_id_part, raw_header = item
            # raw_id_part looks like b"123 (BODY[HEADER.FIELDS ...]"
            try:
                email_id = raw_id_part.decode().split()[0]
            except Exception:
                continue

            parsed = _email_module.message_from_bytes(raw_header)
            subject = parsed.get("Subject", "(no subject)") or "(no subject)"
            sender  = parsed.get("From",    "") or ""
            category = _categorise_email(subject, sender)

            categories[category].append(email_id)
            email_ids_by_category[category].append(email_id)
            if len(sample_subjects[category]) < 5:
                sample_subjects[category].append(subject[:80])

        logger.info(
            f"[email] Scanned {len(ids_to_scan)} emails older than {older_than_days} days"
        )
        return {
            "success":               True,
            "total_found":           len(ids_to_scan),
            "older_than_days":       older_than_days,
            "categories":            {k: len(v) for k, v in categories.items()},
            "sample_subjects":       sample_subjects,
            "email_ids_by_category": email_ids_by_category,
        }
    except imaplib.IMAP4.error as e:
        return {"success": False, "error": f"IMAP error: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def email_classify_and_plan(
    email_ids: str,
    category_hint: str = "",
) -> dict:
    """
    Send a batch of email headers to the local LLM to classify each as
    DELETE or KEEP.

    email_ids: comma-separated IMAP message IDs (max 100 per call).
    The local model sees only From/Subject/Date — never body content.
    Returns: {to_delete: [ids], to_keep: [ids], unsure: [ids], summary: str}
    """

    cfg_path = Path(__file__).resolve().parent.parent.parent / "config.json"
    config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    batch_size = config.get("email", {}).get("classify_batch_size", 20)

    err = _require_connection()
    if err:
        return err

    ids = [i.strip() for i in email_ids.split(",") if i.strip()]
    if not ids:
        return {"success": False, "error": "email_ids is empty."}
    if len(ids) > batch_size:
        ids = ids[:batch_size]
        logger.warning("[email] Truncated classify batch to 100 IDs.")

    try:
        # Fetch headers for this batch
        id_str = ",".join(ids).encode()
        status, messages = _imap_connection.fetch(
            id_str, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"
        )
        if status != "OK":
            return {"success": False, "error": "IMAP FETCH failed."}

        # Build a compact header block for the local LLM
        header_lines: list[str] = []
        id_order: list[str] = []

        for item in messages:
            if not isinstance(item, tuple):
                continue
            raw_id_part, raw_header = item
            try:
                email_id = raw_id_part.decode().split()[0]
            except Exception:
                continue
            parsed  = _email_module.message_from_bytes(raw_header)
            subject = parsed.get("Subject", "(no subject)")[:80]
            sender  = parsed.get("From", "")[:60]
            date    = parsed.get("Date", "")[:30]
            header_lines.append(f"ID:{email_id} | From:{sender} | Subj:{subject} | Date:{date}")
            id_order.append(email_id)

        if not header_lines:
            return {"success": False, "error": "No headers could be fetched for the given IDs."}

        headers_block = "\n".join(header_lines)
        hint_part = f" Category hint: {category_hint}." if category_hint else ""

        prompt = (
            "/no_think\n\n"  # Disables qwen3 chain-of-thought — plain per-email classification
            f"Classify each email as DELETE (clearly unwanted: newsletter, promotion, "
            f"automated notification, old receipt) or KEEP (personal, important, from a real person). "
            f"{hint_part}"
            f"For each email ID, reply with only: ID: DELETE or ID: KEEP. "
            f"One per line. No other text.\n\nEmails:\n{headers_block}"
        )

        # Call local LLM for classification
        try:
            from agent_tools.local_llm import local_llm_call
            import json as _json
            import pathlib
            cfg_path = pathlib.Path(__file__).resolve().parent.parent.parent / "config.json"
            cfg = _json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            local_model = cfg.get("llm", {}).get("local", "qwen2.5:14b")
            ollama_url  = cfg.get("ollama_base_url", "http://localhost:11434")
            response = await local_llm_call(prompt, model=local_model, base_url=ollama_url)
        except Exception as e:
            logger.warning(f"[email] Local LLM classify failed: {e}")
            response = None

        to_delete: list[str] = []
        to_keep:   list[str] = []
        unsure:    list[str] = []

        if response:
            for line in response.splitlines():
                line = line.strip()
                if ":" not in line:
                    continue
                parts = line.split(":", 1)
                eid    = parts[0].strip()
                verdict = parts[1].strip().upper() if len(parts) > 1 else ""
                if eid not in id_order:
                    continue
                if "DELETE" in verdict:
                    to_delete.append(eid)
                elif "KEEP" in verdict:
                    to_keep.append(eid)
                else:
                    unsure.append(eid)

            # Any IDs the LLM didn't mention → unsure
            mentioned = set(to_delete + to_keep + unsure)
            for eid in id_order:
                if eid not in mentioned:
                    unsure.append(eid)
        else:
            # LLM unavailable — mark all as unsure
            unsure = list(id_order)

        summary = (
            f"Classified {len(ids)} emails: "
            f"{len(to_delete)} to delete, {len(to_keep)} to keep, {len(unsure)} unsure."
        )
        logger.info(f"[email] {summary}")

        return {
            "success":   True,
            "to_delete": to_delete,
            "to_keep":   to_keep,
            "unsure":    unsure,
            "summary":   summary,
            "batch_size": len(ids),
        }
    except imaplib.IMAP4.error as e:
        return {"success": False, "error": f"IMAP error: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def email_delete_batch(
    email_ids: str,
    dry_run: bool = True,
) -> dict:
    """
    Mark emails for deletion and expunge them from the mailbox.

    IMPORTANT SAFETY RULES:
    - Always use dry_run=True first to preview what will be deleted.
    - Only set dry_run=False AFTER the user has reviewed and approved the list.
    - For Gmail, emails are copied to [Gmail]/Trash before marking as deleted.
    - This tool is marked destructive and requires user approval every time.
    """
    err = _require_connection()
    if err:
        return err

    ids = [i.strip() for i in email_ids.split(",") if i.strip()]
    if not ids:
        return {"success": False, "error": "email_ids is empty."}

    if dry_run:
        logger.info(f"[email] DRY RUN: would delete {len(ids)} emails: {ids[:10]}…")
        return {
            "success":       True,
            "dry_run":       True,
            "would_delete":  ids,
            "deleted_count": 0,
            "message": (
                f"DRY RUN: {len(ids)} emails would be deleted. "
                "Call again with dry_run=False to actually delete them — "
                "but only after showing this list to the user and getting explicit approval."
            ),
        }

    # --- Actual deletion ---
    try:
        _imap_connection.select("INBOX")
    except imaplib.IMAP4.error as e:
        return {"success": False, "error": f"Could not select INBOX: {e}"}

    deleted = 0
    errors  = []

    # Detect Gmail by host name
    is_gmail = _connected_host and "gmail" in _connected_host.lower()

    for eid in ids:
        try:
            if is_gmail:
                # Gmail: copy to Trash first so the email appears there
                try:
                    _imap_connection.copy(eid, "[Gmail]/Trash")
                except imaplib.IMAP4.error:
                    pass  # non-fatal — some Gmail variants use different Trash names

            # Mark as deleted
            _imap_connection.store(eid, "+FLAGS", "\\Deleted")
            deleted += 1
        except imaplib.IMAP4.error as e:
            errors.append(f"ID {eid}: {e}")
            logger.warning(f"[email] Could not delete email {eid}: {e}")

    # Expunge to finalise deletions
    try:
        _imap_connection.expunge()
    except imaplib.IMAP4.error as e:
        logger.warning(f"[email] Expunge warning (non-fatal): {e}")

    logger.info(f"[email] Deleted {deleted}/{len(ids)} emails.")
    result = {
        "success":       True,
        "dry_run":       False,
        "deleted_count": deleted,
        "email_ids":     ids,
    }
    if errors:
        result["errors"] = errors
    return result


async def email_disconnect() -> dict:
    """Close the active IMAP connection."""
    global _imap_connection, _connected_host, _connected_user

    if _imap_connection is None:
        return {"success": True, "message": "No active connection to close."}

    try:
        _imap_connection.logout()
    except Exception as e:
        logger.debug(f"[email] Logout warning (non-fatal): {e}")

    _imap_connection = None
    _connected_host  = None
    _connected_user  = None

    logger.info("[email] Disconnected from IMAP server.")
    return {"success": True, "message": "Disconnected from IMAP server."}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_email_tools() -> None:
    """Register IMAP email tools into the agent tool registry."""

    register_tool(
        name="email_connect",
        description=(
            "Connect to an IMAP server over SSL and authenticate. "
            "For Gmail use host='imap.gmail.com' and store an App Password via "
            "store_credential('gmail_password') — never use your regular password. "
            "credential_service defaults to 'email_password'. "
            "Must be called before any other email_* tools."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "IMAP hostname, e.g. 'imap.gmail.com', 'imap.mail.yahoo.com'.",
                },
                "username": {
                    "type": "string",
                    "description": "Email address / IMAP username.",
                },
                "credential_service": {
                    "type": "string",
                    "description": "Credential service name to look up the password. Default: 'email_password'.",
                    "default": "email_password",
                },
            },
            "required": ["host", "username"],
        },
        handler=email_connect,
        destructive=True,  # establishes an authenticated session
    )

    register_tool(
        name="email_scan_inbox",
        description=(
            "Scan INBOX for emails older than older_than_days. "
            "Fetches headers only (From, Subject, Date) — never full body. "
            "Returns a categorised count and sample subjects per category: "
            "newsletters, notifications, promotions, receipts, unknown. "
            "Requires an active connection from email_connect()."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "max_emails": {
                    "type": "integer",
                    "description": "Maximum number of old emails to scan. Default: 500.",
                    "default": 500,
                },
                "older_than_days": {
                    "type": "integer",
                    "description": "Only scan emails older than this many days. Default: 30.",
                    "default": 30,
                },
            },
            "required": [],
        },
        handler=email_scan_inbox,
        destructive=False,
    )

    register_tool(
        name="email_classify_and_plan",
        description=(
            "Send a batch of email headers to the local LLM to classify each as "
            "DELETE or KEEP. Returns {to_delete, to_keep, unsure, summary}. "
            "email_ids is a comma-separated list of IMAP IDs (max 100 per call). "
            "Headers only — no body content ever leaves the machine. "
            "ALWAYS show the plan to the user and get approval before calling "
            "email_delete_batch(). Non-destructive."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "email_ids": {
                    "type": "string",
                    "description": "Comma-separated IMAP email IDs from email_scan_inbox results.",
                },
                "category_hint": {
                    "type": "string",
                    "description": "Optional hint like 'newsletters' to guide classification.",
                    "default": "",
                },
            },
            "required": ["email_ids"],
        },
        handler=email_classify_and_plan,
        destructive=False,
    )

    register_tool(
        name="email_delete_batch",
        description=(
            "Mark emails as deleted and expunge from mailbox. "
            "ALWAYS use dry_run=True first to preview — this is the default. "
            "Only set dry_run=False after the user has reviewed and explicitly approved "
            "the delete list from email_classify_and_plan(). "
            "For Gmail, copies to Trash before deleting. "
            "NEVER call with dry_run=False without prior user approval."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "email_ids": {
                    "type": "string",
                    "description": "Comma-separated IMAP email IDs to delete.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": (
                        "If true (default), only previews what would be deleted without deleting. "
                        "Set to false ONLY after user has approved the specific deletion list."
                    ),
                    "default": True,
                },
            },
            "required": ["email_ids"],
        },
        handler=email_delete_batch,
        destructive=True,
    )

    register_tool(
        name="email_disconnect",
        description="Close the active IMAP connection. Call when done with email operations.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=email_disconnect,
        destructive=False,
    )

    logger.info(
        "[email] Registered tools: email_connect, email_scan_inbox, "
        "email_classify_and_plan, email_delete_batch, email_disconnect"
    )
