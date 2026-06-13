from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from agent_tools import register_tool

# SECURITY: credentials.json is encrypted with a key derived from ANTHROPIC_API_KEY.
# Changing ANTHROPIC_API_KEY will make all stored credentials unreadable.
# This file should be in .gitignore — never commit it to version control.

logger = logging.getLogger(__name__)

CREDS_FILE = Path(__file__).resolve().parent.parent.parent / "memory" / "credentials.json"

# Service names must be alphanumeric + underscore + hyphen only.
# Anchored at both ends to prevent injection via a crafted service name.
_SERVICE_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")

# ---------------------------------------------------------------------------
# Key derivation — called once and cached at module level.
#
# We derive a deterministic 32-byte key from ANTHROPIC_API_KEY via SHA-256,
# then base64-url-encode it to produce a valid Fernet key.
#
# WARNING: if ANTHROPIC_API_KEY changes (e.g. rotated), all previously stored
# credentials become unreadable.  Re-store them after rotating the key.
# ---------------------------------------------------------------------------

def _get_fernet() -> Fernet:
    raw = os.getenv("ANTHROPIC_API_KEY", "fallback-key-not-for-production")
    key_material = hashlib.sha256(raw.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(key_material))


# Module-level singleton — initialised on first use so the env var is read
# after .env has been loaded by main.py's load_dotenv() call.
_FERNET: Fernet | None = None


def _fernet() -> Fernet:
    """Return the cached Fernet instance, creating it on the first call."""
    global _FERNET
    if _FERNET is None:
        _FERNET = _get_fernet()
    return _FERNET


# ---------------------------------------------------------------------------
# Private persistence helpers
# ---------------------------------------------------------------------------

def _load() -> dict:
    """
    Read the credentials store from disk.

    Returns {} silently on a missing file (first-run case).
    Logs a warning and returns {} on a corrupt / malformed file so a single
    bad write never bricks the whole credential system.
    """
    if not CREDS_FILE.exists():
        return {}
    try:
        text = CREDS_FILE.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object at root level")
        return data
    except Exception as exc:
        logger.warning(
            f"[credentials] Could not read credentials file (returning empty): {exc}"
        )
        return {}


def _save(data: dict) -> None:
    """
    Atomic write: write to a .tmp sibling file then os.rename() it into
    place so a crash mid-write can never corrupt the existing store.
    """
    CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CREDS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.rename(CREDS_FILE)
    logger.debug(f"[credentials] Saved {len(data)} credential(s) to disk.")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def store_credential(service: str, value: str) -> dict:
    """
    Encrypt and persist a credential for the given service name.

    Security contract:
      - The plaintext `value` is NEVER included in return values, logs, or
        any WebSocket event — only the service name and a success flag are
        returned to the frontend.
      - The encrypted blob is stored as a hex string so the JSON stays
        plain-text and inspectable (though not decryptable without the key).
    """
    if not _SERVICE_RE.match(service):
        return {
            "success": False,
            "error": (
                f"Invalid service name {service!r}. "
                "Only letters, digits, underscores, and hyphens are allowed."
            ),
        }

    encrypted_hex = _fernet().encrypt(value.encode()).hex()
    data = _load()
    data[service] = encrypted_hex
    _save(data)

    # Log only the service name — never the value.
    logger.info(f"[credentials] Stored credential for service: {service!r}")

    # SECURITY: `value` is intentionally omitted from the return dict.
    return {
        "success": True,
        "service": service,
        "message": "Credential stored securely.",
    }


async def get_credential(service: str) -> dict:
    """
    Decrypt and retrieve a stored credential by service name.

    Marked destructive so the user must approve every retrieval in the UI —
    this prevents the agent from silently exfiltrating secrets.
    """
    data = _load()
    if service not in data:
        return {
            "success": False,
            "error": (
                f"No credential for '{service}'. "
                "Store it first with store_credential."
            ),
        }

    try:
        decrypted = _fernet().decrypt(bytes.fromhex(data[service])).decode()
    except (InvalidToken, ValueError) as exc:
        logger.warning(
            f"[credentials] Decryption failed for {service!r}: {exc}"
        )
        return {
            "success": False,
            "error": (
                "Decryption failed. Credential may be corrupt or was stored "
                "with a different API key."
            ),
        }

    logger.info(f"[credentials] Retrieved credential for service: {service!r}")
    return {"success": True, "service": service, "value": decrypted}


async def list_credentials() -> dict:
    """
    Return the names of all stored credentials.
    Values are never exposed — names only.
    """
    data = _load()
    services = list(data.keys())
    return {"success": True, "services": services, "count": len(services)}


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register_credential_tools() -> None:
    register_tool(
        name="store_credential",
        description=(
            "Encrypt and store an API key or token under a service name. "
            "The service name must be alphanumeric with underscores/hyphens only. "
            "The value is encrypted with Fernet symmetric encryption before being "
            "written to disk — it is never logged or echoed back. "
            "Example service names: youtube_api_key, openai_api_key, slack_token. "
            "Always call list_credentials first to check if a key already exists."
        ),
        parameters={
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": (
                        "Unique name for this credential "
                        "(e.g. youtube_api_key, openai_key, slack_token)."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "The plaintext secret to encrypt and store.",
                },
            },
            "required": ["service", "value"],
        },
        handler=store_credential,
        destructive=True,
    )

    register_tool(
        name="get_credential",
        description=(
            "Decrypt and retrieve a stored credential by service name. "
            "Requires user approval on every call — credentials are never "
            "retrieved silently. Returns the decrypted value on success. "
            "If the service was never stored, returns an instructive error."
        ),
        parameters={
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "The service name used when the credential was stored.",
                },
            },
            "required": ["service"],
        },
        handler=get_credential,
        destructive=True,
    )

    register_tool(
        name="list_credentials",
        description=(
            "List the names of all stored credentials. "
            "Only service names are returned — values are never exposed. "
            "Use this to check what is already stored before calling store_credential."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=list_credentials,
        destructive=False,
    )

    logger.info(
        "[credentials] Registered tools: "
        "store_credential, get_credential, list_credentials"
    )
