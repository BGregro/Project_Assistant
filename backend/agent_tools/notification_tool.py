"""
agent_tools/notification_tool.py  —  Phase 9b: Email Notification Tools

Sends email notifications via SMTP (with STARTTLS).
Reads SMTP configuration from config.json under the "email" key.
Passwords are retrieved from the credential manager (never hardcoded).

Config layout in config.json:
    {
        "email": {
            "smtp_host":    "smtp.gmail.com",
            "smtp_port":    587,
            "from_address": "",
            "to_address":   "",
            "enabled":      false
        }
    }

To enable Gmail:
    1. Generate an App Password in your Google Account settings.
    2. Call store_credential("email_password", "<app_password>") via the agent.
    3. Set from_address and to_address in config.json.
    4. Set enabled=true in config.json (or via the settings panel).

Tools registered:
    send_email(subject, body, to_address)  — destructive
    test_email_config()                    — non-destructive
"""

import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from agent_tools import register_tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader  (reads the live config file on every call so hot-changes work)
# ---------------------------------------------------------------------------

def _load_email_config() -> dict:
    """
    Load the "email" section from config.json.
    Returns sensible defaults if the section or file is missing.
    """
    import json
    config_path = Path(__file__).resolve().parent.parent.parent / "config.json"
    defaults = {
        "smtp_host":    "smtp.gmail.com",
        "smtp_port":    587,
        "from_address": "",
        "to_address":   "",
        "enabled":      False,
    }
    if not config_path.exists():
        return defaults
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        cfg = raw.get("email", {})
        return {**defaults, **cfg}
    except Exception as e:
        logger.warning(f"[notification] Could not load email config: {e}")
        return defaults


# ---------------------------------------------------------------------------
# Credential retrieval helper
# ---------------------------------------------------------------------------

async def _get_password(service: str = "email_password") -> str | None:
    """
    Retrieve the SMTP password from the credential store.
    Returns None if not found or if the credential module is unavailable.
    """
    try:
        from agent_tools.credentials import get_credential
        result = await get_credential(service)
        if result.get("success") and result.get("value"):
            return result["value"]
        return None
    except Exception as e:
        logger.warning(f"[notification] Could not retrieve credential '{service}': {e}")
        return None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def send_email(
    subject: str,
    body: str,
    to_address: str = "",
) -> dict:
    """
    Send a plain-text email via SMTP with STARTTLS.

    - SMTP host/port/from_address come from config.json["email"].
    - to_address defaults to config.json["email"]["to_address"] if not provided.
    - Password is read from the credential manager (service: "email_password").
    - Password and full body are NEVER logged.
    """
    cfg = _load_email_config()

    smtp_host    = cfg["smtp_host"]
    smtp_port    = int(cfg["smtp_port"])
    from_address = cfg["from_address"]
    recipient    = to_address.strip() if to_address.strip() else cfg["to_address"]

    # Validate required fields before attempting connection
    if not from_address:
        return {
            "success": False,
            "error": (
                "from_address is not configured. "
                "Set it in config.json under 'email.from_address'."
            ),
        }
    if not recipient:
        return {
            "success": False,
            "error": (
                "No recipient address provided and config.json 'email.to_address' is empty. "
                "Pass to_address or set a default in config.json."
            ),
        }

    password = await _get_password("email_password")
    if not password:
        return {
            "success": False,
            "error": (
                "Email password not found. Store it first:\n"
                "  store_credential('email_password', '<your_app_password>')\n"
                "For Gmail, generate an App Password at "
                "https://myaccount.google.com/apppasswords (requires 2-Step Verification)."
            ),
        }

    try:
        msg = MIMEMultipart()
        msg["From"]    = from_address
        msg["To"]      = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # Use STARTTLS (port 587) — more universally supported than SSL (port 465)
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            server.login(from_address, password)
            server.sendmail(from_address, recipient, msg.as_string())

        logger.info(f"[notification] Email sent → {recipient}: {subject!r}")
        return {
            "success": True,
            "to": recipient,
            "subject": subject,
        }
    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "error": (
                "SMTP authentication failed. "
                "Check the stored password and ensure you are using an App Password "
                "(not your regular account password) for Gmail."
            ),
        }
    except smtplib.SMTPException as e:
        return {"success": False, "error": f"SMTP error: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def test_email_config() -> dict:
    """
    Test SMTP connectivity and authentication without sending a message.

    Connects, upgrades to TLS, and authenticates — then disconnects without
    sending any email.  Useful for validating config before enabling notifications.
    """
    cfg = _load_email_config()
    smtp_host    = cfg["smtp_host"]
    smtp_port    = int(cfg["smtp_port"])
    from_address = cfg["from_address"]

    if not from_address:
        return {
            "success": False,
            "error": "from_address is not configured in config.json['email'].",
        }

    password = await _get_password("email_password")
    if not password:
        return {
            "success": False,
            "error": (
                "Email password not found. "
                "Store it with store_credential('email_password', '<app_password>')."
            ),
        }

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            server.login(from_address, password)
            # Quit cleanly without sending anything

        logger.info(f"[notification] Email config test passed (host={smtp_host}:{smtp_port})")
        return {
            "success": True,
            "message": (
                f"SMTP connection and authentication successful. "
                f"Host: {smtp_host}:{smtp_port}, Account: {from_address}"
            ),
        }
    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "error": "Authentication failed — check password and use an App Password for Gmail.",
        }
    except smtplib.SMTPException as e:
        return {"success": False, "error": f"SMTP error: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_notification_tools() -> None:
    """Register email notification tools into the agent tool registry."""

    register_tool(
        name="send_email",
        description=(
            "Send a plain-text email notification via SMTP (STARTTLS). "
            "SMTP settings come from config.json['email']. "
            "Password must be stored via store_credential('email_password'). "
            "For Gmail, use an App Password (not your regular password). "
            "to_address defaults to config.json['email']['to_address'] if not provided."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Email subject line.",
                },
                "body": {
                    "type": "string",
                    "description": "Plain-text email body.",
                },
                "to_address": {
                    "type": "string",
                    "description": (
                        "Recipient email address. "
                        "Defaults to config.json['email']['to_address'] if empty."
                    ),
                    "default": "",
                },
            },
            "required": ["subject", "body"],
        },
        handler=send_email,
        destructive=True,
    )

    register_tool(
        name="test_email_config",
        description=(
            "Test SMTP connectivity and authentication without sending a message. "
            "Use this to validate email configuration before enabling notifications. "
            "Requires from_address in config.json['email'] and "
            "store_credential('email_password') to have been called."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=test_email_config,
        destructive=False,
    )

    logger.info("[notification] Registered tools: send_email, test_email_config")
