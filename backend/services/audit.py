import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


async def log_action(
    tenant_id: str,
    action: str,
    db,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
) -> None:
    """
    Append an audit log entry. Always succeeds — never poisons the caller's transaction.
    Uses a SAVEPOINT (nested transaction) so a failed audit write rolls back only
    the audit entry, leaving the caller's outer transaction intact.

    action values:
        document.upload, document.delete, document.approve, document.reject
        user.invite, user.update, user.deactivate
        ticket.update
        config.update
    """
    try:
        from models.audit import AuditLog
        import uuid

        entry = AuditLog(
            tenant_id=uuid.UUID(tenant_id),
            user_id=uuid.UUID(user_id) if user_id else None,
            user_email=user_email,
            action=action,
            target_type=target_type,
            target_id=target_id,
            detail=detail or {},
            ip_address=ip_address,
        )
        async with db.begin_nested():   # SAVEPOINT — rolls back only this if it fails
            db.add(entry)
    except Exception as e:
        logger.warning("Audit log failed for action=%s: %s", action, e)
