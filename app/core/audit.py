import logging
from typing import Any

from app.core.config import settings
from app.core.db import EventType, SessionAudit, SessionLocal
from app.core.request_context import get_request_id

logger = logging.getLogger("audit")
logger.setLevel(settings.logging_level)


async def audit(
    message: str,
    data: dict[str, Any],
    *,
    event_type: EventType,
    session_id: str | None = None,
) -> None:
    if settings.logging:
        logger.info(message, extra={"audit": data})

    if not settings.audit_enabled:  # e.g. tests — don't touch the DB
        return

    request_id = get_request_id()
    if request_id is None:
        logger.warning("no request_id in context — skipping audit row")
        return

    try:
        async with SessionLocal() as session:
            session.add(
                SessionAudit(
                    request_id=request_id,
                    event_type=event_type,
                    data=data,
                    session_id=session_id,
                )
            )
            await session.commit()
    except Exception:
        logger.exception("audit write failed")
