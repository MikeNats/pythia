from collections.abc import AsyncGenerator
from logging import getLogger
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth.models import ApiKey
from app.core.db import get_session
from app.core.request_context import get_request_id, tenant_id_var, user_id_var

logger = getLogger("auth")

DbSession = Annotated[AsyncSession, Depends(get_session)]
bearer_scheme = HTTPBearer(auto_error=False)


class UserSession(BaseModel):
    user_id: UUID
    tenant_id: UUID


def _unauthorized(reason: str = "Unauthorized") -> HTTPException:
    logger.warning("auth failed: %s", reason, extra={"request_id": get_request_id()})

    return HTTPException(
        status_code=401,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    session: DbSession,
) -> AsyncGenerator[UserSession]:
    if credentials is None:
        raise _unauthorized("Missing or invalid Authorization header")
    token_str = credentials.credentials  # Bearer

    key = await session.execute(
        select(ApiKey)
        .where(ApiKey.key_hash == ApiKey.hash_key(token_str))
        .where(ApiKey.revoked_at.is_(None))
        .where((ApiKey.expires_at.is_(None)) | (ApiKey.expires_at > func.now()))
    )
    api_key = key.scalar_one_or_none()
    if not api_key:
        raise _unauthorized("Invalid or expired token")

    user = UserSession(user_id=api_key.user_id, tenant_id=api_key.tenant_id)
    tenant_id_var.set(user.tenant_id)
    user_id_var.set(user.user_id)
    try:
        yield user
    finally:
        tenant_id_var.set(None)
        user_id_var.set(None)


CurrentUser = Annotated[UserSession, Depends(get_current_user)]
