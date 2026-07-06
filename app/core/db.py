import uuid
from collections.abc import AsyncGenerator
from enum import StrEnum
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Mapped, mapped_column

from app.core.base import Base, TimestampMixin
from app.core.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=settings.db_echo,
    pool_pre_ping=True,  # validate a pooled conn before use — survives DB restarts
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# from sqlalchemy import event
#
# @event.listens_for(engine.sync_engine, "connect")
# def _tune_pgvector(dbapi_conn, _connection_record):  # type: ignore[no-untyped-def]
#     dbapi_conn.cursor().execute("SET hnsw.iterative_scan = strict_order")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


class EventType(StrEnum):
    http_request = "http_request"
    http_response = "http_response"
    http_error = "http_error"
    llm_call = "llm_call"
    tool_call = "tool_call"


class SessionAudit(TimestampMixin, Base):  # when
    __tablename__ = "session_audit"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    request_id: Mapped[str] = mapped_column(index=True)  # correlation id
    session_id: Mapped[uuid.UUID | None] = mapped_column(index=True)  # conversation
    user_id: Mapped[uuid.UUID | None] = mapped_column(index=True)  # who
    tenant_id: Mapped[uuid.UUID | None] = mapped_column(index=True)  # isolation
    event_type: Mapped[EventType] = mapped_column(index=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
