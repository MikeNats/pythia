from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey, Identity
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.core.auth.models import Tenant, User


class MessageRole(StrEnum):
    user = "user"
    llm = "llm"


class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(default="New Conversation")
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id"))
    deleted_at: Mapped[datetime | None] = mapped_column(default=None)
    user: Mapped[User] = relationship(back_populates="conversations")
    tenant: Mapped[Tenant] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base, TimestampMixin):
    __tablename__ = "messages"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id"))
    seq: Mapped[int] = mapped_column(BigInteger, Identity(), index=True)
    role: Mapped[MessageRole]
    text: Mapped[str]
    conversation: Mapped[Conversation] = relationship(back_populates="messages")
