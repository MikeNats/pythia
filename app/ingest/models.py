import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector  # type: ignore
from sqlalchemy import ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.base import Base, TimestampMixin


class SourceType(StrEnum):
    web = "web"
    upload = "upload"
    sftp = "sftp"


class Document(Base, TimestampMixin):
    __tablename__ = "documents"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str]
    source_type: Mapped[SourceType]
    source_ref: Mapped[str]
    storage_uri: Mapped[str | None]
    content_type: Mapped[str]
    byte_size: Mapped[int | None]
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    deleted_at: Mapped[datetime | None]
    deleted_by: Mapped[uuid.UUID | None]
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document")


class Chunk(Base, TimestampMixin):
    __tablename__ = "chunks"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"))
    position: Mapped[int]
    text: Mapped[str]
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384))
    document: Mapped["Document"] = relationship(back_populates="chunks")
    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
