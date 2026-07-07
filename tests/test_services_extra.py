"""Extra service-layer tests for app/retrieval/services.py.

Covers conversation detail/title/delete flows (tenant isolation) plus the
single_query_to_answer RAG path with a real embedded chunk and the empty-DB
no-hits branch. FakeLLM keeps everything off any real model.
"""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import UserSession
from app.ingest.models import Chunk, Document, SourceType
from app.retrieval.embedder import get_embedder
from app.retrieval.exceptions import ConversationNotFound
from app.retrieval.models import MessageRole
from app.retrieval.schemas import QuestionResponse
from app.retrieval.services import (
    delete_conversation,
    get_conversation_detail,
    get_conversation_messages,
    get_or_create_conversation,
    save_message,
    set_conversation_title,
    single_query_to_answer,
)
from tests.conftest import FakeLLM


def _other_tenant() -> UserSession:
    """A caller from a different tenant (random ids)."""
    return UserSession(user_id=uuid.uuid4(), tenant_id=uuid.uuid4())


async def test_conversation_detail_has_messages_in_seq_order(
    session: AsyncSession, user: UserSession
) -> None:
    cid = uuid.uuid4()
    await get_or_create_conversation(session, cid, user)
    await save_message(cid, "first", MessageRole.user, session)
    await save_message(cid, "second", MessageRole.llm, session)

    conv = await get_conversation_detail(session, cid, user)

    assert conv.id == cid
    assert len(conv.messages) == 2
    ordered = sorted(conv.messages, key=lambda m: m.seq)
    assert [m.text for m in ordered] == ["first", "second"]


async def test_conversation_detail_unknown_id_raises(
    session: AsyncSession, user: UserSession
) -> None:
    with pytest.raises(ConversationNotFound):
        await get_conversation_detail(session, uuid.uuid4(), user)


async def test_set_title_returns_updated_title(
    session: AsyncSession, user: UserSession
) -> None:
    cid = uuid.uuid4()
    await get_or_create_conversation(session, cid, user)

    conv = await set_conversation_title(session, cid, "My Title", user)

    assert conv.title == "My Title"


async def test_set_title_rejects_other_tenant(
    session: AsyncSession, user: UserSession
) -> None:
    cid = uuid.uuid4()
    await get_or_create_conversation(session, cid, user)
    with pytest.raises(ConversationNotFound):
        await set_conversation_title(session, cid, "hacked", _other_tenant())


async def test_delete_then_messages_raises(
    session: AsyncSession, user: UserSession
) -> None:
    cid = uuid.uuid4()
    await get_or_create_conversation(session, cid, user)
    await save_message(cid, "hi", MessageRole.user, session)

    await delete_conversation(cid, session, user)

    with pytest.raises(ConversationNotFound):
        await get_conversation_messages(session, cid, user)


async def test_single_query_no_hits_returns_fallback(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    # force zero retrieval hits so the fallback branch runs regardless of DB contents
    async def _no_hits(*_args: object, **_kwargs: object) -> list[object]:
        return []

    monkeypatch.setattr("app.retrieval.services.search_query_to_chunks", _no_hits)
    resp = await single_query_to_answer(session, "anything at all", FakeLLM(), k=3)
    assert isinstance(resp, QuestionResponse)
    assert resp.answer == "No relevant context found."
    assert resp.citations == []


async def test_single_query_with_embedded_chunk_returns_answer(
    session: AsyncSession, user: UserSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    # relax the relevance floor so any embedded chunk counts as a hit
    monkeypatch.setattr("app.core.config.settings.retrieval_min_score", 0.0)
    text = "Acme Corp offers a 30 day money back guarantee on all purchases."
    vector = get_embedder().embed([text])[0]
    session.add(
        Document(
            name="acme-policy",
            source_type=SourceType.upload,
            source_ref="test",
            content_type="text/plain",
            byte_size=len(text.encode("utf-8")),
            chunks=[Chunk(position=0, text=text, embedding=vector)],
        )
    )
    await session.flush()

    resp = await single_query_to_answer(
        session, "What is the return policy?", FakeLLM(), k=3
    )

    assert isinstance(resp, QuestionResponse)
    assert resp.answer == "fake answer"
