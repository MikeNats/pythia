"""Conversation/message service tests — persistence, ordering, tenant isolation.

The two `*_rejects_other_tenant` tests are the important ones: they prove the
IDOR fix (a caller can only touch their own tenant's conversations).
"""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import UserSession
from app.retrieval.exceptions import ConversationNotFound
from app.retrieval.models import MessageRole
from app.retrieval.services import (
    delete_conversation,
    get_conversation_messages,
    get_or_create_conversation,
    save_message,
    set_conversation_title,
)


def _attacker() -> UserSession:
    """A user from a different tenant."""
    return UserSession(user_id=uuid.uuid4(), tenant_id=uuid.uuid4())


async def test_creates_conversation_when_missing(
    session: AsyncSession, user: UserSession
) -> None:
    cid = uuid.uuid4()
    conv = await get_or_create_conversation(session, cid, user)
    assert conv.id == cid
    assert conv.tenant_id == user.tenant_id
    assert conv.user_id == user.user_id


async def test_get_or_create_is_idempotent(
    session: AsyncSession, user: UserSession
) -> None:
    cid = uuid.uuid4()
    a = await get_or_create_conversation(session, cid, user)
    b = await get_or_create_conversation(session, cid, user)
    assert a.id == b.id


async def test_messages_saved_in_seq_order(
    session: AsyncSession, user: UserSession
) -> None:
    cid = uuid.uuid4()
    await get_or_create_conversation(session, cid, user)
    await save_message(cid, "question", MessageRole.user, session)
    await save_message(cid, "answer", MessageRole.llm, session)
    msgs = await get_conversation_messages(session, cid, user)
    assert [m.text for m in msgs] == ["question", "answer"]
    assert msgs[0].seq < msgs[1].seq


async def test_delete_rejects_other_tenant(
    session: AsyncSession, user: UserSession
) -> None:
    cid = uuid.uuid4()
    await get_or_create_conversation(session, cid, user)
    with pytest.raises(ConversationNotFound):
        await delete_conversation(cid, session, _attacker())


async def test_soft_delete_hides_conversation(
    session: AsyncSession, user: UserSession
) -> None:
    cid = uuid.uuid4()
    await get_or_create_conversation(session, cid, user)
    await save_message(cid, "hi", MessageRole.user, session)
    await delete_conversation(cid, session, user)
    with pytest.raises(ConversationNotFound):
        await get_conversation_messages(session, cid, user)


async def test_set_title_rejects_other_tenant(
    session: AsyncSession, user: UserSession
) -> None:
    cid = uuid.uuid4()
    await get_or_create_conversation(session, cid, user)
    with pytest.raises(ConversationNotFound):
        await set_conversation_title(session, cid, "hacked", _attacker())
