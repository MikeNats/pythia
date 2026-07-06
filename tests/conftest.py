"""Shared test fixtures.

- `session`: rollback-per-test DB session (savepoint; no pollution).
- `user`:    a seeded tenant+user → UserSession.
- `client`:  httpx client against the real app, with auth + DB session overridden.
- `FakeLLM`: a canned LLMClient so tests never call a real model.

Test-mode settings are set at import: audit writes off (no DB pollution),
logging off (quiet output).
"""

from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.auth.models import Tenant, User
from app.core.config import settings
from app.core.db import get_session
from app.core.dependencies import UserSession, get_current_user
from app.llm.llm_client import InferenceResponse
from app.llm.registry import ToolDefinition
from app.main import app

settings.audit_enabled = False  # tests never write audit rows to the DB
settings.logging = False


@pytest_asyncio.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    """DB session whose writes are rolled back after each test (savepoint)."""
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    connection = await engine.connect()
    transaction = await connection.begin()
    sess = AsyncSession(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    try:
        yield sess
    finally:
        await sess.close()
        await transaction.rollback()
        await connection.close()
        await engine.dispose()


@pytest_asyncio.fixture
async def user(session: AsyncSession) -> UserSession:
    """Seed a tenant + user, return the UserSession the services expect."""
    tenant = Tenant(name="Test Tenant")
    session.add(tenant)
    await session.flush()
    u = User(tenant_id=tenant.id, email="test@example.com")
    session.add(u)
    await session.flush()
    return UserSession(user_id=u.id, tenant_id=tenant.id)


@pytest_asyncio.fixture
async def client(
    session: AsyncSession, user: UserSession
) -> AsyncGenerator[AsyncClient, None]:
    """App client with auth + DB session overridden to the test session/user."""

    async def _session_override() -> AsyncGenerator[AsyncSession, None]:
        yield session

    def _user_override() -> UserSession:
        return user

    app.dependency_overrides[get_session] = _session_override
    app.dependency_overrides[get_current_user] = _user_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


class FakeLLM:
    """A canned LLMClient — tests inject this instead of calling a real model."""

    async def inference(
        self,
        name: str,
        system_prompt: str,
        user_prompt: str,
        tool_schema: dict[str, object],
        *,
        audit: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        session_id: str | None = None,
    ) -> InferenceResponse:
        return InferenceResponse(
            content={"answer": "fake answer", "cited_indices": []}, metadata={}
        )

    async def agent_loop(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        run_tool: Callable[[str, Mapping[str, Any]], Awaitable[str]],
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        max_iterations: int = 5,
        audit: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> str:
        return "fake answer"

    async def stream_answer(
        self, system_prompt: str, user_prompt: str, *, model: str | None = None
    ) -> AsyncGenerator[str, None]:
        for tok in ("fake", " answer"):
            yield tok


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
