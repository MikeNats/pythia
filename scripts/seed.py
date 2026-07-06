"""Seed a demo tenant + user + API key for local dev.

Prints the raw API key ONCE — copy it. Only its hash is stored, so it cannot
be recovered later. Re-run any time to mint a fresh tenant/user/key.

    ./run seed
"""

import asyncio
import secrets

from app.core.auth.models import ApiKey, Tenant, User
from app.core.db import SessionLocal
from app.retrieval.models import Conversation, Message  # noqa: F401 — register mappers

_TOKEN_PREFIX = "docq_"  # noqa: S105 — a key prefix, not a secret


async def seed() -> None:
    raw_token = f"{_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"

    async with SessionLocal() as session:
        tenant = Tenant(name="Demo Tenant")
        session.add(tenant)
        await session.flush()  # assigns tenant.id

        user = User(
            tenant_id=tenant.id,
            email="demo@example.com",
            name="Demo",
            lastname="User",
        )
        session.add(user)
        await session.flush()  # assigns user.id

        session.add(
            ApiKey(
                user_id=user.id,
                tenant_id=tenant.id,
                key_hash=ApiKey.hash_key(raw_token),
            )
        )
        await session.commit()

        tenant_id, user_id = tenant.id, user.id

    print("\n✅ Seed complete.\n")
    print(f"  tenant_id : {tenant_id}")
    print(f"  user_id   : {user_id}")
    print(f"  API key   : {raw_token}")
    print("\n⚠️  Copy the API key now — it is only stored hashed.")
    print(f"  Use it as:  Authorization: Bearer {raw_token}\n")


if __name__ == "__main__":
    asyncio.run(seed())
