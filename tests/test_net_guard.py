"""SSRF guard tests — validate_url rejects unsafe URLs before we fetch them.

The DNS resolver (anyio.getaddrinfo) is monkeypatched so no real network or
name resolution happens; we feed it the IP we want validate_url to see.
"""

import socket
from collections.abc import Callable
from typing import Any

import pytest

from app.ingest.net_guard import UnsafeURL, validate_url


def _resolver_returning(ip: str) -> Callable[..., Any]:
    """Build a fake anyio.getaddrinfo that always resolves to `ip`."""

    async def _fake(*_args: object, **_kwargs: object) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    return _fake


async def test_non_http_scheme_rejected() -> None:
    with pytest.raises(UnsafeURL):
        await validate_url("ftp://example.com/file")


async def test_missing_host_rejected() -> None:
    with pytest.raises(UnsafeURL):
        await validate_url("http:///just-a-path")


async def test_loopback_ip_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.ingest.net_guard.anyio.getaddrinfo", _resolver_returning("127.0.0.1")
    )
    with pytest.raises(UnsafeURL):
        await validate_url("http://sneaky.example.com")


async def test_private_ip_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.ingest.net_guard.anyio.getaddrinfo", _resolver_returning("10.0.0.1")
    )
    with pytest.raises(UnsafeURL):
        await validate_url("http://internal.example.com")


async def test_public_ip_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.ingest.net_guard.anyio.getaddrinfo", _resolver_returning("8.8.8.8")
    )
    # Should not raise.
    await validate_url("http://public.example.com")
