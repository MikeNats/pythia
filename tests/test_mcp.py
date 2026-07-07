"""Tests for the hand-rolled MCP JSON-RPC endpoint (POST /mcp)."""

from uuid import uuid4

import app.mcp.tools as tools_mod
from app.retrieval.schemas import SearchHit


async def test_initialize(client) -> None:  # type: ignore[no-untyped-def]
    resp = await client.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["serverInfo"]["name"] == "pythia"


async def test_tools_list_exposes_search_docs(client) -> None:  # type: ignore[no-untyped-def]
    resp = await client.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    )
    tools = resp.json()["result"]["tools"]
    search = next(t for t in tools if t["name"] == "search_docs")
    assert "query" in search["inputSchema"]["properties"]


async def test_unknown_method_errors(client) -> None:  # type: ignore[no-untyped-def]
    resp = await client.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 3, "method": "does_not_exist"}
    )
    assert resp.json()["error"]["code"] == -32601  # METHOD_NOT_FOUND


async def test_missing_jsonrpc_version_errors(client) -> None:  # type: ignore[no-untyped-def]
    resp = await client.post("/mcp", json={"id": 4, "method": "initialize"})
    assert resp.json()["error"]["code"] == -32600  # INVALID_REQUEST


async def test_notification_is_acknowledged(client) -> None:  # type: ignore[no-untyped-def]
    resp = await client.post(
        "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}
    )
    assert resp.status_code == 202


async def test_tools_call_search_docs(client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    async def fake_search(session, query, k=5):  # type: ignore[no-untyped-def]
        return [SearchHit(chunk_id=uuid4(), document_id=uuid4(), text="hi", score=0.9)]

    monkeypatch.setattr(tools_mod, "search_query_to_chunks", fake_search)
    resp = await client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "search_docs", "arguments": {"query": "test"}},
        },
    )
    result = resp.json()["result"]
    assert result["isError"] is False
    assert result["structuredContent"][0]["text"] == "hi"


async def test_tools_call_unknown_tool_errors(client) -> None:  # type: ignore[no-untyped-def]
    resp = await client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        },
    )
    assert resp.json()["error"]["code"] == -32602  # INVALID_PARAMS


async def test_tools_call_ingest_md(client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from app.ingest.models import Chunk
    from app.ingest.service import IngestService

    async def _fake_embed(self, text):  # type: ignore[no-untyped-def]
        return [Chunk(position=0, text=text, embedding=[0.0] * 384)]

    monkeypatch.setattr(IngestService, "_embed_chunks", _fake_embed)
    resp = await client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "ingest_md",
                "arguments": {"name": "notes.md", "content": "# Hi\nhello world"},
            },
        },
    )
    result = resp.json()["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["name"] == "notes.md"
    assert result["structuredContent"]["chunks"] == 1
