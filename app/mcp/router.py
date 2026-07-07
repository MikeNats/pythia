"""MCP endpoint — hand-rolled JSON-RPC 2.0 over Streamable HTTP.

`POST /mcp` handles requests (initialize / tools/list / tools/call); `GET /mcp`
is the SSE keepalive the transport expects. Both sit behind pythia's Bearer
auth (`CurrentUser`), so every call is authenticated and tenant-scoped.
"""

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import CurrentUser, DbSession, UserSession
from app.mcp import tools  # noqa: F401 — importing registers the tools
from app.mcp.protocol import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    JsonRpcId,
    jsonrpc_error,
    jsonrpc_result,
    tool_success,
)
from app.mcp.registry import get_tool, tool_listing

router = APIRouter(tags=["mcp"])

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "pythia", "version": "0.1.0"}


@router.get("/mcp")
async def mcp_stream(user: CurrentUser) -> StreamingResponse:
    async def _keepalive() -> AsyncIterator[bytes]:
        yield b": keepalive\n\n"

    return StreamingResponse(_keepalive(), media_type="text/event-stream")


@router.post("/mcp")
async def mcp_endpoint(
    request: Request, session: DbSession, user: CurrentUser
) -> Response:
    try:
        raw: Any = await request.json()
    except Exception:
        return JSONResponse(jsonrpc_error(None, PARSE_ERROR, "Invalid JSON"))

    if not isinstance(raw, dict):
        return JSONResponse(
            jsonrpc_error(None, INVALID_REQUEST, "Expected a JSON object")
        )

    req_id: JsonRpcId = raw.get("id")
    method = raw.get("method")
    params: dict[str, Any] = raw.get("params") or {}

    if raw.get("jsonrpc") != "2.0":
        return JSONResponse(jsonrpc_error(req_id, INVALID_REQUEST, "Not JSON-RPC 2.0"))

    # notifications (e.g. notifications/initialized) carry no id → just acknowledge
    is_notification = (
        req_id is None
        and isinstance(method, str)
        and method.startswith("notifications/")
    )
    if is_notification:
        return Response(status_code=202)

    if method == "initialize":
        return JSONResponse(
            jsonrpc_result(
                req_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                },
            )
        )
    if method == "tools/list":
        return JSONResponse(jsonrpc_result(req_id, {"tools": tool_listing()}))
    if method == "tools/call":
        return await _handle_tools_call(req_id, params, session, user)

    return JSONResponse(
        jsonrpc_error(req_id, METHOD_NOT_FOUND, f"Unknown method: {method}")
    )


async def _handle_tools_call(
    req_id: JsonRpcId, params: dict[str, Any], session: AsyncSession, user: UserSession
) -> JSONResponse:
    name = params.get("name")
    tool = get_tool(name) if isinstance(name, str) else None
    if tool is None:
        return JSONResponse(
            jsonrpc_error(req_id, INVALID_PARAMS, f"Unknown tool: {name}")
        )
    try:
        args = tool.input_model.model_validate(params.get("arguments") or {})
    except ValidationError as exc:
        return JSONResponse(jsonrpc_error(req_id, INVALID_PARAMS, str(exc)))
    result = await tool.handler(session, args, user)
    return JSONResponse(jsonrpc_result(req_id, tool_success(result)))
