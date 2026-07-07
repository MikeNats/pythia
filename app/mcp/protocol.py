"""JSON-RPC 2.0 envelopes + MCP result builders — hand-rolled, no MCP SDK.

We own the protocol (same pattern as rts/agent-wiki) so MCP mounts into the
FastAPI app and reuses its Bearer auth, instead of a separate stdio process.
"""

import json
from typing import Any

# JSON-RPC 2.0 standard error codes — https://www.jsonrpc.org/specification
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

JsonRpcId = str | int | None


def jsonrpc_result(req_id: JsonRpcId, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def jsonrpc_error(req_id: JsonRpcId, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def tool_success(structured: Any) -> dict[str, Any]:
    """MCP tools/call result: human-readable text + machine-readable structured."""
    return {
        "content": [{"type": "text", "text": json.dumps(structured)}],
        "structuredContent": structured,
        "isError": False,
    }


def tool_error(message: str) -> dict[str, Any]:
    """MCP tools/call result flagged as an error (the model sees the message)."""
    return {"content": [{"type": "text", "text": message}], "isError": True}
