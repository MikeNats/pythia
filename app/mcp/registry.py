"""Import-time MCP tool registry + @mcp_tool decorator.

A tool = a name + description + a Pydantic input model (whose JSON Schema we
hand to `tools/list`) + an async handler (session, validated args, user).
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import UserSession

# args is the validated input model instance; typed Any so concrete handlers
# (each taking its own input model) register without variance complaints.
ToolHandler = Callable[[AsyncSession, Any, UserSession], Awaitable[Any]]


@dataclass
class McpTool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler


_REGISTRY: dict[str, McpTool] = {}


def mcp_tool(
    name: str, description: str, input_model: type[BaseModel]
) -> Callable[[ToolHandler], ToolHandler]:
    def decorator(handler: ToolHandler) -> ToolHandler:
        _REGISTRY[name] = McpTool(name, description, input_model, handler)
        return handler

    return decorator


def tool_listing() -> list[dict[str, Any]]:
    """The `tools/list` payload — each tool's schema derived from its Pydantic model."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "inputSchema": t.input_model.model_json_schema(),
        }
        for t in _REGISTRY.values()
    ]


def get_tool(name: str) -> McpTool | None:
    return _REGISTRY.get(name)
