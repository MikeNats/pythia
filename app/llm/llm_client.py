from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Protocol

from pydantic import BaseModel

from app.llm.registry import ToolDefinition


class InferenceResponse(BaseModel):
    content: dict[str, object]
    metadata: dict[str, object]


class LLMClient(Protocol):
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
    ) -> InferenceResponse: ...

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
    ) -> str: ...

    def stream_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
    ) -> AsyncIterator[str]: ...
