from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, cast

import anthropic
from anthropic import omit
from anthropic.types import (
    MessageParam,
    ToolChoiceToolParam,
    ToolParam,
    ToolResultBlockParam,
)

from app.core.config import settings
from app.llm.llm_client import InferenceResponse, LLMClient
from app.llm.registry import ToolDefinition

DEFAULT_CLAUDE_MODEL = settings.default_claude_model
CLAUDE_MODEL_MAP: dict[str, str] = {
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
    "opus": "claude-opus-4-8",
}


class AnthropicAdapter(LLMClient):
    def __init__(
        self, api_key: str | None = None, model: str = DEFAULT_CLAUDE_MODEL
    ) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or settings.anthropic_api_key,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
        self._model = model

    async def stream_answer(
        self, system_prompt: str, user_prompt: str, *, model: str | None = None
    ) -> AsyncIterator[str]:
        requested = model or self._model
        resolved = CLAUDE_MODEL_MAP.get(requested, requested)
        async with self._client.messages.stream(
            model=resolved,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

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
        requested = model or self._model
        resolved = CLAUDE_MODEL_MAP.get(requested, requested)
        tools = cast(list[ToolParam], [{"name": name, "input_schema": tool_schema}])
        tool_choice: ToolChoiceToolParam = {"type": "tool", "name": name}
        response = await self._client.messages.create(
            model=resolved,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice=tool_choice,
            temperature=(
                temperature if temperature is not None else settings.llm_temperature
            ),
        )

        content: dict[str, object] = {}
        for block in response.content:
            if block.type == "tool_use":
                content = block.input
                break

        metadata: dict[str, object] = {
            "model": response.model,
            "session_id": session_id,
        }
        if audit is not None:
            await audit(
                "llm inference",
                {
                    "model": response.model,
                    "tool": name,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": content,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                },
            )
        return InferenceResponse(content=content, metadata=metadata)

    def _to_anthropic_tool_schema(self, tools: list[ToolDefinition]) -> list[ToolParam]:
        return cast(
            list[ToolParam],
            [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in tools
            ],
        )

    async def agent_loop(
        self,
        prompt: str,
        tools: list[ToolDefinition],
        run_tool: Callable[[str, Mapping[str, Any]], Awaitable[str]],
        *,
        audit: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        max_iterations: int = 5,
    ) -> str:
        requested = model or self._model
        resolved = CLAUDE_MODEL_MAP.get(requested, requested)
        advertised = self._to_anthropic_tool_schema(tools)
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]

        for _ in range(max_iterations):
            response = await self._client.messages.create(
                model=resolved,
                max_tokens=4096,
                system=system_prompt if system_prompt is not None else omit,
                messages=messages,
                tools=advertised,
            )
            if audit is not None:
                await audit(
                    "llm call",
                    {
                        "model": resolved,
                        "system_prompt": system_prompt,
                        "user_prompt": prompt,
                        "stop_reason": response.stop_reason,
                        "response_text": "".join(
                            b.text for b in response.content if b.type == "text"
                        ),
                        "tool_calls": [
                            {"name": b.name, "input": b.input}
                            for b in response.content
                            if b.type == "tool_use"
                        ],
                        "usage": {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                        },
                    },
                )
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                return "".join(b.text for b in response.content if b.type == "text")

            messages.append(
                cast(MessageParam, {"role": "assistant", "content": response.content})
            )
            results: list[ToolResultBlockParam] = []
            for block in tool_uses:
                args = cast(Mapping[str, Any], block.input)
                output = await run_tool(block.name, args)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )
            messages.append({"role": "user", "content": results})

        raise RuntimeError(f"agent loop did not finish in {max_iterations} iterations")
