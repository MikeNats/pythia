import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)

from app.core.config import settings
from app.llm.llm_client import InferenceResponse, LLMClient
from app.llm.registry import ToolDefinition

DEFAULT_OPENAI_MODEL = settings.default_openai_model
OPENAI_MODELS: set[str] = {"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"}


class OpenAIAdapter(LLMClient):
    def __init__(
        self, api_key: str | None = None, model: str = DEFAULT_OPENAI_MODEL
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key or settings.openai_api_key,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
        self._model = model

    async def stream_answer(
        self, system_prompt: str, user_prompt: str, *, model: str | None = None
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=model or self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

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
        tools = cast(
            list[ChatCompletionToolParam],
            [
                {
                    "type": "function",
                    "function": {"name": name, "parameters": tool_schema},
                }
            ],
        )
        tool_choice = cast(
            ChatCompletionToolChoiceOptionParam,
            {"type": "function", "function": {"name": name}},
        )
        response = await self._client.chat.completions.create(
            model=model or self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=tools,
            tool_choice=tool_choice,
            temperature=(
                temperature if temperature is not None else settings.llm_temperature
            ),
        )

        message = response.choices[0].message
        content: dict[str, object] = {}
        first = message.tool_calls[0] if message.tool_calls else None
        if first is not None and first.type == "function":
            content = json.loads(first.function.arguments)

        if audit is not None:
            await audit(
                "llm inference",
                {
                    "model": response.model,
                    "tool": name,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": content,
                    "usage": self._usage(response.usage),
                },
            )

        metadata: dict[str, object] = {
            "model": response.model,
            "session_id": session_id,
        }
        return InferenceResponse(content=content, metadata=metadata)

    def _to_openai_tool_schema(
        self, tools: list[ToolDefinition]
    ) -> list[ChatCompletionToolParam]:
        return cast(
            list[ChatCompletionToolParam],
            [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ],
        )

    def _usage(self, usage: object) -> dict[str, object]:
        if usage is None:
            return {}
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }

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
        advertised = self._to_openai_tool_schema(tools)
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for _ in range(max_iterations):
            response = await self._client.chat.completions.create(
                model=model or self._model,
                messages=messages,
                tools=advertised,
            )
            message = response.choices[0].message
            if audit is not None:
                await audit(
                    "llm call",
                    {
                        "model": response.model,
                        "system_prompt": system_prompt,
                        "user_prompt": prompt,
                        "finish_reason": response.choices[0].finish_reason,
                        "response_text": message.content or "",
                        "tool_calls": [
                            {"name": c.function.name, "arguments": c.function.arguments}
                            for c in (message.tool_calls or [])
                            if c.type == "function"
                        ],
                        "usage": self._usage(response.usage),
                    },
                )

            if not message.tool_calls:
                return message.content or ""

            messages.append(cast(ChatCompletionMessageParam, message.model_dump()))
            for call in message.tool_calls:
                if call.type != "function":
                    continue
                args = cast(Mapping[str, Any], json.loads(call.function.arguments))
                output = await run_tool(call.function.name, args)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": output}
                )

        raise RuntimeError(f"agent loop did not finish in {max_iterations} iterations")
