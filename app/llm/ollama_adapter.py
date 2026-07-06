import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

from ollama import AsyncClient, ChatResponse, Message, ResponseError

from app.core.config import settings
from app.core.decorators import retry
from app.llm.llm_client import InferenceResponse, LLMClient
from app.llm.registry import ToolDefinition

LLAMA_DEFAULT_MODEL = settings.llama_default_model
OLLAMA_MODELS: set[str] = {LLAMA_DEFAULT_MODEL, "qwen2.5:7b"}


class OllamaAdapter(LLMClient):
    def __init__(
        self, host: str | None = None, model: str = LLAMA_DEFAULT_MODEL
    ) -> None:
        self._client = AsyncClient(
            host=host or settings.ollama_host, timeout=settings.llm_timeout
        )
        self._model = model

    @retry(retry_on=(ResponseError,))
    async def _chat(self, **kwargs: Any) -> ChatResponse:
        response: ChatResponse = await self._client.chat(**kwargs)  # pyright: ignore
        return response

    async def stream_answer(
        self, system_prompt: str, user_prompt: str, *, model: str | None = None
    ) -> AsyncIterator[str]:
        stream: AsyncIterator[ChatResponse] = await self._client.chat(  # pyright: ignore[reportUnknownMemberType]
            model=model or self._model,
            messages=[
                self._text_message("system", system_prompt),
                self._text_message("user", user_prompt),
            ],
            stream=True,
        )
        async for part in stream:
            content = part.message.content
            if content:
                yield content

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
        temp = temperature if temperature is not None else settings.llm_temperature
        options = {"temperature": temp}
        response = await self._chat(
            model=model or self._model,
            messages=[
                self._text_message("system", system_prompt),
                self._text_message("user", user_prompt),
            ],
            format=tool_schema,
            options=options,
        )
        raw = response.message.content or "{}"
        content: dict[str, object] = json.loads(raw)
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
                        "prompt_eval_count": response.prompt_eval_count,
                        "eval_count": response.eval_count,
                    },
                },
            )

        metadata: dict[str, object] = {
            "model": response.model,
            "tool_name": name,
            "session_id": session_id,
        }
        return InferenceResponse(content=content, metadata=metadata)

    def _to_ollama_tool_schema(
        self, tools: list[ToolDefinition]
    ) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _text_message(self, role: str, content: str) -> dict[str, Any]:
        return {"role": role, "content": content}

    def _tool_result_message(self, tool_name: str, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_name": tool_name, "content": content}

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
        ollama_tool_scema: list[dict[str, Any]] = self._to_ollama_tool_schema(tools)
        messages: list[Message | dict[str, Any]] = []
        if system_prompt:
            messages.append(self._text_message("system", system_prompt))
        messages.append(self._text_message("user", prompt))

        for _ in range(max_iterations):
            response = await self._chat(
                model=model or self._model,
                messages=messages,
                tools=ollama_tool_scema,
            )
            if audit is not None:
                await audit(
                    "llm call",
                    {
                        "model": response.model,
                        "system_prompt": system_prompt,
                        "user_prompt": prompt,
                        "response_text": response.message.content or "",
                        "tool_calls": [
                            {"name": c.function.name, "arguments": c.function.arguments}
                            for c in (response.message.tool_calls or [])
                        ],
                        "usage": {
                            "prompt_eval_count": response.prompt_eval_count,
                            "eval_count": response.eval_count,
                        },
                    },
                )
            message = response.message
            messages.append(message)

            if not message.tool_calls:
                return message.content or ""

            for call in message.tool_calls:
                result = await run_tool(call.function.name, call.function.arguments)
                messages.append(self._tool_result_message(call.function.name, result))

        raise RuntimeError(f"agent loop did not finish in {max_iterations} iterations")
