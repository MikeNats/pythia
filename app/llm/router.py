from enum import Enum

from app.llm.anthropic_adapter import (
    CLAUDE_MODEL_MAP,
    DEFAULT_CLAUDE_MODEL,
    AnthropicAdapter,
)
from app.llm.ollama_adapter import LLAMA_DEFAULT_MODEL, OLLAMA_MODELS, OllamaAdapter
from app.llm.openai_adapter import DEFAULT_OPENAI_MODEL, OPENAI_MODELS, OpenAIAdapter


class LLMProviders(Enum):
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENAI = "openai"


class LLMNotFoundError(Exception):
    pass


class LLMModelNotFoundError(Exception):
    pass


Client = AnthropicAdapter | OllamaAdapter | OpenAIAdapter


def resolve_client(provider: LLMProviders | None, model: str | None) -> Client:
    return LLMRouter(provider=provider, model=model).get_client()


class LLMRouter:
    def __init__(
        self, provider: LLMProviders | None = None, model: str | None = None
    ) -> None:
        self._provider = provider
        self._model = model

    def get_client(self) -> Client:
        if self._provider == LLMProviders.ANTHROPIC:
            if self._model and self._model not in CLAUDE_MODEL_MAP:
                raise LLMModelNotFoundError(
                    f"Model '{self._model}' not found for Anthropic provider."
                )
            return AnthropicAdapter(model=self._model or DEFAULT_CLAUDE_MODEL)

        if self._provider == LLMProviders.OLLAMA:
            if self._model and self._model not in OLLAMA_MODELS:
                raise LLMModelNotFoundError(
                    f"Model '{self._model}' not found for Ollama provider."
                )
            return OllamaAdapter(model=self._model or LLAMA_DEFAULT_MODEL)

        if self._provider == LLMProviders.OPENAI:
            if self._model and self._model not in OPENAI_MODELS:
                raise LLMModelNotFoundError(
                    f"Model '{self._model}' not found for OpenAI provider."
                )
            return OpenAIAdapter(model=self._model or DEFAULT_OPENAI_MODEL)

        raise LLMNotFoundError("No valid LLM provider specified.")
