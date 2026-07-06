"""LLMRouter / resolve_client tests — provider + model selection.

resolve_client only *instantiates* an adapter; it never calls a model, so no
network mocking is needed. We set dummy API keys on `settings` so the OpenAI
and Anthropic SDK constructors don't reject missing credentials — this keeps
the test hermetic regardless of what's in `.env`.
"""

import pytest

from app.core.config import settings
from app.llm.anthropic_adapter import AnthropicAdapter
from app.llm.ollama_adapter import OllamaAdapter
from app.llm.openai_adapter import OpenAIAdapter
from app.llm.router import (
    LLMModelNotFoundError,
    LLMNotFoundError,
    LLMProviders,
    resolve_client,
)


@pytest.fixture(autouse=True)
def _dummy_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the SDK constructors non-empty keys — no network call is made."""
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "openai_api_key", "sk-openai-test")


def test_anthropic_returns_adapter() -> None:
    client = resolve_client(LLMProviders.ANTHROPIC, "sonnet")
    assert isinstance(client, AnthropicAdapter)


def test_anthropic_default_model_returns_adapter() -> None:
    client = resolve_client(LLMProviders.ANTHROPIC, None)
    assert isinstance(client, AnthropicAdapter)


def test_anthropic_bad_model_raises() -> None:
    with pytest.raises(LLMModelNotFoundError):
        resolve_client(LLMProviders.ANTHROPIC, "not-a-real-model")


def test_ollama_returns_adapter() -> None:
    client = resolve_client(LLMProviders.OLLAMA, "llama3.2")
    assert isinstance(client, OllamaAdapter)


def test_ollama_default_model_returns_adapter() -> None:
    client = resolve_client(LLMProviders.OLLAMA, None)
    assert isinstance(client, OllamaAdapter)


def test_ollama_bad_model_raises() -> None:
    with pytest.raises(LLMModelNotFoundError):
        resolve_client(LLMProviders.OLLAMA, "not-a-real-model")


def test_openai_returns_adapter() -> None:
    client = resolve_client(LLMProviders.OPENAI, "gpt-4o")
    assert isinstance(client, OpenAIAdapter)


def test_openai_default_model_returns_adapter() -> None:
    client = resolve_client(LLMProviders.OPENAI, None)
    assert isinstance(client, OpenAIAdapter)


def test_openai_bad_model_raises() -> None:
    with pytest.raises(LLMModelNotFoundError):
        resolve_client(LLMProviders.OPENAI, "not-a-real-model")


def test_no_provider_raises() -> None:
    with pytest.raises(LLMNotFoundError):
        resolve_client(None, None)


def test_no_provider_with_model_raises() -> None:
    with pytest.raises(LLMNotFoundError):
        resolve_client(None, "sonnet")
