"""Global settings — env-driven via pydantic-settings (no secrets in code)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "postgresql+asyncpg://docqapp:docqapp@localhost:5434/docqapp"
    db_echo: bool = False

    anthropic_api_key: str | None = None
    default_claude_model: str = "claude-sonnet-4-6"

    openai_api_key: str | None = None
    default_openai_model: str = "gpt-4o-mini"

    ollama_host: str | None = None
    llama_default_model: str = "llama3.2"

    llm_timeout: float = 60.0
    llm_max_retries: int = 3
    # Context packing: char budget for prior chat turns fed back to the LLM.
    # Rough — ~4 chars/token, so 8000 ≈ 2000 tokens. Keeps the newest turns.
    chat_history_budget_chars: int = 8000
    # 0.0 = deterministic (greedy). Default for all providers unless a call
    # overrides it — evals and structured extraction want reproducible output.
    llm_temperature: float = 0.0
    # Relevance floor for vector search: drop hits below this cosine similarity
    # (0.0 = off). Raises precision (cuts noise) but too high → zero hits → the
    # answer becomes "no relevant context". Tune it on your evals.
    retrieval_min_score: float = 0.0
    logging_level: str = "INFO"
    logging: bool = True

    # capture request bodies in the audit trail (skips file uploads, capped).
    # NOTE: bodies/prompts can hold PHI — turn off or lower the cap in real prod.
    audit_request_bodies: bool = True
    audit_max_body_bytes: int = 8192
    audit_enabled: bool = True  # set False in tests → no audit rows written to DB
    # input guardrail: also run an LLM classifier (catches paraphrased injections
    # the regex misses). Off by default — it adds an LLM call per request.
    guardrail_llm_check: bool = False


settings = Settings()
