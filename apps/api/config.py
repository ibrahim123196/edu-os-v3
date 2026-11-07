"""Application configuration.

All tunables live here so that swapping an LLM or embedding provider is an
environment change rather than a code change (README: "The agent orchestration
layer should remain modular enough that the underlying model provider can be
changed").
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Core -------------------------------------------------------------
    app_name: str = "EduOS"
    environment: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"

    # --- Database ---------------------------------------------------------
    database_url: str = "postgresql+psycopg://eduos:eduos@localhost:5432/eduos"

    # --- Ingestion --------------------------------------------------------
    upload_dir: str = "./storage/uploads"
    max_upload_mb: int = 25
    chunk_target_words: int = 220
    chunk_overlap_words: int = 45
    chunk_min_words: int = 25

    # --- Embeddings -------------------------------------------------------
    # "fastembed" runs locally with no API key and no torch dependency.
    embedding_provider: Literal["fastembed", "openai"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 32

    # --- Retrieval --------------------------------------------------------
    vector_top_k: int = 20
    keyword_top_k: int = 20
    rrf_k: int = 60  # smoothing constant for reciprocal rank fusion
    rerank_top_n: int = 6  # chunks handed to the LLM
    # Size of the formula-only search arm, used when a question asks for an
    # equation or reaction.
    notation_top_k: int = 10
    min_context_score: float = 0.0  # abstain if nothing clears this

    # How well the student's own notes must match before the built-in
    # chapters are left out of the answer. Cosine similarity, so absolute
    # and comparable across questions. Raise it to lean on the textbook more
    # often, lower it to stay inside the student's own material.
    own_material_threshold: float = 0.72

    # --- Generation -------------------------------------------------------
    llm_provider: Literal["anthropic", "openai", "echo"] = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 1200
    llm_temperature: float = 0.2

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # --- Behaviour --------------------------------------------------------
    enable_query_rewriting: bool = True
    enable_reranking: bool = True

    # --- Spelling ---------------------------------------------------------
    # Typos are corrected against words found in the student's own documents.
    # 0.4 catches "coulumb" -> "coulomb" (similarity ~0.45) while leaving
    # genuinely different words alone. Raise it if unrelated words get
    # swapped; lower it to catch sloppier typing.
    enable_spelling_correction: bool = True
    spelling_min_similarity: float = 0.4


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
