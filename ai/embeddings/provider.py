"""Embedding providers behind one interface.

Default is `fastembed`: ONNX models that run locally, need no API key and pull
in no torch dependency. That matters for a project whose users may be on
metered connections and whose budget may be zero. OpenAI is available for
teams that would rather not ship model weights.

Documents and queries go through separate methods because several strong
retrieval models (bge, e5) expect an instruction prefix on the query only.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache

from apps.api.config import settings

log = logging.getLogger(__name__)

# Models that want a retrieval instruction prefixed to the query.
_QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class EmbeddingProvider(ABC):
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedProvider(EmbeddingProvider):
    def __init__(self, model_name: str, dim: int) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "fastembed is not installed. Run `pip install fastembed`, or set "
                "EMBEDDING_PROVIDER=openai."
            ) from exc

        self.model_name = model_name
        self.dim = dim
        log.info("Loading local embedding model %s", model_name)
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        prefix = _QUERY_PREFIXES.get(self.model_name, "")
        return next(iter(self._model.embed([prefix + text]))).tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str, dim: int, api_key: str | None) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai.")
        from openai import OpenAI

        self.model_name = model_name
        self.dim = dim
        self._client = OpenAI(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        size = settings.embedding_batch_size
        for i in range(0, len(texts), size):
            batch = texts[i : i + size]
            resp = self._client.embeddings.create(
                model=self.model_name, input=batch, dimensions=self.dim
            )
            out.extend(item.embedding for item in resp.data)
        return out

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(
            model=self.model_name, input=[text], dimensions=self.dim
        )
        return resp.data[0].embedding


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Cached so the local model is loaded into memory exactly once."""
    provider = settings.embedding_provider
    if provider == "fastembed":
        return FastEmbedProvider(settings.embedding_model, settings.embedding_dim)
    if provider == "openai":
        return OpenAIEmbeddingProvider(
            settings.embedding_model, settings.embedding_dim, settings.openai_api_key
        )
    raise ValueError(f"Unknown embedding provider: {provider}")
