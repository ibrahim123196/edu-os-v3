"""LLM providers behind one narrow interface.

Deliberately small: one `complete` method taking a system prompt and messages.
Everything provider-specific stays inside the adapter, so the retrieval and
agent layers never import a vendor SDK. Changing model provider is an env
change.

The `echo` provider exists so the whole pipeline can be tested -- ingestion,
retrieval, fusion, reranking, citation parsing -- without an API key or a
network call.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache

from apps.api.config import settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class LLMClient(ABC):
    model: str

    # Whether this client can usefully rewrite a search query. The offline
    # stand-in cannot: it has no idea what was asked, so its "rewrite" would
    # become the text that gets searched for.
    supports_query_rewriting: bool = True

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str: ...


class AnthropicClient(LLMClient):
    def __init__(self, model: str, api_key: str | None) -> None:
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system, messages, max_tokens=None, temperature=None) -> str:
        try:
            resp = self._client.messages.create(
                model=self.model,
                system=system,
                messages=messages,
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=(
                    temperature if temperature is not None else settings.llm_temperature
                ),
            )
        except Exception as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc

        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()


class OpenAIClient(LLMClient):
    def __init__(self, model: str, api_key: str | None) -> None:
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set.")
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key)

    def complete(self, system, messages, max_tokens=None, temperature=None) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, *messages],
                max_tokens=max_tokens or settings.llm_max_tokens,
                temperature=(
                    temperature if temperature is not None else settings.llm_temperature
                ),
            )
        except Exception as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()


class EchoClient(LLMClient):
    """Offline stand-in. Produces a citation-shaped answer from the sources."""

    model = "echo"
    supports_query_rewriting = False

    def complete(self, system, messages, max_tokens=None, temperature=None) -> str:
        content = messages[-1]["content"]
        if "[S1]" not in content:
            from ai.prompts.grounded_answer import INSUFFICIENT_EVIDENCE

            return INSUFFICIENT_EVIDENCE

        question = content.split("Student's question:")[-1].split("\n")[0].strip()
        first = content.split("[S1]", 1)[1].split("\n", 1)[-1]
        # The whole top passage, cut only where the next source begins.
        # A short excerpt made it look as though answers were being
        # truncated by the system, when nothing here writes an answer at
        # all -- this provider exists to prove retrieval works without an
        # API key, and seeing the full retrieved passage is the point.
        excerpt = first.split("[S2]")[0].strip()
        return (
            f"(echo provider — no model configured, so this is the top "
            f"retrieved passage verbatim rather than an answer)\n\n"
            f"Question: {question}\n\n{excerpt} [S1]"
        )


@lru_cache
def get_llm() -> LLMClient:
    provider = settings.llm_provider
    if provider == "anthropic":
        return AnthropicClient(settings.llm_model, settings.anthropic_api_key)
    if provider == "openai":
        return OpenAIClient(settings.llm_model, settings.openai_api_key)
    if provider == "echo":
        return EchoClient()
    raise LLMError(f"Unknown LLM provider: {provider}")
