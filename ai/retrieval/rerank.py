"""Reranking: reorder fused candidates by actual relevance to the question.
 
Retrieval optimises for recall -- pull in 20 plausible chunks. Reranking
optimises for precision -- decide which 6 of those actually answer the
question. Doing this in one step is possible but worse: a single retriever
tuned for precision misses too much, and stuffing all 20 into the prompt
buries the answer.
 
Two implementations:
 
* CrossEncoderReranker -- a real query/passage cross-encoder. Used when
  fastembed exposes one. Best quality.
* LexicalOverlapReranker -- a cheap fallback that scores term coverage and
  proximity. Not as good, but it keeps the pipeline honest with no extra
  dependency, and it beats trusting the fused rank alone.
 
The interface is the point. Swapping in Cohere Rerank or a fine-tuned model
later touches this file only.
"""
 
from __future__ import annotations
 
import logging
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache
 
from ai.retrieval.hybrid import RetrievedChunk
from apps.api.config import settings
 
log = logging.getLogger(__name__)
 
_TOKEN = re.compile(r"[a-z0-9]+")
 
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "but", "by", "can",
    "do", "does", "explain", "for", "from", "how", "i", "in", "is", "it", "its",
    "me", "my", "of", "on", "or", "please", "that", "the", "their", "them",
    "there", "these", "this", "to", "was", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "your", "difference", "tell",
    # Words describing the *shape* of the wanted answer, not its subject.
    # Counting them as content made passages discussing formulas outrank the
    # passages containing them.
    "formula", "formulae", "formulas", "equation", "equations", "expression",
    "derive", "derivation", "reaction", "balanced", "state", "write", "give",
    "calculate", "compute", "value", "relation", "relationship",
}
 
 
def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())
 
 
def _content_terms(text: str) -> list[str]:
    return [t for t in _tokens(text) if t not in _STOPWORDS and len(t) > 2]
 
 
class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
        wants_notation: bool = False,
    ) -> list[RetrievedChunk]: ...
 
 
class NoOpReranker(Reranker):
    """Keep the fused order. Useful as an evaluation baseline."""
 
    def rerank(self, query, chunks, top_n, wants_notation=False):
        for chunk in chunks:
            chunk.rerank_score = chunk.fused_score
        return chunks[:top_n]
 
 
class LexicalOverlapReranker(Reranker):
    """Score by how many distinct query terms a chunk covers, and how tightly.
 
    Coverage dominates: a passage mentioning both "mitosis" and "meiosis" is
    far more likely to answer a question comparing them than one that repeats
    "mitosis" ten times. Proximity breaks ties, and the fused rank contributes
    a small prior so a strong semantic match is not thrown away just because
    it phrases things differently.
    """
 
    def rerank(self, query, chunks, top_n, wants_notation=False):
        query_terms = set(_content_terms(query))
        if not query_terms:
            return NoOpReranker().rerank(query, chunks, top_n, wants_notation)
 
        for chunk in chunks:
            chunk_tokens = _tokens(chunk.content)
            if not chunk_tokens:
                chunk.rerank_score = 0.0
                continue
 
            positions: dict[str, list[int]] = {}
            for i, token in enumerate(chunk_tokens):
                if token in query_terms:
                    positions.setdefault(token, []).append(i)
 
            coverage = len(positions) / len(query_terms)
 
            # Saturating frequency: the 5th mention adds little over the 2nd.
            density = sum(
                math.log1p(len(p)) for p in positions.values()
            ) / (1 + math.log1p(len(chunk_tokens)))
 
            proximity = 0.0
            if len(positions) > 1:
                first = min(p[0] for p in positions.values())
                last = max(p[-1] for p in positions.values())
                span = max(last - first, 1)
                proximity = len(positions) / span
 
            # Section headings are strong signals of topicality.
            heading_bonus = 0.0
            if chunk.section:
                section_terms = set(_content_terms(chunk.section))
                if section_terms & query_terms:
                    heading_bonus = 0.15
 
            prior = chunk.fused_score * 2.0
 
            if wants_notation:
                # A formula is almost all symbols: "F = kq₁q₂/r²" contains
                # none of the words in "formula for coulomb force", so word
                # overlap scores it at zero and any paragraph discussing the
                # formula outranks the formula itself. Adding a bonus is not
                # enough to climb out of that hole, so when the student has
                # explicitly asked for notation it is weighted roughly on par
                # with topical overlap instead.
                #
                # Topicality still matters -- it is what keeps F = ma from
                # surfacing for an electrostatics question -- but it comes
                # mostly from the semantic search that put this chunk in the
                # candidate pool, which `prior` carries.
                chunk.rerank_score = (
                    0.30 * coverage
                    + 0.10 * min(density, 1.0)
                    + 0.05 * min(proximity, 1.0)
                    + 0.55 * chunk.notation_score
                    + heading_bonus
                    + prior
                )
            else:
                chunk.rerank_score = (
                    0.55 * coverage
                    + 0.20 * min(density, 1.0)
                    + 0.10 * min(proximity, 1.0)
                    + heading_bonus
                    + prior
                )
 
        return sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)[:top_n]
 
 
class CrossEncoderReranker(Reranker):
    """A real relevance model, when one is available locally."""
 
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
 
        self._model = TextCrossEncoder(model_name=model_name)
        self._fallback = LexicalOverlapReranker()
 
    def rerank(self, query, chunks, top_n, wants_notation=False):
        if not chunks:
            return []
        try:
            scores = list(self._model.rerank(query, [c.content for c in chunks]))
        except Exception:
            log.exception("Cross-encoder failed; falling back to lexical reranking")
            return self._fallback.rerank(query, chunks, top_n, wants_notation)
 
        for chunk, score in zip(chunks, scores, strict=False):
            # Cross-encoders emit unbounded logits, often negative for a
            # perfectly good passage. Left raw, they are not comparable with
            # the lexical reranker's 0-1 range, and any score floor above
            # zero silently discards the entire candidate list -- which is
            # exactly what happened: 20 candidates in, 0 kept, and an
            # abstention the student could not explain.
            chunk.rerank_score = 1.0 / (1.0 + math.exp(-float(score)))
            # Cross-encoders are trained on prose pairs and share the same
            # blind spot for bare notation.
            if wants_notation and chunk.notation_score > 0:
                chunk.rerank_score += 0.6 * chunk.notation_score
        return sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)[:top_n]
 
 
@lru_cache
def get_reranker() -> Reranker:
    if not settings.enable_reranking:
        return NoOpReranker()
    try:
        reranker = CrossEncoderReranker()
        log.info("Using cross-encoder reranker")
        return reranker
    except Exception:
        log.info("Cross-encoder unavailable; using lexical overlap reranker")
        return LexicalOverlapReranker()