"""The Milestone 1 pipeline, end to end.

    question
       -> query preparation (filler stripping, optional LLM rewrite)
       -> curriculum routing / metadata filters
       -> hybrid retrieval (pgvector + full-text)
       -> reciprocal rank fusion
       -> reranking
       -> context construction
       -> LLM
       -> citation validation

Each stage is a plain function call, and the trace records what every stage
did. When an answer is wrong, the trace says whether retrieval missed the
passage or the model ignored it -- that distinction is most of debugging a RAG
system, and it is very hard to recover after the fact.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace

from sqlalchemy.orm import Session

from ai.embeddings.provider import get_embedding_provider
from ai.generation.answer import GroundedAnswer, build_answer
from ai.generation.llm import LLMError, get_llm
from ai.prompts.grounded_answer import (
    INSUFFICIENT_EVIDENCE,
    SYSTEM_PROMPT,
    build_messages,
)
from ai.retrieval.hybrid import (
    RetrievalFilters,
    hybrid_search,
    reciprocal_rank_fusion,
)
from ai.retrieval.intent import detect as detect_intent
from ai.retrieval.query import prepare_query
from ai.retrieval.rerank import get_reranker
from ai.tutor.conversation import Turn, standalone_question
from apps.api.config import settings
from apps.api.curriculum import SourceKind

log = logging.getLogger(__name__)


@dataclass
class AskResult:
    answer: GroundedAnswer
    trace: dict
    # Typo fixes applied to the question, so the interface can tell the
    # student what was actually searched for.
    corrections: list = field(default_factory=list)
    corrected_question: str | None = None
    # A follow-up as it was searched ("why is that?" -> "why is the field
    # zero inside a conductor?"), so the student can see what "that" was
    # taken to mean -- and say so when it was taken wrongly.
    searched_as: str | None = None


def resolve_follow_up(question: str, history: list[Turn]) -> str | None:
    """"Quiz me on this" as the full request it stands for, or None.

    For the learning modes, which need the topic before deciding whether to
    search at all. Same rewrite as a follow-up question gets in ask().
    """
    try:
        llm = get_llm()
    except LLMError:
        return None
    return standalone_question(llm, history, question)


def ask(
    db: Session,
    question: str,
    filters: RetrievalFilters | None = None,
    history: list[Turn] | None = None,
) -> AskResult:
    started = time.perf_counter()
    trace: dict = {"stages": {}}
    filters = filters or RetrievalFilters()

    llm = None
    try:
        llm = get_llm()
    except LLMError as exc:
        # Retrieval still works without a model; the caller gets a clear error
        # instead of a fabricated answer.
        log.warning("LLM unavailable: %s", exc)
        trace["llm_error"] = str(exc)

    # --- 0. Follow-ups ---------------------------------------------------
    # Only when there are earlier turns: a first question costs no extra
    # call and runs exactly as it did before conversations existed.
    standalone = None
    if history:
        t0 = time.perf_counter()
        standalone = standalone_question(llm, history, question)
        trace["stages"]["follow_up"] = {
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "history_messages": len(history),
            "searched_as": standalone,
        }
    search_question = standalone or question

    # --- 1. Query preparation --------------------------------------------
    t0 = time.perf_counter()
    prepared = prepare_query(search_question, llm=llm, db=db)
    # What kind of answer is wanted -- a formula, or an explanation?
    intent = detect_intent(prepared.corrected or question)
    trace["stages"]["query_preparation"] = {
        "ms": round((time.perf_counter() - t0) * 1000, 1),
        "original": prepared.original,
        "lexical": prepared.lexical,
        "semantic": prepared.semantic,
        "rewritten": prepared.rewritten,
        "corrections": prepared.corrections,
        "wants_notation": intent.wants_notation,
        "intent_signals": list(intent.signals),
    }

    # --- 2. Retrieval -----------------------------------------------------
    #
    # The student's own notes come first. They are what their teacher set
    # and what the exam will follow -- a school's material may use different
    # notation or stress different cases than the textbook. The built-in
    # chapters exist for what those notes do not cover, so they are a
    # fallback rather than a competitor.
    #
    # Two stages, not one ranking with a thumb on the scale. Searching both
    # pools together and boosting the student's material still lets a
    # confident textbook passage beat a merely good one from their own
    # notes, which is the wrong answer to "explain what my teacher taught".
    t0 = time.perf_counter()
    embedding = get_embedding_provider().embed_query(prepared.semantic)

    own = hybrid_search(
        db,
        query=prepared.lexical,
        query_embedding=embedding,
        filters=replace(filters, source_kinds=[SourceKind.student_upload.value]),
        wants_notation=intent.wants_notation,
    )

    # Is the student's own material good enough on its own? Cosine
    # similarity is used rather than the fused or reranked score because it
    # is absolute: fusion and reranking are both relative to the candidate
    # set, so a pool of uniformly poor matches still produces a confident
    # looking top result.
    best_own = max((c.vector_score or 0.0 for c in own), default=0.0)
    sufficient = bool(own) and best_own >= settings.own_material_threshold

    candidates = own
    used_fallback = False
    if not sufficient:
        used_fallback = True
        platform = hybrid_search(
            db,
            query=prepared.lexical,
            query_embedding=embedding,
            filters=replace(filters, source_kinds=[SourceKind.platform.value]),
            wants_notation=intent.wants_notation,
        )
        candidates = reciprocal_rank_fusion([own, platform], k=settings.rrf_k)

    trace["stages"]["retrieval"] = {
        "ms": round((time.perf_counter() - t0) * 1000, 1),
        "candidates": len(candidates),
        "own_material": len(own),
        "best_own_similarity": round(best_own, 4),
        "own_material_sufficient": sufficient,
        "used_builtin_fallback": used_fallback,
        "vector_hits": sum(1 for c in candidates if c.vector_rank),
        "keyword_hits": sum(1 for c in candidates if c.keyword_rank),
        "both": sum(1 for c in candidates if c.vector_rank and c.keyword_rank),
        "notation_hits": sum(1 for c in candidates if c.notation_rank),
        "filters": {
            "document_ids": [str(d) for d in filters.document_ids],
            "subject": filters.subject,
            "grade": filters.grade,
        },
    }

    if not candidates:
        return AskResult(
            answer=GroundedAnswer(
                answer=INSUFFICIENT_EVIDENCE, grounded=False, retrieved_count=0
            ),
            trace=_finish(trace, started, abstained="no_candidates"),
            corrections=prepared.corrections,
            corrected_question=prepared.corrected,
            searched_as=standalone,
        )

    # --- 3. Reranking ------------------------------------------------------
    t0 = time.perf_counter()
    top = get_reranker().rerank(
        prepared.original,
        candidates,
        settings.rerank_top_n,
        wants_notation=intent.wants_notation,
    )
    top = [c for c in top if c.score >= settings.min_context_score]
    trace["stages"]["reranking"] = {
        "ms": round((time.perf_counter() - t0) * 1000, 1),
        "kept": len(top),
        "reranker": type(get_reranker()).__name__,
        "top_sources": [
            {
                "document": c.document_title,
                "page": c.page_label,
                "score": round(c.score, 4),
                "vector_rank": c.vector_rank,
                "keyword_rank": c.keyword_rank,
                "notation": round(c.notation_score, 3),
                "source": c.source_kind,
            }
            for c in top
        ],
    }

    if not top:
        return AskResult(
            answer=GroundedAnswer(
                answer=INSUFFICIENT_EVIDENCE,
                grounded=False,
                retrieved_count=len(candidates),
            ),
            trace=_finish(trace, started, abstained="below_threshold"),
            corrections=prepared.corrections,
            corrected_question=prepared.corrected,
            searched_as=standalone,
        )

    # --- 4. Generation -----------------------------------------------------
    if llm is None:
        raise LLMError(
            trace.get("llm_error", "No language model is configured.")
        )

    t0 = time.perf_counter()
    raw = llm.complete(
        system=SYSTEM_PROMPT,
        # The student's words, not the rewrite: "more simply" is an
        # instruction about the answer, and the rewrite travels alongside.
        messages=build_messages(
            question if history else prepared.original, top,
            history=history, standalone=standalone,
        ),
    )
    trace["stages"]["generation"] = {
        "ms": round((time.perf_counter() - t0) * 1000, 1),
        "model": llm.model,
        "sources_in_context": len(top),
    }

    # --- 5. Citation validation -------------------------------------------
    answer = build_answer(raw, top)
    trace["stages"]["citations"] = {
        "cited": answer.used_source_count,
        "offered": len(top),
        "grounded": answer.grounded,
    }

    return AskResult(
        answer=answer,
        trace=_finish(trace, started),
        corrections=prepared.corrections,
        corrected_question=prepared.corrected,
        searched_as=standalone,
    )


def _finish(trace: dict, started: float, abstained: str | None = None) -> dict:
    trace["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
    if abstained:
        trace["abstained"] = abstained
    return trace
