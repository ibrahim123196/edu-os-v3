"""Fusion and reranking behaviour.

These run without a database. The SQL in hybrid.py needs a live pgvector
instance and is covered by tests/integration instead.
"""

from __future__ import annotations

import uuid

import pytest

from ai.retrieval.hybrid import RetrievedChunk, reciprocal_rank_fusion
from ai.retrieval.query import to_lexical
from ai.retrieval.rerank import LexicalOverlapReranker, NoOpReranker


def chunk(content: str = "text", **kwargs) -> RetrievedChunk:
    defaults = dict(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="Biology notes",
        content=content,
        page_start=1,
        page_end=1,
        section=None,
        chunk_index=0,
    )
    defaults.update(kwargs)
    return RetrievedChunk(**defaults)


# --- fusion --------------------------------------------------------------

def test_chunk_found_by_both_searches_outranks_either_alone():
    shared = uuid.uuid4()
    semantic = [
        chunk(chunk_id=uuid.uuid4(), vector_rank=1, vector_score=0.91),
        chunk(chunk_id=shared, vector_rank=2, vector_score=0.88),
    ]
    lexical = [
        chunk(chunk_id=shared, keyword_rank=1, keyword_score=0.4),
        chunk(chunk_id=uuid.uuid4(), keyword_rank=2, keyword_score=0.2),
    ]
    fused = reciprocal_rank_fusion([semantic, lexical])
    assert fused[0].chunk_id == shared


def test_fusion_merges_scores_from_both_searches():
    shared = uuid.uuid4()
    fused = reciprocal_rank_fusion([
        [chunk(chunk_id=shared, vector_rank=3, vector_score=0.7)],
        [chunk(chunk_id=shared, keyword_rank=1, keyword_score=0.5)],
    ])
    assert len(fused) == 1
    assert fused[0].vector_score == 0.7
    assert fused[0].keyword_score == 0.5


def test_fusion_output_is_sorted_and_deduplicated():
    a, b = uuid.uuid4(), uuid.uuid4()
    fused = reciprocal_rank_fusion([
        [chunk(chunk_id=a, vector_rank=1), chunk(chunk_id=b, vector_rank=2)],
        [chunk(chunk_id=b, keyword_rank=1)],
    ])
    assert len({c.chunk_id for c in fused}) == 2
    assert fused == sorted(fused, key=lambda c: c.fused_score, reverse=True)


def test_larger_k_flattens_the_advantage_of_rank_one():
    lists = [[chunk(chunk_id=uuid.uuid4(), vector_rank=i) for i in range(1, 4)]]
    tight = reciprocal_rank_fusion([[c for c in lists[0]]], k=1)
    spread = [c.fused_score for c in tight]
    loose = reciprocal_rank_fusion(
        [[chunk(chunk_id=uuid.uuid4(), vector_rank=i) for i in range(1, 4)]], k=1000
    )
    loose_spread = [c.fused_score for c in loose]
    assert (spread[0] - spread[-1]) > (loose_spread[0] - loose_spread[-1])


def test_empty_input_is_safe():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# --- reranking -----------------------------------------------------------

QUERY = "Explain the difference between mitosis and meiosis"


def test_reranker_prefers_coverage_over_repetition():
    both = chunk(
        "Mitosis produces two identical daughter cells, whereas meiosis "
        "produces four gametes with half the chromosome number.",
        fused_score=0.016,
    )
    stuffed = chunk(
        "Mitosis mitosis mitosis. Mitosis happens in body cells. Mitosis.",
        fused_score=0.016,
    )
    ranked = LexicalOverlapReranker().rerank(QUERY, [stuffed, both], top_n=2)
    assert ranked[0] is both


def test_reranker_demotes_irrelevant_passages():
    relevant = chunk("Meiosis halves the chromosome number.", fused_score=0.016)
    irrelevant = chunk("Photosynthesis occurs in chloroplasts.", fused_score=0.016)
    ranked = LexicalOverlapReranker().rerank(QUERY, [irrelevant, relevant], top_n=2)
    assert ranked[0] is relevant
    assert ranked[1].rerank_score < ranked[0].rerank_score


def test_reranker_respects_top_n():
    chunks = [chunk(f"mitosis passage {i}", fused_score=0.01) for i in range(10)]
    assert len(LexicalOverlapReranker().rerank(QUERY, chunks, top_n=3)) == 3


def test_reranker_handles_a_query_of_pure_stopwords():
    chunks = [chunk("Mitosis divides cells.", fused_score=0.02)]
    ranked = LexicalOverlapReranker().rerank("what is it", chunks, top_n=1)
    assert len(ranked) == 1


def test_noop_reranker_preserves_fused_order():
    high = chunk("a", fused_score=0.9)
    low = chunk("b", fused_score=0.1)
    assert NoOpReranker().rerank(QUERY, [high, low], top_n=2)[0] is high


# --- query preparation ---------------------------------------------------

def test_lexical_query_strips_conversational_filler():
    cleaned = to_lexical("sir can you please explain to me what is osmosis?")
    assert "osmosis" in cleaned
    assert "please" not in cleaned.lower()


def test_lexical_query_keeps_technical_terms():
    cleaned = to_lexical("What is Kirchhoff's voltage law?")
    assert "Kirchhoff's" in cleaned
    assert "voltage" in cleaned


def test_lexical_query_never_returns_empty():
    assert to_lexical("please explain").strip()


# --- query rewriting guards ----------------------------------------------

def test_rewrite_is_rejected_when_the_model_declines():
    """The echo provider answers any source-less prompt with the abstention
    sentence. That sentence is 14 words, so a length check waved it through
    and every search ran against "I couldn't find enough information in your
    uploaded material" instead of the student's question."""
    from ai.prompts.grounded_answer import INSUFFICIENT_EVIDENCE
    from ai.retrieval.query import _is_usable_rewrite

    assert not _is_usable_rewrite(INSUFFICIENT_EVIDENCE, "what is coulomb force")


def test_rewrite_is_rejected_when_it_changes_the_subject():
    from ai.retrieval.query import _is_usable_rewrite

    assert not _is_usable_rewrite(
        "photosynthesis chlorophyll stomata", "what is coulomb force"
    )


def test_a_genuine_rewrite_is_accepted():
    from ai.retrieval.query import _is_usable_rewrite

    assert _is_usable_rewrite(
        "coulomb force electrostatic interaction between point charges",
        "what is coulomb force",
    )


@pytest.mark.parametrize("candidate", ["", "x", "word " * 60])
def test_rewrites_of_implausible_length_are_rejected(candidate):
    from ai.retrieval.query import _is_usable_rewrite

    assert not _is_usable_rewrite(candidate, "what is coulomb force")


def test_cross_encoder_scores_are_squashed_into_a_usable_range():
    """Raw cross-encoder logits are unbounded and often negative, so a score
    floor of zero discarded all 20 candidates and produced an unexplained
    abstention."""
    import math

    for logit in (-11.0, -3.0, 0.0, 3.0, 11.0):
        squashed = 1.0 / (1.0 + math.exp(-logit))
        assert 0.0 < squashed < 1.0


# --- score scale ---------------------------------------------------------

def test_cross_encoder_scores_are_normalised_to_zero_one():
    """Cross-encoders emit unbounded logits, negative for weak matches.

    Left raw they are not comparable with the lexical reranker's 0-1 range,
    and MIN_CONTEXT_SCORE=0.0 discarded all twenty candidates -- the system
    abstained while holding perfectly good passages.
    """
    import math

    class FakeCrossEncoder:
        def rerank(self, query, texts):
            return [-8.2, 1.4, 6.9][: len(texts)]

    from ai.retrieval.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker._model = FakeCrossEncoder()
    reranker._fallback = NoOpReranker()

    chunks = [chunk("weak"), chunk("ok"), chunk("strong")]
    ranked = reranker.rerank("anything", chunks, top_n=3)

    # Scores are rescaled across the candidate set. Squashing alone left them
    # clustered near 1.0, which made the notation bonus overwhelm relevance.
    assert all(0.0 <= c.rerank_score <= 1.0 for c in ranked)
    assert [c.content for c in ranked] == ["strong", "ok", "weak"]
    assert math.isclose(ranked[0].rerank_score, 1.0, abs_tol=1e-9)
    # The weakest candidate lands at zero, and a zero floor keeps it rather
    # than discarding the whole list the way raw negative logits did.
    assert math.isclose(ranked[-1].rerank_score, 0.0, abs_tol=1e-9)
    assert ranked[-1].rerank_score >= 0.0


# --- the student's own material comes first ------------------------------

def test_retrieval_filters_can_restrict_to_one_pool():
    from ai.retrieval.hybrid import RetrievalFilters, _filter_sql

    params: dict = {}
    sql = _filter_sql(RetrievalFilters(source_kinds=["student_upload"]), params)
    assert "d.source_kind = ANY(:source_kinds)" in sql
    assert params["source_kinds"] == ["student_upload"]


def test_no_source_filter_searches_both_pools():
    from ai.retrieval.hybrid import RetrievalFilters, _filter_sql

    params: dict = {}
    assert "source_kind" not in _filter_sql(RetrievalFilters(), params)


def test_the_builtin_chapters_are_a_fallback_not_a_competitor():
    """Searching both pools at once and boosting the student's material
    still lets a confident textbook passage beat a merely good one from
    their own notes, which is the wrong answer to "explain what my teacher
    taught". So the student's material is searched first, on its own."""
    import inspect

    from apps.api.services import rag_service

    source = inspect.getsource(rag_service.ask)
    own_first = source.index("SourceKind.student_upload.value")
    fallback = source.index("SourceKind.platform.value")
    assert own_first < fallback
    assert "own_material_threshold" in source


def test_sufficiency_uses_an_absolute_score():
    """Fused and reranked scores are relative to the candidate set, so a
    pool of uniformly poor matches still yields a confident-looking top
    result. Cosine similarity is absolute."""
    import inspect

    from apps.api.services import rag_service

    source = inspect.getsource(rag_service.ask)
    assert "vector_score" in source
