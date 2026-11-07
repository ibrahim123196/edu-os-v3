"""Citation handling.

A citation the student cannot open is worse than none at all, because it looks
like proof. These tests cover the ways a model can produce one.
"""

from __future__ import annotations

import uuid

from ai.generation.answer import build_answer
from ai.generation.llm import EchoClient
from ai.prompts.grounded_answer import (
    INSUFFICIENT_EVIDENCE,
    build_messages,
    format_context,
)
from ai.retrieval.hybrid import RetrievedChunk


def chunk(content: str, page: int, title: str = "Biology notes") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title=title,
        content=content,
        page_start=page,
        page_end=page,
        section="Cell Division",
        chunk_index=0,
    )


CHUNKS = [
    chunk("Mitosis produces two identical daughter cells.", 12),
    chunk("Meiosis produces four gametes.", 14),
    chunk("Osmosis is unrelated to cell division.", 20),
]


def test_valid_markers_become_citations():
    result = build_answer("Mitosis is equational [S1]. Meiosis is reductional [S2].", CHUNKS)
    assert result.grounded
    assert [c.marker for c in result.citations] == ["S1", "S2"]
    assert result.citations[0].page_label == "Page 12"
    assert result.citations[1].page_label == "Page 14"


def test_markers_pointing_at_missing_sources_are_removed():
    """The model cited [S9] when only three sources existed."""
    result = build_answer("Cells divide [S1]. Also relevant [S9].", CHUNKS)
    assert "[S9]" not in result.answer
    assert len(result.citations) == 1


def test_sources_are_renumbered_contiguously():
    """The model used only source 2; the student should see it as 1."""
    result = build_answer("Meiosis halves the chromosome number [S2].", CHUNKS)
    assert "[S1]" in result.answer
    assert result.citations[0].marker == "S1"
    assert "Meiosis" in result.citations[0].snippet


def test_renumbering_follows_order_of_first_appearance():
    result = build_answer("Second [S2]. First [S1]. Second again [S2].", CHUNKS)
    assert result.answer.startswith("Second [S1]")
    assert result.citations[0].snippet.startswith("Meiosis")


def test_abstention_is_reported_as_ungrounded():
    result = build_answer(INSUFFICIENT_EVIDENCE, CHUNKS)
    assert result.grounded is False
    assert result.citations == []


def test_empty_model_output_falls_back_to_abstention():
    result = build_answer("", CHUNKS)
    assert result.grounded is False
    assert result.answer == INSUFFICIENT_EVIDENCE


def test_answer_without_any_markers_is_not_grounded():
    """Prose with no citations means the model stopped tracking its evidence."""
    result = build_answer("Mitosis and meiosis are both forms of cell division.", CHUNKS)
    assert result.grounded is False
    assert result.used_source_count == 0


def test_snippets_are_truncated_on_a_boundary():
    long_chunk = chunk("Word " * 400, 5)
    result = build_answer("See this [S1].", [long_chunk])
    snippet = result.citations[0].snippet
    assert len(snippet) <= 340
    assert not snippet.endswith("Wor")


def test_page_range_spanning_two_pages_is_labelled_as_a_range():
    spanning = RetrievedChunk(
        chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), document_title="Notes",
        content="text", page_start=7, page_end=8, section=None, chunk_index=0,
    )
    result = build_answer("Claim [S1].", [spanning])
    assert result.citations[0].page_label == "Pages 7-8"


# --- context construction ------------------------------------------------

def test_context_numbers_sources_and_labels_pages():
    context = format_context(CHUNKS)
    assert "[S1] Biology notes" in context
    assert "Page 12" in context
    assert "[S3]" in context


def test_context_drops_weakest_sources_rather_than_truncating():
    big = [chunk("x " * 3000, i) for i in range(1, 6)]
    context = format_context(big, max_chars=8000)
    assert "[S1]" in context
    assert "[S5]" not in context


def test_prompt_contains_the_question_and_the_sources():
    messages = build_messages("What is meiosis?", CHUNKS)
    assert "What is meiosis?" in messages[0]["content"]
    assert "[S2]" in messages[0]["content"]


# --- offline provider ----------------------------------------------------

def test_echo_provider_abstains_when_given_no_sources():
    reply = EchoClient().complete("system", [{"role": "user", "content": "no sources here"}])
    assert reply == INSUFFICIENT_EVIDENCE


def test_echo_provider_cites_when_given_sources():
    messages = build_messages("What is mitosis?", CHUNKS)
    reply = EchoClient().complete("system", messages)
    assert "[S1]" in reply
