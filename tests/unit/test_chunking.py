"""Chunking is where citations are won or lost.

If a chunk forgets its page, the answer cannot be checked. If chunks run to
double the target, the prompt fills with padding and the reranker has nothing
to discriminate on. Both were real bugs; these tests pin the fixes.
"""

from __future__ import annotations

import pytest

from ai.ingestion.chunking import chunk_pages
from ai.ingestion.extract import Page, clean_text

PROSE = (
    "Mitosis is a type of cell division. It produces two identical daughter "
    "cells. The chromosome number stays the same in each daughter cell. "
)


def test_returns_nothing_for_no_pages():
    assert chunk_pages([]) == []


def test_short_document_still_produces_one_chunk():
    chunks = chunk_pages([Page(3, "Osmosis is the movement of solvent.")])
    assert len(chunks) == 1
    assert chunks[0].page_start == 3


def test_chunks_stay_near_the_target_size():
    chunks = chunk_pages([Page(1, PROSE * 20)], target_words=120, overlap_words=30)
    assert len(chunks) > 1
    # Allowing 1.6x absorbs one trailing paragraph; 2x means the overlap
    # carry-over regression is back.
    assert max(c.word_count for c in chunks) <= 120 * 1.6


def test_text_without_sentence_punctuation_is_still_split():
    """PDF tables and equation dumps arrive as one huge unpunctuated run."""
    blob = " ".join(["chromosome"] * 900)
    chunks = chunk_pages([Page(1, blob)], target_words=100, overlap_words=20)
    assert len(chunks) > 4
    assert max(c.word_count for c in chunks) <= 160


def test_every_chunk_carries_a_page_range():
    pages = [Page(1, PROSE * 6), Page(2, PROSE * 6), Page(3, PROSE * 6)]
    chunks = chunk_pages(pages, target_words=100, overlap_words=25)
    for chunk in chunks:
        assert chunk.page_start is not None
        assert chunk.page_end >= chunk.page_start
        assert 1 <= chunk.page_start <= 3


def test_consecutive_chunks_overlap():
    chunks = chunk_pages([Page(1, PROSE * 20)], target_words=80, overlap_words=25)
    assert len(chunks) >= 2
    tail = " ".join(chunks[0].content.split()[-8:])
    assert tail in " ".join(chunks[1].content.split())


def test_overlap_is_never_emitted_as_its_own_chunk():
    """Carried-over text alone must not become a standalone chunk."""
    chunks = chunk_pages([Page(1, PROSE * 25)], target_words=100, overlap_words=30)
    # A chunk at roughly the overlap size means carry-over was flushed alone.
    assert all(c.word_count > 35 for c in chunks[:-1])


def test_headings_are_attached_to_the_text_below_them():
    pages = [Page(1, "Cell Division\n\n" + PROSE * 4)]
    chunks = chunk_pages(pages, target_words=200, overlap_words=20)
    assert chunks[0].section == "Cell Division"


def test_chunk_indexes_are_contiguous():
    chunks = chunk_pages([Page(1, PROSE * 30)], target_words=90, overlap_words=20)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_rejects_overlap_larger_than_target():
    with pytest.raises(ValueError):
        chunk_pages([Page(1, PROSE)], target_words=50, overlap_words=50)


def test_clean_text_rejoins_hyphenated_line_breaks():
    assert "photosynthesis" in clean_text("photo-\nsynthesis occurs")


def test_clean_text_preserves_paragraph_breaks():
    assert clean_text("One.\n\n\n\nTwo.") == "One.\n\nTwo."
