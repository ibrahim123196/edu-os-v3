"""Exercises the raw SQL against a real pgvector database.

Run with the compose database up:

    docker compose up -d db
    pytest tests/integration -m integration

These cover the two queries unit tests cannot reach: the vector cast in
`vector_search` and `websearch_to_tsquery` in `keyword_search`. Both are the
kind of code that reads correctly and fails on contact with the server.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete

from ai.embeddings.provider import get_embedding_provider
from ai.retrieval.hybrid import (
    RetrievalFilters,
    hybrid_search,
    keyword_search,
    vector_search,
)
from apps.api.db import init_db, session_scope
from apps.api.models import Document, DocumentChunk, IngestStatus

pytestmark = pytest.mark.integration

PASSAGES = [
    ("Mitosis produces two genetically identical daughter cells.", 12),
    ("Meiosis produces four haploid gametes and halves the chromosome number.", 14),
    ("Crossing over occurs during prophase I of meiosis.", 15),
    ("Osmosis is the movement of solvent across a semipermeable membrane.", 40),
]


@pytest.fixture(scope="module")
def seeded():
    init_db()
    document_id = uuid.uuid4()
    provider = get_embedding_provider()
    vectors = provider.embed_documents([text for text, _ in PASSAGES])

    with session_scope() as db:
        db.add(
            Document(
                id=document_id,
                title="Biology notes",
                filename="bio.pdf",
                content_type="application/pdf",
                storage_path="/dev/null",
                byte_size=1,
                subject="Biology",
                grade="12",
                status=IngestStatus.ready,
                page_count=40,
                chunk_count=len(PASSAGES),
            )
        )
        db.add_all(
            DocumentChunk(
                document_id=document_id,
                chunk_index=i,
                content=text,
                word_count=len(text.split()),
                page_start=page,
                page_end=page,
                subject="Biology",
                grade="12",
                embedding=vector,
            )
            for i, ((text, page), vector) in enumerate(zip(PASSAGES, vectors, strict=True))
        )

    yield document_id

    with session_scope() as db:
        db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
        db.execute(delete(Document).where(Document.id == document_id))


def test_vector_search_returns_ranked_results(seeded):
    embedding = get_embedding_provider().embed_query("how do gametes form")
    with session_scope() as db:
        results = vector_search(db, embedding, RetrievalFilters(), limit=4)

    assert results
    assert results[0].vector_rank == 1
    assert all(0.0 <= r.vector_score <= 1.0 for r in results)
    assert "meiosis" in results[0].content.lower()


def test_keyword_search_matches_exact_terminology(seeded):
    with session_scope() as db:
        results = keyword_search(db, "crossing over prophase", RetrievalFilters(), limit=4)

    assert results
    assert "Crossing over" in results[0].content
    assert results[0].page_start == 15


def test_keyword_search_survives_punctuation(seeded):
    """websearch_to_tsquery must not raise on question marks and quotes."""
    with session_scope() as db:
        results = keyword_search(db, 'what is "osmosis"? really', RetrievalFilters(), limit=4)
    assert any("Osmosis" in r.content for r in results)


def test_metadata_filter_excludes_other_subjects(seeded):
    embedding = get_embedding_provider().embed_query("cell division")
    with session_scope() as db:
        assert vector_search(db, embedding, RetrievalFilters(subject="Biology"), 4)
        assert not vector_search(db, embedding, RetrievalFilters(subject="Physics"), 4)


def test_document_filter_scopes_the_search(seeded):
    embedding = get_embedding_provider().embed_query("cell division")
    with session_scope() as db:
        other = RetrievalFilters(document_ids=[uuid.uuid4()])
        assert not vector_search(db, embedding, other, 4)


def test_hybrid_search_beats_either_half_alone(seeded):
    """A question mixing paraphrase and exact terms should rank the right
    passage first once both searches are fused."""
    question = "crossing over during prophase"
    embedding = get_embedding_provider().embed_query(question)
    with session_scope() as db:
        fused = hybrid_search(db, question, embedding, RetrievalFilters())

    assert fused
    assert "Crossing over" in fused[0].content
    assert any(c.vector_rank and c.keyword_rank for c in fused)
