"""Trigram spelling correction against a real database.

The unit tests stub the database to cover the logic. These check the part that
only Postgres can answer: whether pg_trgm actually scores real typos above the
threshold and unrelated words below it.

Run with the database up:
    pytest tests/integration -m integration
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete, text

from ai.retrieval.spelling import correct_query, rebuild_vocabulary
from apps.api.db import init_db, session_scope
from apps.api.models import CorpusTerm, Document, DocumentChunk, IngestStatus

pytestmark = pytest.mark.integration

PASSAGES = [
    "An electron carries a negative charge and orbits the nucleus of an atom.",
    "The Coulomb force between two point charges is proportional to the "
    "product of the charges and inversely proportional to the square of the "
    "distance between them.",
    "Kirchhoff's voltage law states that the sum of potential differences "
    "around any closed loop is zero.",
    "Mitosis produces two genetically identical daughter cells.",
    "Meiosis produces four haploid gametes and halves the chromosome number.",
    "Photosynthesis converts light energy into chemical energy inside the "
    "chloroplast of a plant cell.",
]


@pytest.fixture(scope="module")
def vocabulary():
    init_db()
    document_id = uuid.uuid4()

    with session_scope() as db:
        db.add(
            Document(
                id=document_id,
                title="Science notes",
                filename="science.pdf",
                content_type="application/pdf",
                storage_path="/dev/null",
                byte_size=1,
                status=IngestStatus.ready,
            )
        )
        db.add_all(
            DocumentChunk(
                document_id=document_id,
                chunk_index=i,
                content=passage,
                word_count=len(passage.split()),
                page_start=i + 1,
                page_end=i + 1,
            )
            for i, passage in enumerate(PASSAGES)
        )

    with session_scope() as db:
        rebuild_vocabulary(db)

    yield

    with session_scope() as db:
        db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
        db.execute(delete(Document).where(Document.id == document_id))
        db.execute(delete(CorpusTerm))


def test_vocabulary_contains_terms_from_the_passages(vocabulary):
    with session_scope() as db:
        terms = {
            r[0] for r in db.execute(text("SELECT term FROM corpus_terms")).fetchall()
        }
    assert "coulomb" in terms
    assert "kirchhoff" in terms
    assert "photosynthesis" in terms
    # Short words are not stored; trigram matching is unreliable on them.
    assert "of" not in terms


@pytest.mark.parametrize(
    "typed,expected",
    [
        ("electorn", "electron"),   # transposed letters, trigram score 0.385
        ("coulumb", "coulomb"),
        ("colomb", "coulomb"),
        ("kirchoff", "kirchhoff"),
        ("photosynthasis", "photosynthesis"),
        ("chromosme", "chromosome"),
    ],
)
def test_real_typos_are_corrected(vocabulary, typed, expected):
    with session_scope() as db:
        corrected, changes = correct_query(db, f"what is {typed} force")
    assert expected in corrected.lower()
    assert changes and changes[0].corrected == expected


def test_mitosis_is_never_turned_into_meiosis(vocabulary):
    """Both are real words in the corpus, so neither should be rewritten.
    Getting this wrong would silently answer the opposite question."""
    with session_scope() as db:
        corrected, changes = correct_query(db, "explain mitosis in detail")
    assert "mitosis" in corrected
    assert changes == []


def test_unrelated_unknown_word_is_left_alone(vocabulary):
    with session_scope() as db:
        corrected, changes = correct_query(db, "explain zwitterion chemistry")
    assert "zwitterion" in corrected
    assert not any(c.original == "zwitterion" for c in changes)


def test_correct_spelling_passes_through_untouched(vocabulary):
    with session_scope() as db:
        corrected, changes = correct_query(db, "Coulomb force between charges")
    assert corrected == "Coulomb force between charges"
    assert changes == []


def test_possessive_is_handled(vocabulary):
    with session_scope() as db:
        corrected, _ = correct_query(db, "state Kirchoff's voltage law")
    assert "Kirchhoff's" in corrected


def test_vocabulary_rebuild_is_idempotent(vocabulary):
    with session_scope() as db:
        first = rebuild_vocabulary(db)
    with session_scope() as db:
        second = rebuild_vocabulary(db)
    assert first == second
