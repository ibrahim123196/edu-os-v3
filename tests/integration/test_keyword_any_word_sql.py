"""Keyword search matches any word, and ranks passages with more higher.

It used to require every word, so one word a textbook never uses --
"explain", "really", a synonym from the query rewrite -- made it find
nothing. Local database only (see conftest).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete

from ai.retrieval.hybrid import RetrievalFilters, keyword_search
from apps.api.config import settings
from apps.api.db import init_db, session_scope
from apps.api.models import Document, DocumentChunk, IngestStatus

pytestmark = pytest.mark.integration

TAG = uuid.uuid4().hex[:6]
PASSAGES = {
    "drift": "The drift velocity of electrons is small compared with their thermal speed.",
    "field": "The electric field inside a conductor is zero in electrostatics.",
    "magnet": "Drift of charges in a magnetic field bends their path.",
}
DOC: dict = {}


@pytest.fixture(scope="module")
def corpus():
    init_db()
    with session_scope() as db:
        doc = Document(title=f"kw {TAG}", filename="kw.txt", content_type="text/plain",
                       storage_path="x", byte_size=1, status=IngestStatus.ready)
        db.add(doc)
        db.flush()
        for i, content in enumerate(PASSAGES.values()):
            # Search only considers embedded passages; any vector will do here.
            db.add(DocumentChunk(document_id=doc.id, chunk_index=i, page_start=i + 1,
                                 page_end=i + 1, word_count=12, content=content,
                                 embedding=[0.1] * settings.embedding_dim))
        doc_id = DOC["id"] = doc.id
    yield
    with session_scope() as db:
        db.execute(delete(Document).where(Document.id == doc_id))


def _found(query: str) -> list[str]:
    """Which of our passages come back, best first."""
    # Searched within this test's own document: with any-word matching, the
    # textbook's passages about fields would otherwise crowd these out.
    with session_scope() as db:
        rows = keyword_search(db, query, RetrievalFilters(document_ids=[DOC["id"]]), limit=50)
    names = {v: k for k, v in PASSAGES.items()}
    return [names[r.content] for r in rows if r.content in names]


def test_a_word_no_passage_contains_no_longer_empties_the_search(corpus):
    assert _found("drift velocity really")[0] == "drift"


def test_explain_in_simple_words_still_finds_the_passage(corpus):
    assert _found("explain drift velocity in simple words")[0] == "drift"


def test_more_matching_words_rank_higher(corpus):
    """'drift' alone is in two passages; the one with 'velocity' too wins."""
    found = _found("drift velocity")
    assert found.index("drift") < found.index("magnet")


def test_an_excluded_word_still_excludes(corpus):
    """Swapping & for | would turn 'drift -magnetic' into 'drift, or
    velocity without magnetic' and let the magnetic passage through."""
    found = _found("drift velocity -magnetic")
    assert "magnet" not in found and found == ["drift"]


def test_a_quoted_phrase_still_needs_its_words_together(corpus):
    found = _found('"electric field" zero')
    assert found[0] == "field"
