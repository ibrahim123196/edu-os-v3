"""Each student's typos are fixed against the textbook and their own uploads,
never another student's. Local database only."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete

from ai.retrieval.spelling import correct_query, rebuild_vocabulary
from apps.api.db import init_db, session_scope
from apps.api.models import Document, DocumentChunk, IngestStatus
from apps.api.student import Student

pytestmark = pytest.mark.integration


@pytest.fixture
def world():
    init_db()
    tag = uuid.uuid4().hex[:6]
    ids = {}
    with session_scope() as db:
        for who in ("a", "b"):
            s = Student(id=uuid.uuid4(), display_name=f"Spell {who} {tag}")
            db.add(s)
            db.flush()
            ids[who] = s.id

        def doc(owner, kind, text):
            d = Document(title=kind, filename="x.txt", content_type="text/plain",
                         storage_path="x", byte_size=1, status=IngestStatus.ready,
                         source_kind=kind, student_id=owner)
            db.add(d)
            db.flush()
            db.add(DocumentChunk(document_id=d.id, chunk_index=0, word_count=8, content=text))
            return d.id

        ids["docs"] = [
            # Only A's notes have this made-up word.
            doc(ids["a"], "student_upload", "My grandmother Zubaida explained the zubaidarium."),
            doc(None, "platform", "The capacitance of a parallel plate capacitor."),
        ]
        db.flush()
        rebuild_vocabulary(db)
    yield ids
    with session_scope() as db:
        db.execute(delete(Document).where(Document.id.in_(ids["docs"])))
        db.execute(delete(Student).where(Student.id.in_([ids["a"], ids["b"]])))
        rebuild_vocabulary(db)


def test_a_students_own_words_fix_their_typos(world):
    with session_scope() as db:
        fixed, _ = correct_query(db, "what is the zubaidarim", student_id=world["a"])
    assert "zubaidarium" in fixed


def test_another_students_words_are_never_offered(world):
    with session_scope() as db:
        fixed, corrections = correct_query(db, "what is the zubaidarim", student_id=world["b"])
    assert "zubaidarium" not in fixed and fixed == "what is the zubaidarim"


def test_textbook_words_fix_everyones_typos(world):
    with session_scope() as db:
        for who in ("a", "b"):
            fixed, _ = correct_query(db, "what is capacitence", student_id=world[who])
            assert "capacitance" in fixed


def test_uploads_are_not_in_the_shared_list(world):
    from sqlalchemy import text

    with session_scope() as db:
        shared = db.execute(text("SELECT 1 FROM corpus_terms WHERE term = 'zubaidarium'")).first()
        own = db.execute(text("SELECT student_id FROM student_terms WHERE term = 'zubaidarium'")).all()
    assert shared is None and [r[0] for r in own] == [world["a"]]


def test_a_request_searches_with_its_students_words(world, monkeypatch):
    """The pipeline passes the student through to the spelling fixer."""
    from ai.retrieval import query

    seen = {}

    def spy(db, q, min_similarity=0.4, student_id=None):
        seen["student"] = student_id
        return q, []

    monkeypatch.setattr(query, "correct_query", spy)
    with session_scope() as db:
        query.prepare_query("what is zubaidarim", db=db, student_id=world["b"])
    assert seen["student"] == world["b"]
