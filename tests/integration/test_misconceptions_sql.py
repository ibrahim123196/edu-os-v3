"""Named mistakes against a real database: loading the list, upgrading an
existing database, the repeat rule, and storing matches.

Local database only (see conftest): these create, change and delete rows,
and one drops a column on purpose.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import delete, inspect, select, text

from apps.api.curriculum import Concept, chunk_concepts
from apps.api.db import engine, init_db, session_scope
from apps.api.models import Document, DocumentChunk, IngestStatus
from apps.api.quiz import (
    Misconception,
    Question,
    QuestionKind,
    QuestionOption,
    QuestionStatus,
)
from apps.api.services import quiz_service
from apps.api.services.misconception_service import mistake_summary
from apps.api.services.quiz_service import grade
from apps.api.student import LOCAL_STUDENT_ID, TEST_STUDENT_ID
from scripts.apply_misconception_matches import check
from scripts.seed_misconceptions import UnknownConcepts, load_misconceptions

pytestmark = pytest.mark.integration

TAG = uuid.uuid4().hex[:6]
A, B = f"mc-a-{TAG}", f"mc-b-{TAG}"


def _file(labels: dict[str, list[tuple[str, str]]]) -> dict:
    return {"concepts": {c: [{"id": f"{c}/{s}", "label": l} for s, l in items]
                         for c, items in labels.items()}}


@pytest.fixture(scope="module")
def concepts():
    init_db()
    with session_scope() as db:
        db.add_all([Concept(slug=A, name="Concept A"), Concept(slug=B, name="Concept B")])
    yield
    with session_scope() as db:
        db.execute(delete(Concept).where(Concept.slug.in_([A, B])))


def _question(db, concept_slug: str, mistake_slugs: list[str | None]) -> Question:
    """One question: option 0 correct, the rest wrong, each wrong option
    linked to the named mistake given (or none)."""
    concept = db.scalar(select(Concept).where(Concept.slug == concept_slug))
    by_slug = {m.slug: m.id for m in db.scalars(select(Misconception)).all()}
    q = Question(concept_id=concept.id, kind=QuestionKind.multiple_choice,
                 stem=f"Q {uuid.uuid4().hex[:4]}?", explanation="b",
                 status=QuestionStatus.approved)
    q.options = [QuestionOption(position=0, text="right", is_correct=True)] + [
        QuestionOption(position=i + 1, text=f"wrong {i}", is_correct=False,
                       misconception_id=by_slug.get(s) if s else None)
        for i, s in enumerate(mistake_slugs)
    ]
    db.add(q)
    db.flush()
    return q


def _wrong(db, q: Question, n: int, student=TEST_STUDENT_ID):
    grade(db, q, q.options[n].id, student_id=student)


# ------------------------------------------------------------ loading

def test_loading_adds_then_updates_then_removes(concepts):
    with session_scope() as db:
        r = load_misconceptions(db, _file({A: [("x", "Old label"), ("y", "Y")]}))
        assert (r.added, r.updated, r.removed) == (2, 0, 0)

        r = load_misconceptions(db, _file({A: [("x", "New label")]}))
        assert (r.added, r.updated, r.removed) == (0, 1, 1)
        assert db.scalar(select(Misconception.label).where(Misconception.slug == f"{A}/x")) == "New label"


def test_a_mistake_in_use_is_kept_when_removed_from_the_file(concepts):
    """Deleting it would quietly strip the label from every option using it."""
    with session_scope() as db:
        load_misconceptions(db, _file({A: [("kept", "K")]}))
        _question(db, A, [f"{A}/kept"])
        r = load_misconceptions(db, _file({A: []}))
        assert r.removed == 0
        assert r.kept_in_use and f"{A}/kept" in r.kept_in_use[0]


def test_an_unknown_concept_refuses_the_whole_file(concepts):
    """A typo in one slug must not drop that concept's mistakes silently."""
    with session_scope() as db:
        before = db.scalar(select(text("count(*)")).select_from(Misconception))
        with pytest.raises(UnknownConcepts):
            load_misconceptions(db, _file({A: [("z", "Z")], "no-such-concept": [("q", "Q")]}))
        db.rollback()
    with session_scope() as db:
        assert db.scalar(select(text("count(*)")).select_from(Misconception)) == before


def test_init_db_adds_the_link_to_an_existing_database(concepts):
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE question_options DROP COLUMN misconception_id CASCADE"))
    init_db()
    insp = inspect(engine)
    assert "misconception_id" in {c["name"] for c in insp.get_columns("question_options")}
    assert any(fk["referred_table"] == "misconceptions"
               for fk in insp.get_foreign_keys("question_options"))


# ------------------------------------------------------- the repeat rule

@pytest.fixture
def listed(concepts):
    with session_scope() as db:
        load_misconceptions(db, _file({A: [("m1", "Mistake one"), ("m2", "Mistake two")],
                                       B: [("m3", "Mistake three")]}))
    yield
    with session_scope() as db:
        db.execute(delete(Question).where(Question.concept_id.in_(
            select(Concept.id).where(Concept.slug.in_([A, B])))))
        db.execute(text("DELETE FROM attempts WHERE question_id NOT IN (SELECT id FROM questions)"))


def _repeated(db, student=TEST_STUDENT_ID):
    return [r["slug"] for r in mistake_summary(db, student)["repeated"] if TAG in r["slug"]]


def test_the_same_mistake_on_two_questions_is_repeated(listed):
    with session_scope() as db:
        q1, q2 = _question(db, A, [f"{A}/m1"]), _question(db, A, [f"{A}/m1"])
        _wrong(db, q1, 1)
        _wrong(db, q2, 1)
        assert _repeated(db) == [f"{A}/m1"]


def test_one_question_wrong_twice_is_not_a_repeat(listed):
    """Remembering a wrong answer is not making a mistake twice."""
    with session_scope() as db:
        q = _question(db, A, [f"{A}/m1"])
        _wrong(db, q, 1)
        _wrong(db, q, 1)
        assert _repeated(db) == []


def test_different_mistakes_do_not_add_up(listed):
    with session_scope() as db:
        q1, q2 = _question(db, A, [f"{A}/m1"]), _question(db, A, [f"{A}/m2"])
        _wrong(db, q1, 1)
        _wrong(db, q2, 1)
        assert _repeated(db) == []


def test_another_students_mistakes_do_not_count(listed):
    with session_scope() as db:
        q1, q2 = _question(db, A, [f"{A}/m1"]), _question(db, A, [f"{A}/m1"])
        _wrong(db, q1, 1, student=LOCAL_STUDENT_ID)
        _wrong(db, q2, 1, student=LOCAL_STUDENT_ID)
        assert _repeated(db, TEST_STUDENT_ID) == []
        assert _repeated(db, LOCAL_STUDENT_ID) == [f"{A}/m1"]
        db.execute(delete(Question).where(Question.id.in_([q1.id, q2.id])))


def test_the_summary_says_how_much_it_had_to_go_on(listed):
    with session_scope() as db:
        before = mistake_summary(db, TEST_STUDENT_ID)
        q = _question(db, A, [f"{A}/m1", None])
        _wrong(db, q, 1)   # named
        _wrong(db, q, 2)   # wrong, but no named mistake
        after = mistake_summary(db, TEST_STUDENT_ID)
        assert after["wrong_answers"] - before["wrong_answers"] == 2
        assert after["wrong_answers_named"] - before["wrong_answers_named"] == 1


# ------------------------------------------------------- storing matches

def test_matches_are_checked_before_anything_is_stored(listed):
    with session_scope() as db:
        q = _question(db, A, [None, None, None])
        right, w1, w2, w3 = (str(o.id) for o in q.options)
        ready, problems = check(db, {
            w1: f"{A}/m1",              # fine
            w2: None,                   # fine: none of them fits
            w3: f"{B}/m3",              # another concept's mistake
            right: f"{A}/m1",           # a correct answer is not a mistake
            str(uuid.uuid4()): None,    # no such option
        })
        assert len(ready) == 2
        assert len(problems) == 3
        assert any("different concept" in p for p in problems)


# ------------------------------------------- generating with the list

class FakeLLM:
    """Writes one question; as the checker, picks `checker_picks`."""

    model = "fake"

    def __init__(self, checker_picks: int = 0):
        self.checker_picks = checker_picks

    def complete(self, system, messages, max_tokens):
        if system.startswith("You check physics exam questions"):
            return json.dumps({
                "options": [{"correct": i == self.checker_picks, "why": "-"} for i in range(4)],
                "false_premise": None, "tests_concept": True, "concept_note": "", "repeats": None,
            })
        return json.dumps({
            "stem": "What happens to the force?",
            "options": [
                {"text": "Quartered", "misconception": None, "mistake": None},
                {"text": "Halved", "misconception": "1/r", "mistake": f"{A}/m1"},
                {"text": "Doubled", "misconception": "?", "mistake": "other"},
                {"text": "Same", "misconception": "?", "mistake": "invented/id"},
            ],
            "correct_index": 0,
            "explanation": "Force goes as one over r squared, so it is quartered.",
        })


def _generate(db, monkeypatch, llm: FakeLLM) -> Question:
    # *args: the checker is fetched with get_llm(model) when one is set.
    monkeypatch.setattr(quiz_service, "get_llm", lambda *a, **k: llm)
    concept = db.scalar(select(Concept).where(Concept.slug == A))
    doc = Document(title="t", filename="t.pdf", content_type="application/pdf",
                   storage_path="x", byte_size=1, status=IngestStatus.ready)
    db.add(doc)
    db.flush()
    chunk = DocumentChunk(document_id=doc.id, chunk_index=0, page_start=1, page_end=1,
                          word_count=80, content="word " * 80)
    db.add(chunk)
    db.flush()
    db.execute(chunk_concepts.insert().values(chunk_id=chunk.id, concept_id=concept.id))
    quiz_service.generate_for_concept(db, concept, count=1)
    db.flush()
    return db.scalar(select(Question).where(Question.concept_id == concept.id,
                                            Question.stem == "What happens to the force?"))


def test_a_generated_option_is_stored_with_its_named_mistake(listed, monkeypatch):
    with session_scope() as db:
        q = _generate(db, monkeypatch, FakeLLM(checker_picks=0))
        named = [o.named_misconception.slug if o.named_misconception else None
                 for o in sorted(q.options, key=lambda o: o.position)]
        assert named == [None, f"{A}/m1", None, None]
        assert q.status == QuestionStatus.approved and q.checked_by == "fake"
        db.execute(delete(Document))


def test_a_question_the_checker_disagrees_with_is_not_used(listed, monkeypatch):
    """Answered blind, the checker picks another option: the key is
    suspect, so the question is kept for the record but never handed out."""
    with session_scope() as db:
        q = _generate(db, monkeypatch, FakeLLM(checker_picks=1))
        assert q.status == QuestionStatus.rejected
        assert "key says option 1, checker says 2" in q.rejection_reason
        assert q.checked_by == "fake"
        db.execute(delete(Document))


# ------------------------------------------------ what the student sees

def test_a_wrong_answer_shows_only_a_reviewed_mistake(listed):
    """The generator's free-text note is never sent: some were muddled and
    some described mistakes the student had not made."""
    with session_scope() as db:
        q = _question(db, A, [f"{A}/m1", None])
        q.options[1].misconception = "generator's own note"
        q.options[2].misconception = "another note"

        named = grade(db, q, q.options[1].id, student_id=TEST_STUDENT_ID)
        unnamed = grade(db, q, q.options[2].id, student_id=TEST_STUDENT_ID)
        right = grade(db, q, q.options[0].id, student_id=TEST_STUDENT_ID)

        assert named["mistake"] == "Mistake one"
        assert unnamed["mistake"] is None
        assert right["mistake"] is None
        assert all("misconception" not in r for r in (named, unnamed, right))
