"""When a topic was last practised: concept_mastery.last_attempt_at.

The column existed from the start but nothing ever filled it.
Local database only (see conftest).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, select, update

from apps.api.curriculum import Concept
from apps.api.db import init_db, session_scope
from apps.api.quiz import Attempt, Question, QuestionKind, QuestionOption, QuestionStatus
from apps.api.services.mastery_service import recompute
from apps.api.services.quiz_service import grade
from apps.api.student import TEST_STUDENT_ID, ConceptMastery

pytestmark = pytest.mark.integration

DAY = timedelta(days=1)
LONG_AGO = datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def concept():
    init_db()
    with session_scope() as db:
        c = Concept(slug=f"last-{uuid.uuid4().hex[:6]}", name="Last attempt")
        db.add(c)
        db.flush()
        cid = c.id
    yield cid
    with session_scope() as db:
        db.execute(delete(Question).where(Question.concept_id == cid))
        db.execute(delete(Concept).where(Concept.id == cid))


def _question(db, concept_id) -> Question:
    q = Question(concept_id=concept_id, kind=QuestionKind.multiple_choice,
                 stem=f"Q {uuid.uuid4().hex[:4]}?", explanation="b",
                 status=QuestionStatus.approved)
    q.options = [QuestionOption(position=0, text="keyed", is_correct=True),
                 QuestionOption(position=1, text="other", is_correct=False)]
    db.add(q)
    db.flush()
    return q


def _answer(db, question, when: datetime) -> None:
    """Answer, then move the answer to a chosen time: answers within one
    transaction all get the same clock time otherwise."""
    grade(db, question, question.options[0].id, student_id=TEST_STUDENT_ID)
    db.flush()
    db.execute(update(Attempt)
               .where(Attempt.question_id == question.id, Attempt.student_id == TEST_STUDENT_ID)
               .values(created_at=when))


def _row(db, concept_id) -> ConceptMastery:
    return db.scalar(select(ConceptMastery).where(
        ConceptMastery.student_id == TEST_STUDENT_ID, ConceptMastery.concept_id == concept_id))


def test_answering_fills_it_in(concept):
    with session_scope() as db:
        grade(db, _question(db, concept), None, student_id=TEST_STUDENT_ID)
        db.flush()
        row = _row(db, concept)
        latest = db.scalar(select(Attempt.created_at)
                           .where(Attempt.student_id == TEST_STUDENT_ID)
                           .order_by(Attempt.created_at.desc()).limit(1))
        assert row.last_attempt_at is not None and row.last_attempt_at == latest


def test_it_is_the_newest_answer_not_the_first(concept):
    with session_scope() as db:
        _answer(db, _question(db, concept), LONG_AGO)
        _answer(db, _question(db, concept), LONG_AGO + 3 * DAY)
        recompute(db, TEST_STUDENT_ID, concept)
        assert _row(db, concept).last_attempt_at == LONG_AGO + 3 * DAY


def test_an_answer_to_a_faulty_question_does_not_count(concept):
    """Same rule as the estimate: it says nothing about the student."""
    with session_scope() as db:
        _answer(db, _question(db, concept), LONG_AGO)
        bad = _question(db, concept)
        _answer(db, bad, LONG_AGO + 5 * DAY)
        bad.faulty = True
        db.flush()
        recompute(db, TEST_STUDENT_ID, concept)
        assert _row(db, concept).last_attempt_at == LONG_AGO


def test_startup_fills_rows_from_before(concept):
    with session_scope() as db:
        _answer(db, _question(db, concept), LONG_AGO)
        recompute(db, TEST_STUDENT_ID, concept)
        _row(db, concept).last_attempt_at = None      # as every row was before

    init_db()

    with session_scope() as db:
        assert _row(db, concept).last_attempt_at == LONG_AGO
