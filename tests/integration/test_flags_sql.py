"""Flagging a question that looks wrong, against a real database.

Local database only (see conftest).
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from apps.api.curriculum import Concept
from apps.api.db import init_db, session_scope
from apps.api.main import app
from apps.api.quiz import Question, QuestionFlag, QuestionKind, QuestionOption, QuestionStatus
from apps.api.services import flag_service
from apps.api.services.quiz_service import grade, select_questions
from apps.api.student import ConceptMastery, TEST_STUDENT_ID
from scripts import mark_faulty_questions, review_flags

pytestmark = pytest.mark.integration

SLUG = f"flags-{uuid.uuid4().hex[:6]}"


@pytest.fixture
def concept():
    init_db()
    with session_scope() as db:
        c = Concept(slug=SLUG, name="Flags")
        db.add(c)
        db.flush()
        cid = c.id
    yield cid
    with session_scope() as db:
        db.execute(delete(Question).where(Question.concept_id == cid))
        db.execute(delete(Concept).where(Concept.id == cid))


def _question(db, concept_id, **kw) -> Question:
    q = Question(concept_id=concept_id, kind=QuestionKind.multiple_choice,
                 stem=f"Q {uuid.uuid4().hex[:4]}?", explanation="b",
                 status=kw.pop("status", QuestionStatus.approved), **kw)
    q.options = [QuestionOption(position=0, text="keyed", is_correct=True),
                 QuestionOption(position=1, text="other", is_correct=False)]
    db.add(q)
    db.flush()
    return q


def _served(db, concept_id) -> set:
    return {q.id for q in select_questions(db, concept_ids=[concept_id], count=50,
                                           student_id=TEST_STUDENT_ID)}


def _mastery(db, concept_id):
    return db.scalar(select(ConceptMastery).where(
        ConceptMastery.student_id == TEST_STUDENT_ID, ConceptMastery.concept_id == concept_id))


def test_a_flagged_question_leaves_quizzes_until_restored(concept):
    with session_scope() as db:
        q = _question(db, concept)
        flag_service.flag(db, student_id=TEST_STUDENT_ID, question_id=q.id,
                          reason="wrong_key", note=None)
        assert q.id not in _served(db, concept)
        flag_service.resolve(db, q.id, retire=False)
        db.flush()
        assert q.id in _served(db, concept)
        assert q.status == QuestionStatus.approved and not q.faulty


def test_retiring_a_flag_stops_its_answers_counting(concept):
    """Wrong against a wrong key is not wrong."""
    with session_scope() as db:
        good, bad = _question(db, concept), _question(db, concept)
        grade(db, good, good.options[0].id, student_id=TEST_STUDENT_ID)
        grade(db, bad, bad.options[1].id, student_id=TEST_STUDENT_ID)
        db.flush()
        assert (_mastery(db, concept).attempts, _mastery(db, concept).correct) == (2, 1)

        flag_service.flag(db, student_id=TEST_STUDENT_ID, question_id=bad.id,
                          reason="wrong_key", note="key says quarter, should be half")
        flag_service.resolve(db, bad.id, retire=True)
        db.flush()

        assert bad.faulty and bad.status == QuestionStatus.rejected
        assert bad.rejection_reason == "faulty (flagged): The answer key is wrong"
        assert (_mastery(db, concept).attempts, _mastery(db, concept).correct) == (1, 1)


def test_a_second_click_is_the_same_flag(concept):
    with session_scope() as db:
        q = _question(db, concept)
        for _ in range(2):
            flag_service.flag(db, student_id=TEST_STUDENT_ID, question_id=q.id,
                              reason="unclear", note=None)
        n = db.scalar(select(func.count()).select_from(QuestionFlag)
                      .where(QuestionFlag.question_id == q.id))
        assert n == 1


def test_the_flag_endpoint(concept):
    with session_scope() as db:
        qid = str(_question(db, concept).id)
    with TestClient(app) as client:
        assert client.post("/quiz/flag", json={"question_id": qid, "reason": "two_right"}).status_code == 200
        assert client.post("/quiz/flag", json={"question_id": qid, "reason": "nonsense"}).status_code == 400
        assert client.post("/quiz/flag", json={"question_id": str(uuid.uuid4()),
                                               "reason": "other"}).status_code == 400
        assert {r["key"] for r in client.get("/quiz/flag-reasons").json()} == {
            "wrong_key", "two_right", "unclear", "other"}


def test_review_refuses_unclear_numbers(concept, capsys):
    with session_scope() as db:
        q = _question(db, concept)
        flag_service.flag(db, student_id=TEST_STUDENT_ID, question_id=q.id,
                          reason="other", note=None)
        n = len(flag_service.open_flags(db))
    assert review_flags.main(["--retire", str(n), "--restore", str(n)]) == 1
    assert review_flags.main(["--retire", str(n + 5)]) == 1
    with session_scope() as db:
        assert db.get(Question, q.id).status == QuestionStatus.approved
        flag_service.resolve(db, q.id, retire=False)


def test_backfill_marks_only_faulty_retirements(concept):
    with session_scope() as db:
        key = _question(db, concept, status=QuestionStatus.rejected,
                        rejection_reason="faulty: answer key is wrong")
        copy = _question(db, concept, status=QuestionStatus.rejected,
                         rejection_reason="check: repeats: an earlier one")
        grade(db, key, key.options[1].id, student_id=TEST_STUDENT_ID)
        grade(db, copy, copy.options[0].id, student_id=TEST_STUDENT_ID)
        db.flush()
        assert _mastery(db, concept).attempts == 2
        key_id, copy_id = key.id, copy.id

    mark_faulty_questions.main(apply=True)

    with session_scope() as db:
        assert db.get(Question, key_id).faulty
        assert not db.get(Question, copy_id).faulty
        assert _mastery(db, concept).attempts == 1    # the repeat's answer still counts
