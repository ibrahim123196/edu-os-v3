"""Test me: timed tests that can be left and continued; quizzes that can be
ended early. Through the real API against a real database.

Sessions are started directly with known questions, so what a test draws
from the rest of the bank cannot change the outcome. The clock is moved by
editing timing_since rather than by waiting. Local database only (see
conftest).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from apps.api.curriculum import Concept
from apps.api.db import init_db, session_scope
from apps.api.main import app
from apps.api.quiz import Question, QuestionKind, QuestionOption, QuestionStatus, QuizSession
from apps.api.services import session_service
from apps.api.student import active_student_id

pytestmark = pytest.mark.integration

TAG = "t" + uuid.uuid4().hex[:6]


@pytest.fixture(scope="module")
def client():
    init_db()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def questions(client):
    with session_scope() as db:
        concept = Concept(slug=f"testme-{TAG}", name=f"Test me {TAG}")
        db.add(concept)
        db.flush()
        for i in range(4):
            q = Question(concept_id=concept.id, kind=QuestionKind.multiple_choice,
                         stem=f"Test me {TAG} {i}?", explanation="b",
                         status=QuestionStatus.approved)
            q.options = [QuestionOption(position=0, text="right", is_correct=True),
                         QuestionOption(position=1, text="wrong", is_correct=False)]
            db.add(q)
        concept_id = concept.id
    yield concept_id
    with session_scope() as db:
        ids = select(Question.id).where(Question.concept_id == concept_id)
        from apps.api.quiz import QuizSessionQuestion
        sessions = select(QuizSessionQuestion.session_id).where(
            QuizSessionQuestion.question_id.in_(ids))
        db.execute(delete(QuizSession).where(QuizSession.id.in_(sessions)))
        db.execute(delete(Concept).where(Concept.id == concept_id))


def _start(kind: str, concept_id) -> tuple[str, list[dict]]:
    """A session over our four questions, with their right answers."""
    with session_scope() as db:
        qs = db.scalars(select(Question).where(Question.concept_id == concept_id)
                        .order_by(Question.stem)).all()
        s = session_service.start_session(db, active_student_id(), qs, kind=kind)
        sid = str(s.id)
        plan = [{"id": str(q.id), "right": str(next(o.id for o in q.options if o.is_correct))}
                for q in qs]
    return sid, plan


def _answer(client, sid, q):
    return client.post("/quiz/answer", json={"session_id": sid, "question_id": q["id"],
                                             "selected_option_id": q["right"]})


def _rewind(sid, seconds):
    """Pretend the running stretch started `seconds` ago."""
    with session_scope() as db:
        s = db.get(QuizSession, uuid.UUID(sid))
        s.timing_since = datetime.now(timezone.utc) - timedelta(seconds=seconds)


def _row(client, sid):
    return next(h for h in client.get("/quiz/sessions").json() if h["id"] == sid)


# --- tests: timed, left, continued -----------------------------------------------

def test_a_test_is_timed_and_a_quiz_is_not(client, questions):
    test, _ = _start("test", questions)
    quiz, _ = _start("practice", questions)
    _rewind(test, 90)
    assert 90 <= _row(client, test)["seconds"] < 100
    assert _row(client, quiz)["seconds"] is None
    assert _row(client, test)["kind"] == "test"


def test_leaving_stops_the_clock_and_keeps_the_answers(client, questions):
    sid, plan = _start("test", questions)
    _rewind(sid, 120)
    assert _answer(client, sid, plan[0]).status_code == 200
    left = client.post(f"/quiz/sessions/{sid}/leave").json()
    assert left["stopped"] == "left" and left["answered"] == 1
    banked = left["elapsed_seconds"]
    assert 120 <= banked < 130

    # Time away does not count: an hour later it still says the same.
    with session_scope() as db:
        s = db.get(QuizSession, uuid.UUID(sid))
        assert s.timing_since is None
    row = _row(client, sid)
    assert row["seconds"] == banked and row["can_continue"] and row["stopped"] == "left"


def test_a_left_test_takes_no_answers_until_continued(client, questions):
    sid, plan = _start("test", questions)
    client.post(f"/quiz/sessions/{sid}/leave")
    r = _answer(client, sid, plan[0])
    assert r.status_code == 409 and "Continue" in r.json()["detail"]


def test_continuing_returns_every_question_with_the_answered_ones_locked(client, questions):
    """Going back to look at an answered question must work after
    continuing too; answering it again must not."""
    sid, plan = _start("test", questions)
    _answer(client, sid, plan[0])
    _answer(client, sid, plan[1])
    client.post(f"/quiz/sessions/{sid}/leave")

    back = client.post(f"/quiz/sessions/{sid}/resume").json()
    assert [q["id"] for q in back["questions"]] == [q["id"] for q in plan]
    assert (back["answered"], back["total"], back["stopped"]) == (2, 4, None)
    first, third = back["questions"][0], back["questions"][2]
    assert first["selected_option_id"] == plan[0]["right"]
    assert first["result"]["correct"] and first["result"]["correct_option_id"] == plan[0]["right"]
    assert "result" not in third and "selected_option_id" not in third
    # The right answer travels only with questions already answered.
    assert all("is_correct" not in o for q in back["questions"] for o in q["options"])
    assert _answer(client, sid, plan[0]).status_code == 409      # locked
    assert _answer(client, sid, plan[2]).status_code == 200


def test_questions_can_be_answered_in_any_order(client, questions):
    sid, plan = _start("test", questions)
    for q in (plan[3], plan[0], plan[2]):
        assert _answer(client, sid, q).json()["session"]["finished"] is False
    assert _answer(client, sid, plan[1]).json()["session"]["finished"] is True


def test_the_clock_carries_on_after_continuing(client, questions):
    sid, plan = _start("test", questions)
    _rewind(sid, 60)
    client.post(f"/quiz/sessions/{sid}/leave")
    client.post(f"/quiz/sessions/{sid}/resume")
    _rewind(sid, 30)
    for q in plan:
        _answer(client, sid, q)
    row = _row(client, sid)
    assert row["finished"] and not row["can_continue"]
    assert 90 <= row["seconds"] < 100              # 60 before leaving + 30 after


def test_a_forgotten_test_does_not_come_back_hours_long(client, questions):
    """A tab closed without Leave keeps its clock 'running' until touched."""
    sid, _ = _start("test", questions)
    _rewind(sid, 5 * 3600)
    assert _row(client, sid)["seconds"] == session_service.MAX_STRETCH_SECONDS
    assert _row(client, sid)["can_continue"]        # and it can still be continued
    back = client.post(f"/quiz/sessions/{sid}/resume").json()
    assert back["elapsed_seconds"] == session_service.MAX_STRETCH_SECONDS


def test_a_finished_test_cannot_be_left_or_continued(client, questions):
    sid, plan = _start("test", questions)
    for q in plan:
        _answer(client, sid, q)
    assert client.post(f"/quiz/sessions/{sid}/leave").status_code == 409
    assert client.post(f"/quiz/sessions/{sid}/resume").status_code == 409


def test_answering_the_last_question_stops_the_clock(client, questions):
    sid, plan = _start("test", questions)
    for q in plan[:-1]:
        _answer(client, sid, q)
    last = _answer(client, sid, plan[-1]).json()["session"]
    assert last["finished"] and "elapsed_seconds" in last
    with session_scope() as db:
        assert db.get(QuizSession, uuid.UUID(sid)).timing_since is None


# --- quizzes: ended early ------------------------------------------------------

def test_ending_a_quiz_keeps_what_was_answered(client, questions):
    sid, plan = _start("practice", questions)
    _answer(client, sid, plan[0])
    ended = client.post(f"/quiz/sessions/{sid}/end").json()
    assert (ended["stopped"], ended["answered"], ended["correct"], ended["total"]) == ("ended", 1, 1, 4)
    row = _row(client, sid)
    assert row["stopped"] == "ended" and not row["can_continue"] and not row["finished"]


def test_an_ended_quiz_takes_no_more_answers(client, questions):
    sid, plan = _start("practice", questions)
    client.post(f"/quiz/sessions/{sid}/end")
    assert _answer(client, sid, plan[0]).status_code == 409


def test_a_quiz_cannot_be_left_or_continued_and_a_test_cannot_be_ended(client, questions):
    quiz, _ = _start("practice", questions)
    test, _ = _start("test", questions)
    assert client.post(f"/quiz/sessions/{quiz}/leave").status_code == 409
    assert client.post(f"/quiz/sessions/{quiz}/resume").status_code == 409
    assert client.post(f"/quiz/sessions/{test}/end").status_code == 409


# --- starting ------------------------------------------------------------------

def test_a_test_covers_every_topic(client, questions):
    r = client.post("/quiz", json={"kind": "test", "concept_slugs": [f"testme-{TAG}"]})
    assert r.status_code == 400


def test_starting_a_test_through_the_api(client, questions):
    r = client.post("/quiz", json={"kind": "test", "count": 3}).json()
    assert r["kind"] == "test" and r["count"] >= 1
    with session_scope() as db:
        s = db.get(QuizSession, uuid.UUID(r["session_id"]))
        assert s.kind == "test" and s.timing_since is not None
        db.delete(s)


def test_a_test_can_be_longer_than_a_quiz(client, questions):
    r = client.post("/quiz", json={"kind": "test", "count": 50})
    assert r.status_code == 200
    assert client.post("/quiz", json={"kind": "test", "count": 51}).status_code == 422
    with session_scope() as db:
        db.delete(db.get(QuizSession, uuid.UUID(r.json()["session_id"])))


def test_unknown_sessions_are_refused(client, questions):
    for action in ("leave", "end", "resume"):
        assert client.post(f"/quiz/sessions/{uuid.uuid4()}/{action}").status_code == 404
