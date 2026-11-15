"""Quiz sessions, through the real API against a real database.

What these pin down is what the answer endpoint used to accept without
complaint: the same question twice, a question from no quiz, an option from
another question. Each of those recorded something the student never did.

Only ever run against a local database (see conftest): they create and
delete rows, and one test drops a column on purpose.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, inspect, text

from apps.api.curriculum import Concept
from apps.api.db import Base, engine, init_db, session_scope
from apps.api.main import app
from apps.api.quiz import (
    Question,
    QuestionKind,
    QuestionOption,
    QuestionStatus,
    QuizSession,
)
from apps.api.services import session_service
from apps.api.student import LOCAL_STUDENT_ID, active_student_id

pytestmark = pytest.mark.integration

SLUG = f"session-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def client():
    init_db()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def bank(client):
    """Three questions under one throwaway concept, plus one under another
    concept that no quiz in these tests will hand out."""
    with session_scope() as db:
        concept = Concept(slug=SLUG, name="Session test concept")
        other = Concept(slug=SLUG + "-other", name="Other concept")
        db.add_all([concept, other])
        db.flush()
        for i, c in enumerate([concept, concept, concept, other]):
            q = Question(
                concept_id=c.id,
                kind=QuestionKind.multiple_choice,
                stem=f"Question {i}?",
                explanation="Because.",
                status=QuestionStatus.approved,
            )
            q.options = [
                QuestionOption(position=0, text="right", is_correct=True),
                QuestionOption(position=1, text="wrong", is_correct=False,
                               misconception="a named mistake"),
            ]
            db.add(q)
        concept_ids = [concept.id, other.id]

    yield

    with session_scope() as db:
        # Sessions first: deleting the concepts cascades to the questions,
        # their attempts and session rows, but not to the sessions.
        db.execute(
            delete(QuizSession).where(
                QuizSession.id.in_(
                    text(
                        "SELECT sq.session_id FROM quiz_session_questions sq "
                        "JOIN questions q ON q.id = sq.question_id "
                        "WHERE q.concept_id = ANY(:ids)"
                    ).bindparams(ids=concept_ids)
                )
            )
        )
        db.execute(delete(Concept).where(Concept.id.in_(concept_ids)))


def _start(client, count=3):
    r = client.post("/quiz", json={"concept_slugs": [SLUG], "count": count})
    assert r.status_code == 200, r.text
    return r.json()


def _answer(client, session_id, question, correct=True):
    option = _option(question, correct)
    return client.post(
        "/quiz/answer",
        json={
            "session_id": session_id,
            "question_id": question["id"],
            "selected_option_id": option,
        },
    )


def _option(question, correct):
    with session_scope() as db:
        q = db.get(Question, uuid.UUID(question["id"]))
        return str(next(o.id for o in q.options if o.is_correct == correct))


def test_starting_a_quiz_returns_a_session(client, bank):
    data = _start(client)
    assert data["session_id"]
    assert data["count"] == 3


def test_the_session_survives_the_request(client, bank):
    """get_db does not commit; a session that only existed inside the
    request would make every answer fail with 'No such quiz'."""
    data = _start(client)
    with session_scope() as db:
        assert db.get(QuizSession, uuid.UUID(data["session_id"])) is not None


def test_a_full_quiz_is_scored_and_finished(client, bank):
    data = _start(client)
    sid, qs = data["session_id"], data["questions"]

    first = _answer(client, sid, qs[0], correct=True).json()
    assert first["session"] == {"total": 3, "answered": 1, "correct": 1, "finished": False}

    _answer(client, sid, qs[1], correct=False)
    last = _answer(client, sid, qs[2], correct=True).json()
    assert last["session"] == {"total": 3, "answered": 3, "correct": 2, "finished": True}


def test_the_same_question_cannot_be_answered_twice(client, bank):
    """A double click used to count a question twice."""
    data = _start(client)
    sid, q = data["session_id"], data["questions"][0]
    assert _answer(client, sid, q).status_code == 200
    assert _answer(client, sid, q).status_code == 409


def test_a_question_from_outside_the_quiz_is_refused(client, bank):
    data = _start(client)
    with session_scope() as db:
        outsider = db.scalar(
            text(
                "SELECT q.id FROM questions q JOIN concepts c ON c.id = q.concept_id "
                "WHERE c.slug = :s"
            ).bindparams(s=SLUG + "-other")
        )
    r = _answer(client, data["session_id"], {"id": str(outsider)})
    assert r.status_code == 400


def test_an_option_from_another_question_is_refused(client, bank):
    """It used to be graded as a wrong answer the student never gave."""
    data = _start(client)
    sid, qs = data["session_id"], data["questions"]
    r = client.post(
        "/quiz/answer",
        json={
            "session_id": sid,
            "question_id": qs[0]["id"],
            "selected_option_id": _option(qs[1], correct=True),
        },
    )
    assert r.status_code == 400


def test_an_answer_without_a_session_is_refused(client, bank):
    data = _start(client)
    q = data["questions"][0]
    r = client.post(
        "/quiz/answer",
        json={"question_id": q["id"], "selected_option_id": _option(q, True)},
    )
    assert r.status_code == 422


def test_an_unknown_session_is_not_found(client, bank):
    data = _start(client)
    r = _answer(client, str(uuid.uuid4()), data["questions"][0])
    assert r.status_code == 404


def test_another_students_session_is_not_found(client, bank):
    """Test clicks must not be able to land in the real student's quizzes,
    nor the reverse."""
    assert active_student_id() != LOCAL_STUDENT_ID
    with session_scope() as db:
        questions = db.scalars(
            text(
                "SELECT q.id FROM questions q JOIN concepts c ON c.id = q.concept_id "
                "WHERE c.slug = :s"
            ).bindparams(s=SLUG)
        ).all()
        theirs = session_service.start_session(
            db, LOCAL_STUDENT_ID, [db.get(Question, q) for q in questions]
        )
        sid = str(theirs.id)
    r = _answer(client, sid, {"id": str(questions[0])})
    assert r.status_code == 404


def test_history_lists_the_quiz_with_its_score(client, bank):
    data = _start(client, count=2)
    sid, qs = data["session_id"], data["questions"]
    _answer(client, sid, qs[0], correct=True)

    past = client.get("/quiz/sessions").json()
    mine = next(s for s in past if s["id"] == sid)
    assert mine["total"] == 2
    assert mine["answered"] == 1
    assert mine["correct"] == 1
    assert mine["finished"] is False
    assert mine["concepts"] == ["Session test concept"]


def test_init_db_adds_the_session_column_to_an_existing_database(client, bank):
    """create_all never alters an existing table. Recreate a database from
    before sessions -- attempts without session_id -- and check init_db
    brings back the column, its link to quiz_sessions, and the rule that
    stops a question being answered twice."""
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE attempts DROP COLUMN session_id CASCADE"))

    init_db()

    insp = inspect(engine)
    assert "session_id" in {c["name"] for c in insp.get_columns("attempts")}
    assert any(
        fk["referred_table"] == "quiz_sessions" and fk["constrained_columns"] == ["session_id"]
        for fk in insp.get_foreign_keys("attempts")
    )
    assert any(
        ix["name"] == "uq_attempt_session_question" and ix["unique"]
        for ix in insp.get_indexes("attempts")
    )


def test_every_model_column_exists_after_init_db(client):
    """The general form of the rule above: whatever the code expects,
    the database has. A column added to a model without its ALTER fails
    here on a database that already had the table."""
    insp = inspect(engine)
    missing = [
        f"{table.name}.{column.name}"
        for table in Base.metadata.sorted_tables
        for column in table.columns
        if column.name not in {c["name"] for c in insp.get_columns(table.name)}
    ]
    assert missing == []
