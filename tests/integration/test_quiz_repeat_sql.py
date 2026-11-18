"""A reattempt asks the very same questions: POST /quiz with repeat_of.
Local database only (see conftest)."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from apps.api.curriculum import Concept
from apps.api.db import init_db, session_scope
from apps.api.main import app
from apps.api.quiz import Question, QuestionKind, QuestionOption, QuestionStatus
from apps.api.routes import quiz as quiz_route
from apps.api.student import Student

pytestmark = pytest.mark.integration

TAG = "t" + uuid.uuid4().hex[:5]


@pytest.fixture
def bank(monkeypatch):
    init_db()
    with session_scope() as db:
        me, other = Student(id=uuid.uuid4(), display_name=f"r {TAG}"), Student(id=uuid.uuid4(), display_name=f"o {TAG}")
        concept = Concept(slug=f"repeat-{TAG}", name=f"Repeatium {TAG}")
        db.add_all([me, other, concept])
        db.flush()
        for k in range(8):
            q = Question(concept_id=concept.id, kind=QuestionKind.multiple_choice,
                         stem=f"Repeat {TAG} {k}?", explanation="x", status=QuestionStatus.approved)
            q.options = [QuestionOption(position=0, text="A", is_correct=True),
                         QuestionOption(position=1, text="B", is_correct=False)]
            db.add(q)
        ids = {"me": me.id, "other": other.id, "concept": concept.id}
    who = {"id": ids["me"]}
    monkeypatch.setattr(quiz_route, "active_student_id", lambda: who["id"])
    with TestClient(app) as client:
        yield client, ids, who
    with session_scope() as db:
        db.execute(delete(Question).where(Question.concept_id == ids["concept"]))
        db.execute(delete(Concept).where(Concept.id == ids["concept"]))
        db.execute(delete(Student).where(Student.id.in_([ids["me"], ids["other"]])))


def _start(client, **body):
    r = client.post("/quiz", json={"concept_slugs": [], "count": 5, **body})
    assert r.status_code == 200, r.text
    return r.json()


def test_a_reattempt_asks_the_same_questions_in_the_same_order(bank):
    client, _, _ = bank
    first = _start(client, concept_slugs=[f"repeat-{TAG}"])
    again = _start(client, repeat_of=first["session_id"])
    assert [q["id"] for q in again["questions"]] == [q["id"] for q in first["questions"]]
    assert again["session_id"] != first["session_id"]                 # a new quiz, a new score


def test_a_question_retired_since_is_left_out(bank):
    client, _, _ = bank
    first = _start(client, concept_slugs=[f"repeat-{TAG}"])
    gone = first["questions"][1]["id"]
    with session_scope() as db:
        db.get(Question, uuid.UUID(gone)).faulty = True
    again = _start(client, repeat_of=first["session_id"])
    assert gone not in [q["id"] for q in again["questions"]] and again["count"] == 4


def test_someone_elses_quiz_cannot_be_repeated(bank):
    client, ids, who = bank
    first = _start(client, concept_slugs=[f"repeat-{TAG}"])
    who["id"] = ids["other"]
    r = client.post("/quiz", json={"repeat_of": first["session_id"]})
    assert r.status_code == 404
    assert client.post("/quiz", json={"repeat_of": str(uuid.uuid4())}).status_code == 404
