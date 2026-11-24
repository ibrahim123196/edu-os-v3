"""Reporting a tutor answer, and reviewing the reports. Local database only."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from apps.api import auth
from apps.api.answer_reports import AnswerReport
from apps.api.config import get_settings
from apps.api.db import init_db, session_scope
from apps.api.main import app
from apps.api.models import Conversation, Message
from apps.api.student import Student

pytestmark = pytest.mark.integration

PASSWORD = "correct horse battery"


@pytest.fixture
def world(monkeypatch):
    init_db()
    monkeypatch.setattr(get_settings(), "require_login", True)
    auth.clear_cache()
    auth.reset_failures()
    tag = uuid.uuid4().hex[:8]
    ids = {"tag": tag}
    with session_scope() as db:
        for who in ("a", "b"):
            s = Student(id=uuid.uuid4(), display_name=f"Rep {who} {tag}",
                        email=f"rep-{who}-{tag}@example.com",
                        password_hash=auth.hash_password(PASSWORD))
            db.add(s)
            db.flush()
            ids[who] = s.id
        conv = Conversation(title="emf", student_id=ids["a"])
        db.add(conv)
        db.flush()
        q = Message(conversation_id=conv.id, role="user", content=f"is emf only in cells {tag}")
        db.add(q)
        db.flush()
        ans = Message(conversation_id=conv.id, role="assistant",
                      content="emf is defined specifically for cells [S1].",
                      citations=[{"document_title": "NCERT Physics", "page_label": "p. 13"}])
        db.add(ans)
        db.flush()
        ids.update(question=q.id, answer=ans.id)
    yield ids
    with session_scope() as db:
        db.execute(delete(Student).where(Student.id.in_([ids["a"], ids["b"]])))
    auth.clear_cache()


def _client(ids, who):
    c = TestClient(app)
    r = c.post("/auth/login", json={"email": f"rep-{who}-{ids['tag']}@example.com",
                                    "password": PASSWORD})
    assert r.status_code == 200
    return c


def _reports(ids):
    with session_scope() as db:
        return db.scalars(select(AnswerReport).where(AnswerReport.message_id == ids["answer"])).all()


def test_the_reasons_are_offered(world):
    reasons = _client(world, "a").get("/chat/report-reasons").json()
    assert {"key": "not_in_book", "label": "It says something the textbook doesn't"} in reasons


def test_a_student_reports_their_answer(world):
    c = _client(world, "a")
    r = c.post(f"/chat/messages/{world['answer']}/report",
               json={"reason": "not_in_book", "note": "generators have emf too",
                     "part": "Step 3: Worked example"})
    assert r.status_code == 200
    (rep,) = _reports(world)
    assert (rep.reason, rep.note, rep.part, rep.resolved_at) == (
        "not_in_book", "generators have emf too", "Step 3: Worked example", None)


def test_reporting_again_replaces_and_reopens(world):
    c = _client(world, "a")
    url = f"/chat/messages/{world['answer']}/report"
    c.post(url, json={"reason": "confusing"})
    with session_scope() as db:
        rep = db.scalar(select(AnswerReport).where(AnswerReport.message_id == world["answer"]))
        rep.outcome = "fine"
        from datetime import datetime, timezone
        rep.resolved_at = datetime.now(timezone.utc)
    c.post(url, json={"reason": "wrong_fact"})
    (rep,) = _reports(world)
    assert rep.reason == "wrong_fact" and rep.resolved_at is None and rep.outcome is None


def test_another_student_cannot_report_it(world):
    r = _client(world, "b").post(f"/chat/messages/{world['answer']}/report",
                                 json={"reason": "other"})
    assert r.status_code == 404 and _reports(world) == []


def test_only_answers_can_be_reported_with_a_known_reason(world):
    c = _client(world, "a")
    assert c.post(f"/chat/messages/{world['question']}/report",
                  json={"reason": "other"}).status_code == 404
    assert c.post(f"/chat/messages/{world['answer']}/report",
                  json={"reason": "made-up"}).status_code == 422


def test_reporting_does_not_count_against_ai_limits(world, monkeypatch):
    monkeypatch.setattr(get_settings(), "ai_requests_per_hour", 1)
    c = _client(world, "a")
    for reason in ("other", "confusing", "wrong_fact"):
        assert c.post(f"/chat/messages/{world['answer']}/report",
                      json={"reason": reason}).status_code == 200


def test_the_review_lists_and_closes_reports(world, capsys):
    from scripts import review_answers

    _client(world, "a").post(f"/chat/messages/{world['answer']}/report",
                             json={"reason": "not_in_book", "note": "generators too"})
    assert review_answers.main([]) == 0
    out = capsys.readouterr().out
    assert f"is emf only in cells {world['tag']}" in out
    assert "defined specifically for cells" in out
    assert "NCERT Physics p. 13" in out
    assert "It says something the textbook doesn't -- generators too" in out

    # Its number in the list, whatever other open reports exist.
    numbers = [line.split(".")[0] for line in out.splitlines() if f"Rep a {world['tag']}" in line]
    n = numbers[0]
    assert review_answers.main(["--fixed", n, "--note", "prompt now says so"]) == 0
    (rep,) = _reports(world)
    assert rep.outcome == "fixed" and rep.review_note == "prompt now says so"
    assert rep.resolved_at is not None
