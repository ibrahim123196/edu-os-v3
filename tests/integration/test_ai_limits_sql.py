"""Limits on asking the AI, and the record of what each request cost.

A throwaway student with a login and one passage of notes; a stand-in
model answers and reports tokens as Haiku 4.5 would. Local database only.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from ai.generation import usage
from ai.retrieval.rerank import NoOpReranker
from apps.api import auth
from apps.api.ai_limits import AiUsage
from apps.api.config import get_settings, settings
from apps.api.db import init_db, session_scope
from apps.api.main import app
from apps.api.models import Document, DocumentChunk, IngestStatus
from apps.api.services import rag_service
from apps.api.student import Student

pytestmark = pytest.mark.integration

PASSWORD = "correct horse battery"
PASSAGE = "Capacitance is the charge stored per unit potential difference."


def _vector(text: str) -> list[float]:
    v = [0.0] * settings.embedding_dim
    for w in re.findall(r"[a-z]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % len(v)] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class Embedder:
    def embed_query(self, text):
        return _vector(text)


class Model:
    """Answers from the passage, and reports tokens as Haiku 4.5 would."""

    model = "claude-haiku-4-5"

    def complete(self, system, messages, max_tokens=None, temperature=None):
        usage.record(self.model, 1000, 200)
        return "Capacitance is charge per unit potential difference [S1]."


@pytest.fixture
def student(monkeypatch):
    init_db()
    s = get_settings()
    monkeypatch.setattr(s, "require_login", True)
    monkeypatch.setattr(s, "ai_requests_per_hour", 0)
    monkeypatch.setattr(s, "ai_requests_per_day", 0)
    monkeypatch.setattr(s, "ai_daily_budget_usd", 0.0)
    auth.clear_cache()
    auth.reset_failures()

    monkeypatch.setattr(rag_service, "get_llm", lambda: Model())
    monkeypatch.setattr(rag_service, "get_embedding_provider", lambda: Embedder())
    monkeypatch.setattr(rag_service, "get_reranker", lambda: NoOpReranker())
    monkeypatch.setattr(settings, "enable_query_rewriting", False)
    monkeypatch.setattr(settings, "enable_spelling_correction", False)

    tag = uuid.uuid4().hex[:8]
    with session_scope() as db:
        row = Student(id=uuid.uuid4(), display_name=f"Limits {tag}",
                      email=f"limits-{tag}@example.com",
                      password_hash=auth.hash_password(PASSWORD))
        db.add(row)
        db.flush()
        doc = Document(title="limits", filename="l.txt", content_type="text/plain",
                       storage_path="x", byte_size=1, status=IngestStatus.ready,
                       student_id=row.id)
        db.add(doc)
        db.flush()
        db.add(DocumentChunk(document_id=doc.id, chunk_index=0, page_start=1, page_end=1,
                             word_count=9, content=PASSAGE, embedding=_vector(PASSAGE)))
        sid, email = row.id, row.email
    client = TestClient(app)
    assert client.post("/auth/login", json={"email": email, "password": PASSWORD}).status_code == 200
    yield client, sid
    with session_scope() as db:
        db.execute(delete(Student).where(Student.id == sid))
    auth.clear_cache()


def _ask(client):
    return client.post("/chat/ask", json={"question": "What is capacitance?"})


def _rows(sid):
    with session_scope() as db:
        return db.scalars(select(AiUsage).where(AiUsage.student_id == sid)
                          .order_by(AiUsage.created_at)).all()


def test_each_question_is_recorded_with_what_it_cost(student):
    client, sid = student
    assert _ask(client).status_code == 200
    (row,) = _rows(sid)
    assert row.kind == "ask" and row.calls >= 1
    assert row.input_tokens == 1000 * row.calls
    # Haiku 4.5: $1 in, $5 out per million tokens -> $0.002 a call.
    assert row.cost_usd == pytest.approx(0.002 * row.calls)


def test_too_many_in_an_hour_must_wait(student, monkeypatch):
    client, sid = student
    monkeypatch.setattr(get_settings(), "ai_requests_per_hour", 2)
    assert _ask(client).status_code == 200
    assert _ask(client).status_code == 200
    r = _ask(client)
    assert r.status_code == 429
    assert "in the last hour" in r.json()["detail"] and "ask again at" in r.json()["detail"]
    assert len(_rows(sid)) == 2                 # the refused one is not counted


def test_the_day_limit_counts_24_hours(student, monkeypatch):
    client, sid = student
    monkeypatch.setattr(get_settings(), "ai_requests_per_day", 1)
    with session_scope() as db:                 # one from 20 hours ago still counts
        db.add(AiUsage(student_id=sid, kind="ask",
                       created_at=datetime.now(timezone.utc) - timedelta(hours=20)))
    r = _ask(client)
    assert r.status_code == 429 and "24 hours" in r.json()["detail"]


def test_an_old_request_no_longer_counts(student, monkeypatch):
    client, sid = student
    monkeypatch.setattr(get_settings(), "ai_requests_per_hour", 1)
    with session_scope() as db:
        db.add(AiUsage(student_id=sid, kind="ask",
                       created_at=datetime.now(timezone.utc) - timedelta(minutes=61)))
    assert _ask(client).status_code == 200


def test_the_daily_budget_stops_everyone(student, monkeypatch):
    client, sid = student
    monkeypatch.setattr(get_settings(), "ai_daily_budget_usd", 1.0)
    with session_scope() as db:
        db.add(AiUsage(student_id=sid, kind="ask", cost_usd=1.5))
    r = _ask(client)
    assert r.status_code == 429 and "after midnight" in r.json()["detail"]


def test_lesson_and_work_it_out_replies_count_too(student, monkeypatch):
    client, sid = student
    monkeypatch.setattr(get_settings(), "ai_requests_per_hour", 1)
    assert _ask(client).status_code == 200
    fake = uuid.uuid4()
    assert client.post(f"/chat/socratic/{fake}/reply", json={"reply": "x"}).status_code == 429
    assert client.post(f"/chat/lessons/{fake}/feedback", json={"reply": "x"}).status_code == 429


def test_quizzes_and_pages_are_not_limited(student, monkeypatch):
    client, sid = student
    monkeypatch.setattr(get_settings(), "ai_requests_per_hour", 1)
    assert _ask(client).status_code == 200
    for path in ("/progress", "/plan", "/documents", "/quiz/concepts", "/chat/conversations"):
        assert client.get(path).status_code == 200, path


def test_a_script_calling_the_model_records_nothing(student):
    client, sid = student
    usage.record("claude-haiku-4-5", 10, 10)     # no request, so no row to add to
    assert _rows(sid) == []
