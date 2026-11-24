"""Logging in, and one student never seeing another's things.

Two throwaway students, A and B, each with a login. A has an upload and a
chat; every way B could reach them is tried and must fail as if they did
not exist. Local database only (see conftest).
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from ai.retrieval.hybrid import RetrievalFilters, hybrid_search
from apps.api import auth
from apps.api.config import get_settings, settings
from apps.api.db import init_db, session_scope
from apps.api.main import app
from apps.api.models import Conversation, Document, DocumentChunk, IngestStatus, Message
from apps.api.student import Student

pytestmark = pytest.mark.integration

TAG = uuid.uuid4().hex[:8]
PASSWORD = "correct horse battery"


def _vector(text: str) -> list[float]:
    v = [0.0] * settings.embedding_dim
    for w in re.findall(r"[a-z]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % len(v)] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


@pytest.fixture
def world(monkeypatch):
    init_db()
    monkeypatch.setattr(get_settings(), "require_login", True)
    auth.clear_cache()
    auth.reset_failures()
    ids = {}
    with session_scope() as db:
        for who in ("a", "b"):
            s = Student(id=uuid.uuid4(), display_name=f"Student {who.upper()} {TAG}",
                        email=f"{who}-{TAG}@example.com",
                        password_hash=auth.hash_password(PASSWORD))
            db.add(s)
            db.flush()
            ids[who] = s.id
        passage = f"zorblax {TAG} notes on capacitors written by student A"
        doc = Document(title=f"A's notes {TAG}", filename="a.txt", content_type="text/plain",
                       storage_path="nowhere.txt", byte_size=1, status=IngestStatus.ready,
                       student_id=ids["a"])
        db.add(doc)
        db.flush()
        db.add(DocumentChunk(document_id=doc.id, chunk_index=0, word_count=8,
                             content=passage, embedding=_vector(passage)))
        conv = Conversation(title=f"A's chat {TAG}", student_id=ids["a"])
        db.add(conv)
        db.flush()
        msg = Message(conversation_id=conv.id, role="assistant", content="hi",
                      retrieval_trace={"lesson_context": {"chunk_ids": [], "check": "x"},
                                       "socratic_state": {"steps": [{}], "step": 0}})
        db.add(msg)
        db.flush()
        ids.update(doc=doc.id, conv=conv.id, msg=msg.id)

    yield ids

    with session_scope() as db:
        db.execute(delete(Student).where(
            Student.id.in_([ids["a"], ids["b"]]) | Student.email.like(f"%-{TAG}@example.com")))
    auth.clear_cache()
    auth.reset_failures()


def _client(who: str | None = None) -> TestClient:
    client = TestClient(app)
    if who:
        r = client.post("/auth/login", json={"email": f"{who}-{TAG}@example.com",
                                             "password": PASSWORD})
        assert r.status_code == 200, r.text
    return client


# --- logging in -----------------------------------------------------------


def test_nothing_but_the_page_and_login_without_a_login(world):
    c = _client()
    assert c.get("/progress").status_code == 401
    assert c.get("/documents").status_code == 401
    assert c.get("/plan").status_code == 401
    assert c.post("/chat/ask", json={"question": "hi"}).status_code == 401
    assert c.get("/health").status_code == 200
    assert c.get("/").status_code == 200
    assert c.get("/auth/me").json()["logged_in"] is False


def test_logging_in_and_out(world):
    c = _client("a")
    me = c.get("/auth/me").json()
    assert me["logged_in"] and me["id"] == str(world["a"]) and not me["is_test"]
    assert c.get("/progress").status_code == 200   # a sync route sees the student
    c.post("/auth/logout")
    assert c.get("/progress").status_code == 401
    assert c.get("/auth/me").json()["logged_in"] is False


def test_the_cookie_is_httponly(world):
    r = TestClient(app).post("/auth/login", json={"email": f"a-{TAG}@example.com",
                                                   "password": PASSWORD})
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie


def test_email_is_not_case_sensitive(world):
    r = TestClient(app).post("/auth/login", json={"email": f"  A-{TAG}@Example.COM ",
                                                   "password": PASSWORD})
    assert r.status_code == 200


def test_a_wrong_password_and_an_unknown_email_look_the_same(world):
    c = TestClient(app)
    wrong = c.post("/auth/login", json={"email": f"a-{TAG}@example.com", "password": "nope"})
    unknown = c.post("/auth/login", json={"email": f"z-{TAG}@example.com", "password": "nope"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_too_many_wrong_tries_wait(world):
    c = TestClient(app)
    for _ in range(5):
        c.post("/auth/login", json={"email": f"b-{TAG}@example.com", "password": "nope"})
    r = c.post("/auth/login", json={"email": f"b-{TAG}@example.com", "password": PASSWORD})
    assert r.status_code == 429


def test_a_made_up_cookie_is_no_login(world):
    c = TestClient(app)
    c.cookies.set(auth.COOKIE, "made-up")
    assert c.get("/progress").status_code == 401


def test_a_reset_logs_out_everywhere(world):
    c = _client("a")
    with session_scope() as db:
        assert auth.end_all_sessions(db, world["a"]) >= 1
    assert c.get("/progress").status_code == 401


def test_only_a_hash_of_the_token_is_stored(world):
    c = _client("a")
    token = c.cookies.get(auth.COOKIE)
    with session_scope() as db:
        rows = db.query(auth.LoginSession).filter_by(student_id=world["a"]).all()
    assert rows and all(r.token_hash != token for r in rows)


# --- keeping students apart -----------------------------------------------


def test_a_sees_their_own_upload_and_b_does_not(world):
    a, b = _client("a"), _client("b")
    doc = str(world["doc"])
    assert doc in {d["id"] for d in a.get("/documents").json()}
    assert doc not in {d["id"] for d in b.get("/documents").json()}
    assert a.get(f"/documents/{doc}").status_code == 200
    assert b.get(f"/documents/{doc}").status_code == 404
    assert b.get(f"/documents/{doc}/chunks").status_code == 404
    assert b.post(f"/documents/{doc}/reingest").status_code == 404
    assert b.delete(f"/documents/{doc}").status_code == 404
    assert b.get(f"/figures/{doc}/1.1.png").status_code == 404
    with session_scope() as db:
        assert db.get(Document, world["doc"]) is not None   # B's delete did nothing


def test_b_cannot_open_list_or_continue_a_chat(world):
    b = _client("b")
    conv, msg = str(world["conv"]), str(world["msg"])
    assert b.get(f"/chat/conversations/{conv}").status_code == 404
    assert conv not in {c["id"] for c in b.get("/chat/conversations").json()}
    r = b.post("/chat/ask", json={"question": "what did A ask?", "conversation_id": conv})
    assert r.status_code == 404
    assert b.post(f"/chat/lessons/{msg}/feedback", json={"reply": "x"}).status_code == 404
    assert b.post(f"/chat/socratic/{msg}/reply", json={"reply": "x"}).status_code == 404
    a = _client("a")
    assert a.get(f"/chat/conversations/{conv}").status_code == 200
    assert conv in {c["id"] for c in a.get("/chat/conversations").json()}


def test_search_finds_only_the_students_own_uploads(world):
    word = f"zorblax {TAG}"
    with session_scope() as db:
        def found(student_id):
            hits = hybrid_search(db, query=word, query_embedding=_vector(word),
                                 filters=RetrievalFilters(student_id=student_id))
            return any(h.document_id == world["doc"] for h in hits)

        assert found(world["a"])
        assert not found(world["b"])


def test_uploads_and_chats_made_in_a_request_belong_to_the_student(world):
    """The model defaults: something made without naming its owner is the
    logged-in student's, not the stand-in's."""
    from apps.api.student import _current

    token = _current.set(world["b"])
    try:
        with session_scope() as db:
            doc = Document(title="b", filename="b.txt", content_type="text/plain",
                           storage_path="x", byte_size=1)
            conv = Conversation(title="b")
            db.add_all([doc, conv])
            db.flush()
            assert doc.student_id == world["b"] and conv.student_id == world["b"]
            platform = Document(title="p", filename="p.pdf", content_type="application/pdf",
                                storage_path="x", byte_size=1, source_kind="platform")
            db.add(platform)
            db.flush()
            assert platform.student_id is None
            db.rollback()
    finally:
        _current.reset(token)


# --- signing up -----------------------------------------------------------


def _signup(c, who, **over):
    body = {"name": f"New {who}", "email": f"{who}-{TAG}@example.com", "password": PASSWORD}
    body.update(over)
    return c.post("/auth/signup", json=body)


def test_a_student_can_make_an_account_and_is_logged_in(world):
    c = TestClient(app)
    assert c.get("/auth/me").json()["signup"] is True
    r = _signup(c, "new")
    assert r.status_code == 200, r.text
    assert c.get("/auth/me").json()["name"] == "New new"
    assert c.get("/progress").json()["questions_answered"] == 0
    assert c.get("/documents").json() == []          # sees nobody else's uploads
    c.post("/auth/logout")
    assert _client("new").get("/progress").status_code == 200   # can log back in


def test_sign_up_refuses_a_taken_email_and_bad_input(world):
    c = TestClient(app)
    assert _signup(c, "a").status_code == 409                       # A already has it
    assert _signup(c, "x", email="not-an-email").status_code == 400
    assert _signup(c, "x", password="short").status_code == 400
    assert _signup(c, "x", name="   ").status_code == 400


def test_sign_up_can_be_switched_off(world, monkeypatch):
    monkeypatch.setattr(get_settings(), "open_signup", False)
    c = TestClient(app)
    assert c.get("/auth/me").json()["signup"] is False
    assert _signup(c, "off").status_code == 403


def test_only_a_few_accounts_from_one_address(world):
    c = TestClient(app)
    for i in range(5):
        assert _signup(c, f"flood{i}").status_code == 200
        c.post("/auth/logout")
    assert _signup(c, "flood5").status_code == 429


# --- deleting an account --------------------------------------------------


def test_deleting_a_student_removes_everything_that_is_theirs(world, tmp_path):
    import argparse

    from scripts.accounts import cmd_delete

    upload = tmp_path / "a.txt"
    upload.write_text("notes")
    with session_scope() as db:
        db.get(Document, world["doc"]).storage_path = str(upload)
    _client("a")                                   # a login to be removed too
    args = argparse.Namespace(email=f"a-{TAG}@example.com")

    with session_scope() as db, pytest.raises(ValueError):
        cmd_delete(db, args, confirm=lambda _: "wrong@example.com")
    with session_scope() as db:
        assert db.get(Student, world["a"]) is not None   # a wrong email keeps it

    with session_scope() as db:
        cmd_delete(db, args, confirm=lambda _: f"A-{TAG}@example.com ")
    with session_scope() as db:
        assert db.get(Student, world["a"]) is None
        assert db.get(Document, world["doc"]) is None
        assert db.get(Conversation, world["conv"]) is None
        assert db.query(auth.LoginSession).filter_by(student_id=world["a"]).count() == 0
        assert db.get(Student, world["b"]) is not None    # B untouched
    assert not upload.exists()


def test_the_stand_in_students_cannot_be_deleted(world):
    import argparse

    from apps.api.student import LOCAL_STUDENT_ID
    from scripts.accounts import cmd_delete

    with session_scope() as db:
        local = db.get(Student, LOCAL_STUDENT_ID)
        had = local.email
        local.email = f"local-{TAG}@example.com"
    try:
        with session_scope() as db, pytest.raises(ValueError):
            cmd_delete(db, argparse.Namespace(email=f"local-{TAG}@example.com"),
                       confirm=lambda _: f"local-{TAG}@example.com")
    finally:
        with session_scope() as db:
            db.get(Student, LOCAL_STUDENT_ID).email = had
