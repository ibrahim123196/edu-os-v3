"""Work it out, through /chat/ask and the reply endpoint.

Stand-in model, embedder and reranker, as in test_lesson_sql. Local only.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from ai.retrieval.rerank import NoOpReranker
from apps.api.config import settings
from apps.api.db import init_db, session_scope
from apps.api.main import app
from apps.api.models import Conversation, Document, DocumentChunk, IngestStatus, Message
from apps.api.services import rag_service
from tests.integration.test_conversation_sql import Embedder, _vector

pytestmark = pytest.mark.integration

TAG = "t" + uuid.uuid4().hex[:5]
PASSAGE = (f"Conductium {TAG}: free electrons move opposite to the field and gather on the "
           "surface, where their own field cancels the applied field inside.")

PLAN = f"""[GOAL]
Inside conductium {TAG} the field is zero: the electrons' own field cancels it [S1].
[STEP]
Q: What do the free electrons do in a field?
POINT: They move opposite to the field [S1].
HINT: Think about the force on a charge.
[STEP]
Q: Where do they gather?
POINT: On the surface [S1].
HINT: They cannot leave."""


class Model:
    model = "fake"

    def __init__(self):
        self.requests: list[str] = []
        self.plan = PLAN
        self.verdicts = ["right"]

    def complete(self, system, messages, max_tokens=None, temperature=None):
        content = messages[-1]["content"]
        self.requests.append(content)
        if "They answered:" in content:
            verdict = self.verdicts.pop(0) if self.verdicts else "right"
            return f"VERDICT: {verdict}\nFEEDBACK: Feedback for {verdict}."
        return self.plan


@pytest.fixture
def work(monkeypatch):
    init_db()
    with session_scope() as db:
        doc = Document(title=f"socratic {TAG}", filename="s.txt", content_type="text/plain",
                       storage_path="x", byte_size=1, status=IngestStatus.ready)
        db.add(doc)
        db.flush()
        db.add(DocumentChunk(document_id=doc.id, chunk_index=0, page_start=46, page_end=46,
                             word_count=20, content=PASSAGE, embedding=_vector(PASSAGE)))
        doc_id = doc.id

    model = Model()
    monkeypatch.setattr(rag_service, "get_llm", lambda: model)
    monkeypatch.setattr(rag_service, "get_embedding_provider", lambda: Embedder())
    monkeypatch.setattr(rag_service, "get_reranker", lambda: NoOpReranker())
    monkeypatch.setattr(settings, "enable_query_rewriting", False)
    monkeypatch.setattr(settings, "enable_spelling_correction", False)
    conversations: list = []

    with TestClient(app) as client:
        def start(question=f"why is the field inside conductium {TAG} zero?", **extra):
            r = client.post("/chat/ask", json={"question": question, "socratic": True,
                                               "document_ids": [str(doc_id)], **extra})
            assert r.status_code == 200, r.text
            conversations.append(uuid.UUID(r.json()["conversation_id"]))
            return r.json()

        def reply(message_id, text="They move."):
            return client.post(f"/chat/socratic/{message_id}/reply", json={"reply": text})
        yield start, reply, model, client

    with session_scope() as db:
        db.execute(delete(Conversation).where(Conversation.id.in_(conversations)))
        db.execute(delete(Document).where(Document.id == doc_id))


def test_it_starts_with_a_question_not_the_answer(work):
    start, _, model, _ = work
    body = start()
    assert body["mode"] == "socratic"
    assert body["answer"].endswith("What do the free electrons do in a field?")
    assert "[GOAL]" not in body["answer"] and "cancels" not in body["answer"]   # plan hidden
    assert body["socratic"] == {"step": 1, "total": 2, "done": False, "event": "start",
                                "message_id": None}
    assert body["citations"] and body["grounded"]
    assert "Do not answer it" in model.requests[-1]


def test_right_answers_lead_to_the_end(work):
    start, reply, model, _ = work
    first = start()
    r = reply(first["message_id"])
    assert r.status_code == 200
    turn = r.json()
    assert turn["verdict"] == "right" and turn["socratic"]["step"] == 2
    assert turn["text"] == "Feedback for right.\n\nWhere do they gather?"
    asked = model.requests[-1]
    assert PASSAGE in asked and "They answered: They move." in asked      # same passages

    model.verdicts = ["right"]
    end = reply(turn["socratic"]["message_id"], "The surface.").json()
    assert end["socratic"]["done"]
    assert end["text"].endswith(f"Inside conductium {TAG} the field is zero: the electrons' own "
                                "field cancels it [S1].")


def test_wrong_tries_hint_then_show_the_step(work):
    start, reply, model, _ = work
    model.verdicts = ["wrong", "wrong", "wrong"]
    mid = start()["message_id"]
    t1 = reply(mid, "No idea").json()
    assert t1["socratic"]["event"] == "again" and t1["socratic"]["step"] == 1
    t2 = reply(t1["socratic"]["message_id"], "Stay still").json()
    assert t2["socratic"]["event"] == "hint" and "Hint: Think about the force" in t2["text"]
    t3 = reply(t2["socratic"]["message_id"], "Vanish").json()
    assert t3["socratic"]["event"] == "shown" and t3["socratic"]["step"] == 2
    assert "Here's this step: They move opposite to the field [S1]." in t3["text"]


def test_the_conversation_keeps_every_turn_in_order(work):
    start, reply, _, _ = work
    body = start()
    reply(body["message_id"])
    with session_scope() as db:
        roles = [m.role for m in db.scalars(
            select(Message).where(Message.conversation_id == uuid.UUID(body["conversation_id"]))
            .order_by(Message.created_at))]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_only_the_latest_question_can_be_answered(work):
    """Answering an older turn again would fork the chain."""
    start, reply, _, _ = work
    first = start()
    reply(first["message_id"])
    assert reply(first["message_id"]).status_code == 409


def test_a_finished_one_cannot_be_answered(work):
    start, reply, _, _ = work
    first = start()
    t = reply(first["message_id"]).json()
    end = reply(t["socratic"]["message_id"]).json()
    assert reply(end["socratic"]["message_id"]).status_code == 409


def test_replying_to_something_else_is_refused(work):
    _, reply, _, client = work
    assert reply(uuid.uuid4()).status_code == 404
    assert reply(uuid.uuid4(), "").status_code == 422


def test_a_plan_not_in_form_is_not_used(work):
    """Never a half plan: the student is told it is not covered."""
    start, _, model, _ = work
    model.plan = f"Inside conductium {TAG} the field is zero [S1]."
    body = start()
    assert body["socratic"] is None and not body["grounded"]
    assert "cancels" not in body["answer"]


def test_a_plan_without_sources_is_not_used(work):
    start, _, model, _ = work
    model.plan = PLAN.replace(" [S1]", "")
    body = start()
    assert body["socratic"] is None and not body["grounded"]


def test_lesson_and_work_it_out_together_is_refused(work):
    _, _, _, client = work
    r = client.post("/chat/ask", json={"question": "x", "socratic": True, "lesson": True})
    assert r.status_code == 422


def test_quiz_words_still_work_it_out(work):
    """The section decides, not the words typed in it."""
    start, _, _, _ = work
    body = start(f"quiz me on conductium {TAG}")
    assert body["mode"] == "socratic" and body["socratic"]
