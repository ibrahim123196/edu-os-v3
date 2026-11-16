"""Follow-up questions through /chat/ask, against a real database.

The model, embedder and reranker are stand-ins: what is tested is what the
pipeline sends them and what comes back to the page. Local database only
(see conftest).
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from ai.generation.llm import LLMError
from ai.retrieval.rerank import NoOpReranker
from apps.api.config import settings
from apps.api.db import init_db, session_scope
from apps.api.main import app
from apps.api.models import Conversation, Document, DocumentChunk, IngestStatus
from apps.api.services import rag_service

pytestmark = pytest.mark.integration

REWRITE = "explain more simply why the drift velocity of electrons is small"


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
    """Records every request; answers by what kind of request it is."""

    model = "fake"

    def __init__(self):
        self.answers: list[list[dict]] = []
        self.rewrites = 0
        self.down = False

    def complete(self, system, messages, max_tokens=None, temperature=None):
        if system.startswith("You rewrite a student's follow-up"):
            self.rewrites += 1
            return REWRITE
        if self.down:
            raise LLMError("The model is unavailable.")
        self.answers.append(messages)
        return "Electrons drift slowly because they keep colliding [S1]."


@pytest.fixture
def chat(monkeypatch):
    init_db()
    passage = "The drift velocity of electrons is small because electrons collide with ions."
    with session_scope() as db:
        doc = Document(title="conv test", filename="c.txt", content_type="text/plain",
                       storage_path="x", byte_size=1, status=IngestStatus.ready)
        db.add(doc)
        db.flush()
        db.add(DocumentChunk(document_id=doc.id, chunk_index=0, page_start=6, page_end=6,
                             word_count=14, content=passage, embedding=_vector(passage)))
        doc_id = doc.id

    model = Model()
    monkeypatch.setattr(rag_service, "get_llm", lambda: model)
    monkeypatch.setattr(rag_service, "get_embedding_provider", lambda: Embedder())
    monkeypatch.setattr(rag_service, "get_reranker", lambda: NoOpReranker())
    # Keyword rewriting and typo fixing are separate features with their own
    # tests; switched off so the only rewrite here is the follow-up one.
    monkeypatch.setattr(settings, "enable_query_rewriting", False)
    monkeypatch.setattr(settings, "enable_spelling_correction", False)
    conversations: list = []

    with TestClient(app) as client:
        def ask(question, conversation_id=None):
            r = client.post("/chat/ask", json={
                "question": question, "document_ids": [str(doc_id)],
                "conversation_id": conversation_id, "include_trace": True})
            if r.status_code == 200:
                conversations.append(r.json()["conversation_id"])
            return r
        yield ask, model

    with session_scope() as db:
        db.execute(delete(Document).where(Document.id == doc_id))
        if conversations:
            db.execute(delete(Conversation).where(
                Conversation.id.in_({uuid.UUID(c) for c in conversations})))


def test_a_first_question_works_as_before(chat):
    ask, model = chat
    body = ask("what is drift velocity").json()
    assert body["grounded"] and body["searched_as"] is None
    assert model.rewrites == 0                       # no extra call, no extra cost
    assert len(model.answers[0]) == 1                # the single old-style request
    assert "follow_up" not in body["trace"]["stages"]


def test_a_follow_up_is_searched_as_the_full_question(chat):
    ask, model = chat
    first = ask("what is drift velocity").json()
    body = ask("explain that more simply", first["conversation_id"]).json()

    assert body["conversation_id"] == first["conversation_id"]
    assert body["searched_as"] == REWRITE
    assert body["trace"]["stages"]["query_preparation"]["original"] == REWRITE
    assert body["grounded"]


def test_a_follow_up_answer_sees_the_earlier_exchange(chat):
    ask, model = chat
    first = ask("what is drift velocity").json()
    ask("explain that more simply", first["conversation_id"])

    sent = model.answers[-1]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[0]["content"] == "what is drift velocity"
    # The earlier answer without its old marker: [S1] means a new passage now.
    assert sent[1]["content"] == "Electrons drift slowly because they keep colliding."
    assert "Student's question: explain that more simply" in sent[2]["content"]


def test_a_failed_answer_does_not_break_the_conversation(chat):
    """The unanswered question is stored, but must not become a turn: two
    student turns in a row would make every later request fail."""
    ask, model = chat
    first = ask("what is drift velocity").json()
    model.down = True
    assert ask("and mobility?", first["conversation_id"]).status_code == 503
    model.down = False

    assert ask("explain that more simply", first["conversation_id"]).status_code == 200
    roles = [m["role"] for m in model.answers[-1]]
    assert roles == ["user", "assistant", "user"]
    assert "and mobility?" not in str(model.answers[-1][:-1])


def test_an_unknown_conversation_is_refused(chat):
    ask, _ = chat
    assert ask("why", str(uuid.uuid4())).status_code == 404
