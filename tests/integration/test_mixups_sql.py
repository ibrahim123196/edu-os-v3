"""Spotting a known mistake in a question, through /chat/ask.

Stand-in model, embedder and reranker, as in test_conversation_sql: what is
tested is what the request offers the model, and what the page gets back
from its reply. Local database only (see conftest).
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, insert

from ai.retrieval.rerank import NoOpReranker
from apps.api.config import settings
from apps.api.curriculum import Concept, chunk_concepts
from apps.api.db import init_db, session_scope
from apps.api.main import app
from apps.api.models import Conversation, Document, DocumentChunk, IngestStatus
from apps.api.quiz import Misconception
from apps.api.services import rag_service
from tests.integration.test_conversation_sql import Embedder, _vector

pytestmark = pytest.mark.integration

TAG = "t" + uuid.uuid4().hex[:5]
SLUG = f"capa-{TAG}/depends-on-v"
LABEL = "Thinks capacitance depends on the voltage applied"
TAGGED = f"Capacitance {TAG} depends only on the plates and the gap, not on the voltage."
UNTAGGED = f"Resistivity {TAG} of copper is low."


class Model:
    """Replies with whatever the test sets, and keeps every request."""

    model = "fake"

    def __init__(self):
        self.reply = "C does not change with V [S1]."
        self.requests: list[str] = []

    def complete(self, system, messages, max_tokens=None, temperature=None):
        self.requests.append(messages[-1]["content"])
        return self.reply


@pytest.fixture
def chat(monkeypatch):
    init_db()
    with session_scope() as db:
        concept = Concept(slug=f"capa-{TAG}", name=f"Capa {TAG}")
        db.add(concept)
        db.flush()
        db.add(Misconception(slug=SLUG, concept_id=concept.id, label=LABEL))
        doc = Document(title=f"mixups {TAG}", filename="m.txt", content_type="text/plain",
                       storage_path="x", byte_size=1, status=IngestStatus.ready)
        db.add(doc)
        db.flush()
        tagged = DocumentChunk(document_id=doc.id, chunk_index=0, page_start=23, page_end=23,
                               word_count=14, content=TAGGED, embedding=_vector(TAGGED))
        untagged = DocumentChunk(document_id=doc.id, chunk_index=1, page_start=9, page_end=9,
                                 word_count=6, content=UNTAGGED, embedding=_vector(UNTAGGED))
        db.add_all([tagged, untagged])
        db.flush()
        db.execute(insert(chunk_concepts).values(chunk_id=tagged.id, concept_id=concept.id))
        ids = {"doc": doc.id, "concept": concept.id, "tagged": tagged.id, "untagged": untagged.id}

    model = Model()
    monkeypatch.setattr(rag_service, "get_llm", lambda: model)
    monkeypatch.setattr(rag_service, "get_embedding_provider", lambda: Embedder())
    monkeypatch.setattr(rag_service, "get_reranker", lambda: NoOpReranker())
    monkeypatch.setattr(settings, "enable_query_rewriting", False)
    monkeypatch.setattr(settings, "enable_spelling_correction", False)
    conversations: list = []

    with TestClient(app) as client:
        def ask(question):
            r = client.post("/chat/ask", json={"question": question, "include_trace": True,
                                               "document_ids": [str(ids["doc"])]})
            assert r.status_code == 200, r.text
            conversations.append(uuid.UUID(r.json()["conversation_id"]))
            return r.json()
        yield ask, model, ids

    with session_scope() as db:
        db.execute(delete(Conversation).where(Conversation.id.in_(conversations)))
        db.execute(delete(Document).where(Document.id == ids["doc"]))
        db.execute(delete(Concept).where(Concept.id == ids["concept"]))


def test_a_stated_mistake_is_named_and_the_label_never_shown(chat):
    ask, model, _ = chat
    model.reply = f"No: capacitance {TAG} does not depend on voltage [S1].\n[MIX-UP: {SLUG}]"
    body = ask(f"capacitance {TAG} doubles if the voltage doubles, right?")
    assert body["misconception"] == {"slug": SLUG, "label": LABEL}
    assert "MIX-UP" not in body["answer"]
    assert body["answer"].startswith("No: capacitance")


def test_the_mistakes_offered_are_those_of_the_passages_topics(chat):
    ask, model, _ = chat
    body = ask(f"capacitance {TAG} doubles if the voltage doubles, right?")
    assert f"- {SLUG}: {LABEL}" in model.requests[-1]
    assert body["trace"]["stages"]["mix_up"] == {"offered": [SLUG], "spotted": None}
    assert body["misconception"] is None            # no label in the reply


def test_every_detection_is_kept_with_the_message(chat):
    """Saved, so how often it is right can be checked before it counts."""
    from apps.api.models import Message

    ask, model, _ = chat
    model.reply = f"No [S1].\n[MIX-UP: {SLUG}]"
    body = ask(f"capacitance {TAG} doubles if the voltage doubles, right?")
    with session_scope() as db:
        saved = db.get(Message, uuid.UUID(str(body["message_id"])))
        assert saved.retrieval_trace["stages"]["mix_up"]["spotted"] == SLUG


def test_a_made_up_label_is_ignored(chat):
    ask, model, _ = chat
    model.reply = "No [S1].\n[MIX-UP: something/else]"
    body = ask(f"capacitance {TAG} doubles if the voltage doubles, right?")
    assert body["misconception"] is None and "MIX-UP" not in body["answer"]


def test_no_named_mistakes_means_the_request_is_unchanged(chat):
    """A passage tagged with no topic offers nothing, so nothing is added."""
    ask, model, ids = chat
    with session_scope() as db:
        db.execute(delete(chunk_concepts).where(chunk_concepts.c.chunk_id == ids["tagged"]))
    ask(f"resistivity {TAG} of copper")
    assert "Common mistakes" not in model.requests[-1]


def test_an_answer_without_sources_carries_no_correction(chat):
    """A correction with no citation behind it is the model's opinion."""
    ask, model, _ = chat
    model.reply = f"I think you are wrong.\n[MIX-UP: {SLUG}]"
    body = ask(f"capacitance {TAG} doubles if the voltage doubles, right?")
    assert not body["grounded"] and body["misconception"] is None


def test_a_revision_summary_is_not_checked(chat):
    """The request is ours, not the student's: there is no mistake to find."""
    ask, model, _ = chat
    body = ask(f"revise capa {TAG}")
    assert body["mode"] == "review"
    assert "Common mistakes" not in model.requests[-1]
    assert body["trace"]["stages"]["mix_up"]["offered"] == []
