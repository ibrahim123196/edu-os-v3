"""Answers pitched to the student's record, through /chat/ask.

Stand-in model, embedder and reranker, as in test_mixups_sql. The record is
written straight into concept_mastery, except for repeated mistakes, which
come from real wrong answers. Local database only (see conftest).
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
from apps.api.quiz import Misconception, Question, QuestionKind, QuestionOption, QuestionStatus
from apps.api.services import rag_service
from apps.api.services.quiz_service import grade
from apps.api.student import ConceptMastery, active_student_id
from tests.integration.test_conversation_sql import Embedder, _vector

pytestmark = pytest.mark.integration

TAG = "t" + uuid.uuid4().hex[:5]
NAME = f"Pitchium {TAG}"
PASSAGE = f"Pitchium {TAG} is the ratio of charge to potential, fixed by geometry."


class Model:
    model = "fake"

    def __init__(self):
        self.requests: list[str] = []
        self.reply = f"Pitchium {TAG} is fixed by geometry [S1]."

    def complete(self, system, messages, max_tokens=None, temperature=None):
        self.requests.append(messages[-1]["content"])
        return self.reply


@pytest.fixture
def chat(monkeypatch):
    init_db()
    with session_scope() as db:
        concept = Concept(slug=f"pitch-{TAG}", name=NAME)
        db.add(concept)
        db.flush()
        doc = Document(title=f"pitch {TAG}", filename="p.txt", content_type="text/plain",
                       storage_path="x", byte_size=1, status=IngestStatus.ready)
        db.add(doc)
        db.flush()
        chunk = DocumentChunk(document_id=doc.id, chunk_index=0, page_start=1, page_end=1,
                              word_count=12, content=PASSAGE, embedding=_vector(PASSAGE))
        db.add(chunk)
        db.flush()
        db.execute(insert(chunk_concepts).values(chunk_id=chunk.id, concept_id=concept.id))
        ids = {"concept": concept.id, "doc": doc.id}

    model = Model()
    monkeypatch.setattr(rag_service, "get_llm", lambda: model)
    monkeypatch.setattr(rag_service, "get_embedding_provider", lambda: Embedder())
    monkeypatch.setattr(rag_service, "get_reranker", lambda: NoOpReranker())
    monkeypatch.setattr(settings, "enable_query_rewriting", False)
    monkeypatch.setattr(settings, "enable_spelling_correction", False)
    conversations: list = []

    with TestClient(app) as client:
        def ask(question=f"what is pitchium {TAG}"):
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


def _record(concept_id, mastery, attempts, confidence):
    with session_scope() as db:
        db.execute(delete(ConceptMastery).where(ConceptMastery.concept_id == concept_id,
                                                ConceptMastery.student_id == active_student_id()))
        db.add(ConceptMastery(student_id=active_student_id(), concept_id=concept_id,
                              mastery=mastery, confidence=confidence, attempts=attempts,
                              correct=round(mastery * attempts)))


def test_with_no_record_the_answer_is_as_for_anyone(chat):
    ask, model, _ = chat
    body = ask()
    assert body["pitch_note"] is None
    assert "This student" not in model.requests[-1]
    assert body["trace"]["stages"]["pitch"] == {"level": None, "note": None}


def test_a_weak_record_pitches_simpler_and_the_page_says_why(chat):
    ask, model, ids = chat
    _record(ids["concept"], mastery=0.35, attempts=8, confidence=0.7)
    body = ask()
    assert body["pitch_note"] == f"Pitched simpler: your quiz record on {NAME} is 35% from 8 questions."
    assert "from the basics" in model.requests[-1]
    assert body["grounded"]


def test_a_strong_record_pitches_deeper(chat):
    ask, model, ids = chat
    _record(ids["concept"], mastery=0.92, attempts=12, confidence=0.8)
    body = ask()
    assert body["pitch_note"].startswith("Pitched for more depth:")
    assert "skip the basics" in model.requests[-1]


def test_a_record_on_too_few_answers_changes_nothing(chat):
    ask, model, ids = chat
    _record(ids["concept"], mastery=0.1, attempts=2, confidence=0.2)
    assert ask()["pitch_note"] is None
    assert "This student" not in model.requests[-1]


def test_the_pitch_comes_before_the_mistake_list_and_both_are_sent(chat):
    ask, model, ids = chat
    with session_scope() as db:
        db.add(Misconception(slug=f"pitch-{TAG}/m", concept_id=ids["concept"],
                             label="Thinks pitchium depends on voltage"))
    _record(ids["concept"], mastery=0.35, attempts=8, confidence=0.7)
    ask()
    request = model.requests[-1]
    assert request.index("from the basics") < request.index("Common mistakes")


def _repeat_a_mistake(ids):
    """The same named mistake on two different questions: 'repeated'."""
    with session_scope() as db:
        m = Misconception(slug=f"pitch-{TAG}/rep", concept_id=ids["concept"],
                          label="Thinks pitchium grows with charge")
        db.add(m)
        db.flush()
        for i in range(2):
            q = Question(concept_id=ids["concept"], kind=QuestionKind.multiple_choice,
                         stem=f"Pitch {TAG} {i}?", explanation="b", status=QuestionStatus.approved)
            wrong = QuestionOption(position=1, text="grows", is_correct=False, misconception_id=m.id)
            q.options = [QuestionOption(position=0, text="fixed", is_correct=True), wrong]
            db.add(q)
            db.flush()
            grade(db, q, wrong.id, student_id=active_student_id())


def test_a_mistake_repeated_in_quizzes_is_watched_for(chat):
    ask, model, ids = chat
    _repeat_a_mistake(ids)
    body = ask()
    assert body["pitch_note"] == "Watching for a mistake you repeat: Thinks pitchium grows with charge."
    assert "repeatedly chosen answers" in model.requests[-1]


def test_a_watched_mistake_is_not_shown_again_as_a_mix_up_in_the_message(chat):
    """Seen: 'What is capacitance?' got 'Common mix-up' as well as the
    'Watching for a mistake you repeat' line -- the model labelled the quiz
    mistake it had just been told about, as if the question stated it."""
    ask, model, ids = chat
    _repeat_a_mistake(ids)
    model.reply = f"Pitchium {TAG} is fixed by geometry [S1].\n[MIX-UP: pitch-{TAG}/rep]"
    body = ask()
    assert body["pitch_note"].startswith("Watching for a mistake you repeat")
    assert body["misconception"] is None
    assert body["trace"]["stages"]["mix_up"]["spotted"] == f"pitch-{TAG}/rep"   # still recorded
    assert "not in this message" in model.requests[-1]


def test_a_different_mix_up_in_the_message_is_still_shown(chat):
    ask, model, ids = chat
    _repeat_a_mistake(ids)
    with session_scope() as db:
        db.add(Misconception(slug=f"pitch-{TAG}/other", concept_id=ids["concept"],
                             label="Thinks pitchium is measured in volts"))
    model.reply = f"No, in farads [S1].\n[MIX-UP: pitch-{TAG}/other]"
    body = ask(f"pitchium {TAG} is measured in volts, right?")
    assert body["misconception"]["slug"] == f"pitch-{TAG}/other"
