"""Learning modes through /chat/ask, against a real database.

Stand-in model, embedder and reranker, as in test_conversation_sql. Concept
names are made up so they cannot collide with the real curriculum.
Local database only (see conftest).
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
from apps.api.quiz import Question, QuestionKind, QuestionOption, QuestionStatus
from apps.api.services import rag_service, tutor_service
from apps.api.services.quiz_service import grade
from apps.api.student import active_student_id
from tests.integration.test_conversation_sql import Embedder, _vector

pytestmark = pytest.mark.integration

# Starts with a letter: an all-digit tag ("94616") is read as a count, like
# the 5 in "5 questions on ...", and trimmed off the topic -- which made
# these tests fail one run in ten.
TAG = "t" + uuid.uuid4().hex[:5]
ZORB = f"Zorblax law {TAG}"          # has questions
QUUX = f"Quuxian flow {TAG}"         # none
PASSAGE = f"Flibber resonance {TAG}: the zorblax constant links flibber and resonance."


class Model:
    model = "fake"

    def __init__(self):
        self.answers: list[list[dict]] = []

    def complete(self, system, messages, max_tokens=None, temperature=None):
        if system.startswith("You rewrite a student's follow-up"):
            return f"Give me practice questions on {ZORB}"
        self.answers.append(messages)
        return "The zorblax constant links flibber and resonance [S1]."


@pytest.fixture
def world(monkeypatch):
    init_db()
    with session_scope() as db:
        zorb, quux = Concept(slug=f"zorb-{TAG}", name=ZORB), Concept(slug=f"quux-{TAG}", name=QUUX)
        db.add_all([zorb, quux])
        db.flush()
        for i in range(3):
            q = Question(concept_id=zorb.id, kind=QuestionKind.multiple_choice,
                         stem=f"Zorb {i}?", explanation="b", status=QuestionStatus.approved)
            q.options = [QuestionOption(position=0, text="yes", is_correct=True),
                         QuestionOption(position=1, text="no", is_correct=False)]
            db.add(q)
        doc = Document(title=f"modes {TAG}", filename="m.txt", content_type="text/plain",
                       storage_path="x", byte_size=1, status=IngestStatus.ready)
        db.add(doc)
        db.flush()
        chunk = DocumentChunk(document_id=doc.id, chunk_index=0, page_start=1, page_end=1,
                              word_count=9, content=PASSAGE, embedding=_vector(PASSAGE))
        db.add(chunk)
        db.flush()
        # Tagged to Zorblax, though the passage never says "law": found by search.
        db.execute(insert(chunk_concepts).values(chunk_id=chunk.id, concept_id=zorb.id))
        ids = {"zorb": zorb.id, "quux": quux.id, "doc": doc.id}

    model = Model()
    monkeypatch.setattr(rag_service, "get_llm", lambda: model)
    monkeypatch.setattr(rag_service, "get_embedding_provider", lambda: Embedder())
    monkeypatch.setattr(rag_service, "get_reranker", lambda: NoOpReranker())
    import ai.embeddings.provider as provider
    monkeypatch.setattr(provider, "get_embedding_provider", lambda: Embedder())
    monkeypatch.setattr(settings, "enable_query_rewriting", False)
    monkeypatch.setattr(settings, "enable_spelling_correction", False)
    conversations: list = []

    with TestClient(app) as client:
        def ask(question, conversation_id=None):
            r = client.post("/chat/ask", json={
                "question": question, "conversation_id": conversation_id,
                "document_ids": [str(ids["doc"])], "include_trace": True})
            assert r.status_code == 200, r.text
            conversations.append(r.json()["conversation_id"])
            return r.json()
        yield ask, model, ids

    with session_scope() as db:
        db.execute(delete(Conversation).where(
            Conversation.id.in_({uuid.UUID(c) for c in conversations})))
        db.execute(delete(Document).where(Document.id == ids["doc"]))
        db.execute(delete(Question).where(Question.concept_id.in_([ids["zorb"], ids["quux"]])))
        db.execute(delete(Concept).where(Concept.id.in_([ids["zorb"], ids["quux"]])))


# --- practice and test -----------------------------------------------------------

def test_practice_on_a_named_topic_offers_that_quiz(world):
    ask, model, _ = world
    body = ask(f"quiz me on {ZORB}")
    assert body["mode"] == "practice"
    assert body["understood_as"] == f"Practice questions on {ZORB}"
    assert body["action"] == {"kind": "quiz", "concept_slug": f"zorb-{TAG}",
                              "concept": ZORB, "count": 3}
    assert "only 3 on this topic" in body["answer"]
    assert model.answers == []                       # no textbook answer, no cost


def test_a_topic_without_questions_is_not_offered_an_empty_quiz(world):
    ask, _, _ = world
    body = ask(f"quiz me on {QUUX}")
    assert body["action"] == {"kind": "open_practice"}
    assert body["answer"] == f"There are no practice questions on {QUUX} yet. Pick another topic in the Practise tab."


def test_an_unknown_topic_is_not_swapped_for_another(world):
    """Search always returns something; a quiz on that something is not
    what was asked for."""
    ask, _, _ = world
    body = ask("quiz me on photosynthesis")
    assert body["action"] == {"kind": "open_practice"}
    assert "couldn't match \"photosynthesis\"" in body["answer"]


def test_test_me_is_a_mixed_quiz(world):
    ask, _, _ = world
    body = ask("test me")
    assert body["mode"] == "test"
    assert body["action"]["concept_slug"] is None and body["action"]["count"] >= 3
    assert body["understood_as"] == "Test across all topics"


def test_quiz_me_on_this_takes_the_topic_from_the_conversation(world):
    ask, _, _ = world
    first = ask("what links flibber and resonance")
    body = ask("quiz me on this", first["conversation_id"])
    assert body["action"]["concept_slug"] == f"zorb-{TAG}"
    assert body["trace"]["mode"]["resolved_from_conversation"] == f"Give me practice questions on {ZORB}"


def test_test_me_after_a_question_is_still_mixed(world):
    """Nothing named is not "that": it once became a test on the last topic."""
    ask, _, _ = world
    first = ask("what links flibber and resonance")
    body = ask("test me", first["conversation_id"])
    assert body["action"]["concept_slug"] is None
    assert body["understood_as"] == "Test across all topics"


def test_the_conversation_carries_on_after_an_offer(world):
    """The offer is saved as the tutor's turn; without it the next request
    would have two student turns in a row and be refused by the API."""
    ask, model, _ = world
    offer = ask(f"quiz me on {ZORB}")
    ask("what links flibber and resonance", offer["conversation_id"])
    assert [m["role"] for m in model.answers[-1]] == ["user", "assistant", "user"]


# --- review -------------------------------------------------------------------

def test_review_asks_for_a_summary_and_shows_the_record(world):
    ask, model, _ = world
    body = ask(f"revise {ZORB}")
    assert body["mode"] == "review"
    assert body["understood_as"] == f"Revision of {ZORB}"
    assert body["progress_note"] == f"Your progress on {ZORB}: not practised yet."
    assert f"revision summary of {ZORB}" in model.answers[-1][-1]["content"]
    assert body["action"]["concept_slug"] == f"zorb-{TAG}"
    assert body["action"]["count"] == 3              # all there are, not 5


def test_the_record_is_the_students_own(world):
    ask, _, ids = world
    with session_scope() as db:
        for q in db.query(Question).filter(Question.concept_id == ids["zorb"]):
            grade(db, q, q.options[0].id, student_id=active_student_id())
    body = ask(f"revise {ZORB}")
    assert body["progress_note"].startswith(f"Your progress on {ZORB}: ")
    assert "from only 3 question(s)" in body["progress_note"]


def test_revise_with_no_topic_asks_which(world):
    ask, model, _ = world
    body = ask("let's revise")
    assert "Which topic" in body["answer"] and model.answers == []


# --- learn and clarify -----------------------------------------------------------

def test_a_plain_question_is_answered_as_before(world):
    ask, _, _ = world
    body = ask("what links flibber and resonance")
    assert body["mode"] == "clarify" and body["understood_as"] is None
    assert body["action"] is None and body["grounded"]


def test_learn_is_answered_and_labelled(world):
    ask, _, _ = world
    body = ask("teach me flibber resonance")
    assert body["mode"] == "learn"
    assert body["understood_as"] == "Learning about flibber resonance"
    assert body["grounded"]


# --- finding the topic ---------------------------------------------------------

def test_a_topic_not_named_is_found_through_the_passages_tagged_with_it(world):
    _, _, ids = world
    with session_scope() as db:
        match = tutor_service.find_concept(db, "flibber resonance")
        assert match.concept.id == ids["zorb"] and match.how == "search"
