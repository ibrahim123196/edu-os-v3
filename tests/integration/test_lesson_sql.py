"""'Teach me ...' as a lesson, through /chat/ask and the feedback endpoint.

Stand-in model, embedder and reranker, as in test_pitch_sql. Local database
only (see conftest).
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, insert, select, update

from ai.ingestion.figures import VERSION
from ai.retrieval.rerank import NoOpReranker
from apps.api.config import settings
from apps.api.curriculum import Concept, chunk_concepts
from apps.api.db import init_db, session_scope
from apps.api.main import app
from apps.api.models import Conversation, Document, DocumentChunk, IngestStatus, Message
from apps.api.quiz import Misconception, Question, QuestionKind, QuestionOption, QuestionStatus
from apps.api.services import rag_service
from tests.integration.test_conversation_sql import Embedder, _vector

pytestmark = pytest.mark.integration

TAG = "t" + uuid.uuid4().hex[:5]
NAME = f"Lessonium {TAG}"
PASSAGE = f"Lessonium {TAG} is charge over potential. FIGURE 2.24 shows two plates."
MISTAKE = f"lesson-{TAG}/depends-on-v"

LESSON = f"""[INTUITION]
Lessonium {TAG} is how much charge two plates hold per volt [S1].
[EXPLANATION]
It is C = Q/V [S1].
[VISUAL]
Look at Figure 2.24: two plates facing each other [S1].
[EXAMPLE]
Double V and Q doubles; C stays [S1].
[MISCONCEPTION: {MISTAKE}]
It does not depend on V [S1].
[CHECK]
In your own words, what fixes lessonium {TAG}?"""


class Model:
    model = "fake"

    def __init__(self):
        self.requests: list[str] = []
        self.reply = LESSON

    def complete(self, system, messages, max_tokens=None, temperature=None):
        self.requests.append(messages[-1]["content"])
        if "The student was asked:" in messages[-1]["content"]:
            return "Right that it is fixed by the plates [S1]. Add that V does not change it [S1]."
        return self.reply


@pytest.fixture
def chat(monkeypatch):
    init_db()
    with session_scope() as db:
        concept = Concept(slug=f"lesson-{TAG}", name=NAME)
        db.add(concept)
        db.flush()
        db.add(Misconception(slug=MISTAKE, concept_id=concept.id, label="Thinks it depends on V"))
        q = Question(concept_id=concept.id, kind=QuestionKind.multiple_choice,
                     stem=f"What fixes lessonium {TAG}?", explanation="The plates.",
                     status=QuestionStatus.approved)
        q.options = [QuestionOption(position=0, text="The plates", is_correct=True),
                     QuestionOption(position=1, text="The voltage", is_correct=False)]
        db.add(q)
        doc = Document(title=f"lesson {TAG}", filename="l.txt", content_type="text/plain",
                       storage_path="x", byte_size=1, status=IngestStatus.ready)
        db.add(doc)
        db.flush()
        chunk = DocumentChunk(document_id=doc.id, chunk_index=0, page_start=23, page_end=23,
                              word_count=14, content=PASSAGE, embedding=_vector(PASSAGE))
        db.add(chunk)
        db.flush()
        db.execute(insert(chunk_concepts).values(chunk_id=chunk.id, concept_id=concept.id))
        ids = {"concept": concept.id, "doc": doc.id, "question": q.id}

    model = Model()
    monkeypatch.setattr(rag_service, "get_llm", lambda: model)
    monkeypatch.setattr(rag_service, "get_embedding_provider", lambda: Embedder())
    monkeypatch.setattr(rag_service, "get_reranker", lambda: NoOpReranker())
    monkeypatch.setattr(settings, "enable_query_rewriting", False)
    monkeypatch.setattr(settings, "enable_spelling_correction", False)
    conversations: list = []

    with TestClient(app) as client:
        def ask(question=f"lessonium {TAG}", lesson=True):
            r = client.post("/chat/ask", json={"question": question, "include_trace": True,
                                               "lesson": lesson,
                                               "document_ids": [str(ids["doc"])]})
            assert r.status_code == 200, r.text
            conversations.append(uuid.UUID(r.json()["conversation_id"]))
            return r.json()
        yield ask, model, ids, client

    with session_scope() as db:
        db.execute(delete(Conversation).where(Conversation.id.in_(conversations)))
        db.execute(delete(Document).where(Document.id == ids["doc"]))
        db.execute(delete(Concept).where(Concept.id == ids["concept"]))


def test_teach_me_gives_the_steps_in_order(chat):
    ask, model, _, _ = chat
    body = ask()
    assert body["mode"] == "learn"
    assert [s["key"] for s in body["lesson"]["steps"]] == [
        "intuition", "explanation", "visual", "example", "misconception"]
    assert body["lesson"]["steps"][2]["text"].startswith("Look at Figure 2.24")
    assert body["lesson"]["misconception"]["slug"] == MISTAKE
    assert body["lesson"]["check"] == f"In your own words, what fixes lessonium {TAG}?"
    assert body["grounded"] and body["citations"]
    assert "Teach the student this topic as a short lesson" in model.requests[-1]


def test_the_mini_question_is_a_real_one_and_is_scored(chat):
    ask, _, ids, client = chat
    quiz = ask()["lesson"]["quiz"]
    assert quiz["question"]["id"] == str(ids["question"])
    assert all("is_correct" not in o for o in quiz["question"]["options"])
    right = next(o["id"] for o in quiz["question"]["options"] if o["text"] == "The plates")
    r = client.post("/quiz/answer", json={"session_id": quiz["session_id"],
                                          "question_id": quiz["question"]["id"],
                                          "selected_option_id": right})
    assert r.status_code == 200 and r.json()["correct"]


def test_feedback_is_judged_against_the_lessons_own_passages(chat):
    ask, model, _, client = chat
    body = ask()
    r = client.post(f"/chat/lessons/{body['message_id']}/feedback",
                    json={"reply": "The plates decide it."})
    assert r.status_code == 200
    fb = r.json()
    assert fb["grounded"] and fb["feedback"].startswith("Right that it is fixed")
    asked = model.requests[-1]
    assert PASSAGE in asked                                   # the same passage
    assert f"what fixes lessonium {TAG}?" in asked and "The plates decide it." in asked


def test_feedback_is_kept_in_the_conversation(chat):
    """So a follow-up question can build on it."""
    ask, _, _, client = chat
    body = ask()
    client.post(f"/chat/lessons/{body['message_id']}/feedback", json={"reply": "The plates."})
    with session_scope() as db:
        roles = [m.role for m in db.scalars(
            select(Message).where(Message.conversation_id == uuid.UUID(body["conversation_id"]))
            .order_by(Message.created_at))]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_feedback_without_a_lesson_is_refused(chat):
    _, _, _, client = chat
    r = client.post(f"/chat/lessons/{uuid.uuid4()}/feedback", json={"reply": "x"})
    assert r.status_code == 404


def test_a_reply_without_the_sections_is_shown_as_an_answer(chat):
    """Never a page of tags: no sections found, no lesson."""
    ask, model, _, _ = chat
    model.reply = f"Lessonium {TAG} is charge over potential [S1]."
    body = ask()
    assert body["lesson"] is None and body["answer"].startswith(f"Lessonium {TAG}")


def test_a_lesson_without_sources_behind_it_is_not_given(chat):
    ask, model, _, _ = chat
    model.reply = LESSON.replace(" [S1]", "")
    body = ask()
    assert body["lesson"] is None and not body["grounded"]


def test_the_ask_box_never_gives_a_lesson(chat):
    """The lesson is asked for in its own section, not guessed from the
    wording: "teach me" in the ask box is answered like any question."""
    ask, model, _, _ = chat
    model.reply = f"Lessonium {TAG} is charge over potential [S1]."
    for question in (f"what is lessonium {TAG}", f"teach me lessonium {TAG}"):
        body = ask(question, lesson=False)
        assert body["lesson"] is None
        assert "short lesson" not in model.requests[-1]


def test_the_lesson_section_takes_a_topic_or_a_request(chat):
    ask, model, _, _ = chat
    for typed in (f"lessonium {TAG}", f"teach me lessonium {TAG}"):
        body = ask(typed)
        assert body["lesson"] and body["understood_as"] == f"Lesson on lessonium {TAG}"
        assert f"short lesson: lessonium {TAG}" in model.requests[-1]


def test_quiz_words_in_the_lesson_section_still_give_a_lesson(chat):
    """The section decides what happens, not the words typed in it."""
    ask, _, _, _ = chat
    body = ask(f"quiz me on lessonium {TAG}")
    assert body["lesson"] is not None and body["mode"] == "learn"
    assert body["understood_as"] == f"Lesson on lessonium {TAG}"


def test_a_lesson_ignores_earlier_turns(chat):
    ask, model, _, client = chat
    first = ask(f"what is lessonium {TAG}", lesson=False)
    r = client.post("/chat/ask", json={"question": f"lessonium {TAG}", "lesson": True,
                                       "conversation_id": first["conversation_id"]})
    assert r.status_code == 200
    assert f"what is lessonium {TAG}" not in model.requests[-1]


# --------------------------------------------------------------- the figure
# "Picture it" shows the textbook's own figure, cut from the PDF the lesson's
# passage came from.

@pytest.fixture
def with_pdf(chat, tmp_path, monkeypatch):
    """The lesson's document is a real PDF with FIGURE 2.24 in it, and
    figures are kept in a scratch folder."""
    from tests.unit.test_figures import make_pdf
    ask, model, ids, client = chat
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    pdf = make_pdf(tmp_path / "chapter.pdf", pages_before=22)
    with session_scope() as db:
        db.get(Document, ids["doc"]).storage_path = str(pdf)
    return chat, tmp_path


def test_picture_it_shows_the_textbooks_figure(with_pdf):
    (ask, _, ids, client), tmp = with_pdf
    lesson = ask()["lesson"]
    fig = lesson["figure"]
    assert fig == {"label": "2.24", "page": 23, "url": f"/figures/{ids['doc']}/2.24.png"}
    r = client.get(fig["url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG")
    assert (tmp / "figures" / f"{ids['doc']}_2.24_v{VERSION}.png").exists()     # cut once, kept


def test_no_figure_when_the_document_is_not_a_pdf(chat):
    """The fixture's document is a text file: words only."""
    ask, _, _, _ = chat
    assert ask()["lesson"]["figure"] is None


def test_a_figure_the_passages_do_not_mention_is_not_shown(with_pdf):
    """The model naming a figure is not enough: a made-up number, or one
    from another topic, must not put a picture on the step."""
    (ask, model, _, _), _ = with_pdf
    model.reply = LESSON.replace("Figure 2.24", "Figure 2.25")
    assert ask()["lesson"]["figure"] is None


def test_a_figure_that_cannot_be_cut_is_remembered(with_pdf):
    (ask, _, ids, client), tmp = with_pdf
    with session_scope() as db:
        db.execute(update(DocumentChunk).where(DocumentChunk.document_id == ids["doc"])
                   .values(content=PASSAGE.replace("2.24", "2.99")))
    model = with_pdf[0][1]
    model.reply = LESSON.replace("Figure 2.24", "Figure 2.99")
    assert ask()["lesson"]["figure"] is None
    assert (tmp / "figures" / f"{ids['doc']}_2.99_v{VERSION}.none").exists()
    assert client.get(f"/figures/{ids['doc']}/2.99.png").status_code == 404


def test_the_figure_endpoint_refuses_what_it_should(with_pdf):
    (_, _, ids, client), _ = with_pdf
    assert client.get(f"/figures/{uuid.uuid4()}/2.24.png").status_code == 404   # no document
    assert client.get(f"/figures/{ids['doc']}/..%2Fsecret.png").status_code == 404
    assert client.get(f"/figures/{ids['doc']}/abc.png").status_code == 404
    assert client.get(f"/figures/not-a-uuid/2.24.png").status_code == 422


def test_a_cutter_that_crashes_costs_only_the_picture(with_pdf, monkeypatch):
    from ai.ingestion import figures

    def broken(*a, **k):
        raise RuntimeError("bad page")
    monkeypatch.setattr(figures, "render", broken)
    (ask, _, _, _), _ = with_pdf
    lesson = ask()["lesson"]
    assert lesson["figure"] is None and len(lesson["steps"]) == 5


def test_the_model_is_told_which_figures_the_sources_have(chat):
    ask, model, _, _ = chat
    ask()
    assert "The sources mention these textbook figures: Figure 2.24" in model.requests[-1]


def test_a_step_that_describes_a_drawing_gets_its_passages_figure(with_pdf):
    """The model described a picture to draw instead of naming the figure:
    the figure in the passage that step cites is shown anyway."""
    (ask, model, ids, _), _ = with_pdf
    model.reply = LESSON.replace("Look at Figure 2.24: two plates facing each other [S1].",
                                 "Draw two plates facing each other [S1].")
    fig = ask()["lesson"]["figure"]
    assert fig and fig["label"] == "2.24"


def test_a_step_citing_nothing_gets_no_figure(with_pdf):
    """No name and no cited passage: nothing to say which figure fits."""
    (ask, model, _, _), _ = with_pdf
    model.reply = LESSON.replace("Look at Figure 2.24: two plates facing each other [S1].",
                                 "Picture two plates facing each other.")
    assert ask()["lesson"]["figure"] is None


def test_a_named_figure_that_cannot_be_cut_is_not_swapped_for_another(with_pdf):
    """The words under the picture are about the figure they name."""
    (ask, model, ids, _), _ = with_pdf
    with session_scope() as db:
        db.execute(update(DocumentChunk).where(DocumentChunk.document_id == ids["doc"])
                   .values(content=PASSAGE + " FIGURE 2.99 is elsewhere."))
    model.reply = LESSON.replace("Look at Figure 2.24", "Look at Figure 2.99")
    assert ask()["lesson"]["figure"] is None
