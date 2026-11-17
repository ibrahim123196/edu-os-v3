"""Grounded study chat."""

from __future__ import annotations

import logging
import re
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ai.generation.llm import LLMError
from ai.retrieval.hybrid import RetrievalFilters
from apps.api.db import get_db
from apps.api.models import Conversation, Message
from apps.api.schemas import AskRequest, AskResponse
from ai.tutor.conversation import recent_history
from ai.tutor.modes import Mode, ModeGuess, detect, points_back
from apps.api.student import active_student_id
from apps.api.services import tutor_service
from apps.api.services.rag_service import ask as run_pipeline
from apps.api.services.rag_service import resolve_follow_up

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, db: Session = Depends(get_db)):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Type a question first.")

    conversation = _resolve_conversation(db, payload.conversation_id, question)
    # Read before this question is saved, so it is not its own history.
    history = recent_history(conversation.messages)

    db.add(Message(conversation_id=conversation.id, role="user", content=question))
    db.commit()

    started = time.perf_counter()
    guess = detect(question)
    if payload.lesson:
        # The Lesson section: the box holds a topic ("capacitance"), or a
        # request for one ("teach me capacitance"), whose request words are
        # dropped. Never a quiz or a revision summary, whatever the words.
        lesson_on = guess.topic if guess.cue and guess.topic else question
        guess = ModeGuess(Mode.learn, lesson_on, "lesson section")
    topic = guess.topic
    mode_trace: dict = {"mode": guess.mode.value, "cue": guess.cue, "topic": topic}
    # "Quiz me on this": the topic is in the conversation, not the message.
    if guess.mode in (Mode.practice, Mode.test, Mode.review) and points_back(topic) and history:
        resolved = resolve_follow_up(question, history)
        if resolved:
            topic = detect(resolved).topic or resolved
            mode_trace["resolved_from_conversation"] = resolved

    # Practice and test: a quiz, not a textbook answer.
    if guess.mode in (Mode.practice, Mode.test):
        reply = tutor_service.quiz_reply(db, guess.mode, topic)
        mode_trace.update(reply.trace)
        return _mode_response(db, conversation, guess.mode, reply, started, mode_trace,
                              payload.include_trace)

    if guess.mode == Mode.review and not topic:
        reply = tutor_service.ModeReply(
            text="Which topic would you like to revise? For example: revise Gauss's law.",
            understood_as="Revision (no topic yet)")
        return _mode_response(db, conversation, guess.mode, reply, started, mode_trace,
                              payload.include_trace)

    ask_text, ask_history, search_as = question, history, None
    understood_as = None
    progress_note = None
    action = None
    if guess.mode == Mode.review:
        # A revision summary of the topic, searched as one. Asked on its
        # own: earlier turns would pull the summary towards whatever was
        # discussed last instead of the whole topic.
        match = tutor_service.find_topic(db, topic)
        name = match.name if match else topic
        # A group is revised as its parts, named, so the search reaches each
        # of them rather than whichever one its name sounds most like.
        scope = name
        if match and len(match.concepts) > 1:
            scope = f"{name} ({', '.join(c.name for c in match.concepts)})"
        ask_text, ask_history = tutor_service.REVIEW_REQUEST.format(topic=scope), None
        search_as = scope
        understood_as = f"Revision of {name}"
        if match:
            mode_trace.update(matched=match.group or match.slugs[0], how=match.how)
            progress_note = tutor_service.progress_note(db, active_student_id(), match)
            available = tutor_service.concepts_with_questions(db)
            members = [c for c in match.concepts if available.get(c.id)]
            questions = sum(available[c.id] for c in members)
            if questions:
                action = tutor_service.quiz_action(
                    members, name, min(tutor_service.PRACTICE_COUNT, questions))
    elif payload.lesson:
        understood_as = f"Lesson on {topic}"
        # A lesson teaches the topic whole: earlier turns would pull it
        # towards whatever was discussed last.
        ask_history = None
    elif guess.mode == Mode.learn:
        understood_as = f"Learning about {topic}" if topic else None

    try:
        result = run_pipeline(
            db,
            question=ask_text,
            filters=RetrievalFilters(
                document_ids=payload.document_ids,
                subject=payload.subject,
                grade=payload.grade,
            ),
            history=ask_history,
            search_as=search_as,
            # A request the student did not phrase as a claim -- a revision
            # summary, "teach me" -- has no mistake in it to look for.
            watch_for_mixups=guess.mode not in (Mode.review, Mode.learn),
            student_id=active_student_id(),
            # Only from the Lesson section. "Teach me ..." typed in the ask
            # box gets an ordinary answer: the long lesson is asked for, not
            # guessed from the wording.
            lesson_topic=topic if payload.lesson else None,
        )
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Pipeline failed")
        raise HTTPException(
            status_code=500, detail=f"Something broke while answering: {exc}"
        ) from exc

    answer = result.answer
    result.trace["mode"] = mode_trace
    lesson = _lesson_payload(db, result) if result.lesson else None
    latency = round((time.perf_counter() - started) * 1000, 1)

    message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer.answer,
        citations=[c.__dict__ for c in answer.citations],
        retrieval_trace=result.trace,
        grounded=answer.grounded,
        latency_ms=latency,
    )
    db.add(message)
    db.commit()
    db.refresh(message)

    return AskResponse(
        answer=answer.answer,
        citations=[c.__dict__ for c in answer.citations],
        grounded=answer.grounded,
        used_source_count=answer.used_source_count,
        retrieved_count=answer.retrieved_count,
        conversation_id=conversation.id,
        message_id=message.id,
        latency_ms=latency,
        corrections=result.corrections,
        corrected_question=result.corrected_question,
        searched_as=result.searched_as,
        mode=guess.mode.value,
        understood_as=understood_as or result.searched_as,
        action=action if answer.grounded else None,
        progress_note=progress_note,
        pitch_note=result.pitch.note,
        misconception=(
            {"slug": result.misconception.slug, "label": result.misconception.label}
            if result.misconception else None
        ),
        lesson=lesson,
        trace=result.trace if payload.include_trace else None,
    )


def _lesson_payload(db, result) -> dict:
    """The lesson's steps, its mini-question and its check, for the page.

    The mini-question is a reviewed question from the bank on the lesson's
    topic, handed out as a one-question quiz, so it is scored and saved
    like any other. The passages are kept in the trace (saved with the
    message), so feedback on the check is judged against the same ones.
    """
    from apps.api.routes.quiz import _question_out
    from apps.api.services import figure_service, session_service
    from apps.api.services.quiz_service import select_questions
    from apps.api.services.rag_service import lead_concepts

    lesson = result.lesson
    concept_ids = lead_concepts(db, result.chunk_ids)
    picked = select_questions(db, concept_ids, count=1, student_id=active_student_id()) \
        if concept_ids else []
    quiz = None
    if picked:
        session = session_service.start_session(db, active_student_id(), picked)
        quiz = {"session_id": str(session.id), "question": _question_out(picked[0])}
    result.trace["lesson_context"] = {
        "chunk_ids": [str(c) for c in result.chunk_ids],
        "check": lesson.check,
        "quiz_session": quiz and quiz["session_id"],
    }
    visual = next((x for k, _, x in lesson.steps if k == "visual"), "")
    # The passages the visual step itself cites, by its [S#] markers.
    by_marker = {c.marker: c.chunk_id for c in result.answer.citations}
    cited = [uuid.UUID(str(by_marker[m])) for m in
             dict.fromkeys(re.findall(r"\[(S\d+)\]", visual)) if m in by_marker]
    figure = figure_service.for_lesson(db, visual, result.chunk_ids, cited) if visual else None
    result.trace["stages"].setdefault("lesson", {})["figure"] = figure and figure["label"]
    return {
        "steps": [{"key": k, "title": t, "text": x} for k, t, x in lesson.steps],
        "figure": figure,
        "quiz": quiz,
        "check": lesson.check,
        "misconception": (
            {"slug": lesson.misconception.slug, "label": lesson.misconception.label}
            if lesson.misconception else None
        ),
    }


class FeedbackRequest(BaseModel):
    reply: str = Field(min_length=1, max_length=2000)


@router.post("/lessons/{message_id}/feedback")
def lesson_feedback(message_id: uuid.UUID, payload: FeedbackRequest, db: Session = Depends(get_db)):
    """Feedback on the student's own explanation, judged against the same
    passages the lesson was written from. Saved in the conversation, so a
    follow-up question can build on it."""
    from ai.generation.answer import build_answer
    from ai.prompts.grounded_answer import SYSTEM_PROMPT, format_context
    from ai.tutor.lesson import FEEDBACK
    from apps.api.services import rag_service
    from apps.api.services.lesson_sources import load_chunks

    message = db.get(Message, message_id)
    context = (message.retrieval_trace or {}).get("lesson_context") if message else None
    if not context or not context.get("check"):
        raise HTTPException(status_code=404, detail="No lesson to give feedback on.")

    chunks = load_chunks(db, [uuid.UUID(c) for c in context["chunk_ids"]])
    reply = payload.reply.strip()
    try:
        raw = rag_service.get_llm().complete(system=SYSTEM_PROMPT, messages=[{
            "role": "user",
            "content": f"Sources:\n\n{format_context(chunks)}\n\n---\n\n"
                       + FEEDBACK.format(check=context["check"], reply=reply),
        }])
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    feedback = build_answer(raw, chunks)

    db.add(Message(conversation_id=message.conversation_id, role="user", content=reply))
    db.add(Message(conversation_id=message.conversation_id, role="assistant",
                   content=feedback.answer, citations=[c.__dict__ for c in feedback.citations],
                   grounded=feedback.grounded))
    db.commit()
    return {"feedback": feedback.answer, "grounded": feedback.grounded,
            "citations": [c.__dict__ for c in feedback.citations]}


def _mode_response(db, conversation, mode, reply, started, mode_trace, include_trace):
    """A reply that is not a textbook answer: an offer, or a question back.

    Saved like any answer, so the conversation stays question-answer pairs
    and "and on capacitors too?" has something to follow on from.
    """
    trace = {"mode": mode_trace}
    message = Message(conversation_id=conversation.id, role="assistant",
                      content=reply.text, retrieval_trace=trace, grounded=None)
    db.add(message)
    db.commit()
    db.refresh(message)
    return AskResponse(
        answer=reply.text,
        citations=[],
        grounded=False,
        used_source_count=0,
        retrieved_count=0,
        conversation_id=conversation.id,
        message_id=message.id,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        mode=mode.value,
        understood_as=reply.understood_as,
        action=reply.action,
        trace=trace if include_trace else None,
    )


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: uuid.UUID, db: Session = Depends(get_db)):
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="No such conversation.")
    return {
        "id": str(conversation.id),
        "title": conversation.title,
        "created_at": conversation.created_at,
        "messages": [
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "citations": m.citations or [],
                "grounded": m.grounded,
                "created_at": m.created_at,
            }
            for m in conversation.messages
        ],
    }


@router.get("/conversations")
def list_conversations(limit: int = 20, db: Session = Depends(get_db)):
    rows = db.scalars(
        select(Conversation).order_by(Conversation.created_at.desc()).limit(limit)
    ).all()
    return [
        {"id": str(c.id), "title": c.title, "created_at": c.created_at} for c in rows
    ]


def _resolve_conversation(
    db: Session, conversation_id: uuid.UUID | None, question: str
) -> Conversation:
    if conversation_id is not None:
        conversation = db.get(Conversation, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="No such conversation.")
        return conversation

    title = question[:80] + ("…" if len(question) > 80 else "")
    conversation = Conversation(title=title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation
