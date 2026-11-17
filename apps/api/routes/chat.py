"""Grounded study chat."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ai.generation.llm import LLMError
from ai.retrieval.hybrid import RetrievalFilters
from apps.api.db import get_db
from apps.api.models import Conversation, Message
from apps.api.schemas import AskRequest, AskResponse
from ai.tutor.conversation import recent_history
from ai.tutor.modes import Mode, detect, points_back
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
            watch_for_mixups=guess.mode != Mode.review,
            student_id=active_student_id(),
        )
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Pipeline failed")
        raise HTTPException(
            status_code=500, detail=f"Something broke while answering: {exc}"
        ) from exc

    latency = round((time.perf_counter() - started) * 1000, 1)
    answer = result.answer
    result.trace["mode"] = mode_trace

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
        trace=result.trace if payload.include_trace else None,
    )


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
