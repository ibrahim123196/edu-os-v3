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
from apps.api.services.rag_service import ask as run_pipeline

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, db: Session = Depends(get_db)):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Type a question first.")

    conversation = _resolve_conversation(db, payload.conversation_id, question)

    db.add(Message(conversation_id=conversation.id, role="user", content=question))
    db.commit()

    started = time.perf_counter()
    try:
        result = run_pipeline(
            db,
            question=question,
            filters=RetrievalFilters(
                document_ids=payload.document_ids,
                subject=payload.subject,
                grade=payload.grade,
            ),
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
        trace=result.trace if payload.include_trace else None,
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
