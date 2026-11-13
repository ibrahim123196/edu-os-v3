"""Quiz endpoints: take a quiz, answer a question."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.curriculum import Concept
from apps.api.db import get_db
from apps.api.quiz import Question
from apps.api.services.quiz_service import grade, select_questions, shuffled

router = APIRouter(prefix="/quiz", tags=["quiz"])


class QuizRequest(BaseModel):
    concept_slugs: list[str] = Field(default_factory=list)
    count: int = Field(default=5, ge=1, le=20)
    difficulty: int | None = Field(default=None, ge=1, le=5)
    exclude: list[uuid.UUID] = Field(default_factory=list)


class AnswerRequest(BaseModel):
    question_id: uuid.UUID
    selected_option_id: uuid.UUID | None = None
    seconds_taken: float | None = None


@router.post("")
def start_quiz(payload: QuizRequest, db: Session = Depends(get_db)):
    """Hand back questions. Correct answers are not included."""
    concept_ids: list[uuid.UUID] = []
    if payload.concept_slugs:
        concepts = db.scalars(
            select(Concept).where(Concept.slug.in_(payload.concept_slugs))
        ).all()
        found = {c.slug for c in concepts}
        missing = set(payload.concept_slugs) - found
        if missing:
            raise HTTPException(
                status_code=404, detail=f"Unknown concept(s): {', '.join(sorted(missing))}"
            )
        concept_ids = [c.id for c in concepts]

    questions = select_questions(
        db,
        concept_ids=concept_ids or None,
        count=payload.count,
        difficulty=payload.difficulty,
        exclude=payload.exclude or None,
    )

    return {
        "count": len(questions),
        "questions": [
            {
                "id": str(q.id),
                "stem": q.stem,
                "difficulty": q.difficulty,
                "concept": q.concept_id and str(q.concept_id),
                # Shuffled, and is_correct is deliberately absent.
                "options": [
                    {"id": str(o.id), "text": o.text} for o in shuffled(q)
                ],
            }
            for q in questions
        ],
    }


@router.post("/answer")
def answer(payload: AnswerRequest, db: Session = Depends(get_db)):
    question = db.get(Question, payload.question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="No such question.")

    result = grade(db, question, payload.selected_option_id, payload.seconds_taken)
    db.commit()
    return result


@router.get("/concepts")
def concepts_with_questions(db: Session = Depends(get_db)):
    """Concepts a quiz can actually be built from."""
    from sqlalchemy import func

    from apps.api.quiz import QuestionStatus

    rows = db.execute(
        select(Concept.slug, Concept.name, func.count(Question.id))
        .join(Question, Question.concept_id == Concept.id)
        .where(Question.status == QuestionStatus.approved)
        .group_by(Concept.slug, Concept.name)
        .order_by(func.count(Question.id).desc())
    ).all()
    return [{"slug": s, "name": n, "questions": c} for s, n, c in rows]
