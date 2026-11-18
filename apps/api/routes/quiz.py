"""Quiz endpoints: take a quiz, answer inside it, look back at past ones."""

from __future__ import annotations

import uuid

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.curriculum import Concept
from apps.api.db import get_db
from apps.api.quiz import Question
from apps.api.quiz import FLAG_REASONS
from apps.api.services import flag_service, session_service
from apps.api.services.quiz_service import select_questions, shuffled
from apps.api.student import active_student_id

router = APIRouter(prefix="/quiz", tags=["quiz"])


class QuizRequest(BaseModel):
    concept_slugs: list[str] = Field(default_factory=list)
    # Up to 50 for a test; the page keeps a quiz to 20.
    count: int = Field(default=5, ge=1, le=50)
    # "test" is Test me: every topic, timed, can be left and continued.
    kind: Literal["practice", "test"] = "practice"
    difficulty: int | None = Field(default=None, ge=1, le=5)
    exclude: list[uuid.UUID] = Field(default_factory=list)
    # The same questions as an earlier quiz of this student, in the same
    # order: a reattempt ("Try again" in the plan) asks exactly those again.
    repeat_of: uuid.UUID | None = None


class AnswerRequest(BaseModel):
    # Required. An answer outside a quiz has no score to count towards and
    # nothing to stop it being sent twice.
    session_id: uuid.UUID
    question_id: uuid.UUID
    selected_option_id: uuid.UUID | None = None
    seconds_taken: float | None = None


@router.post("")
def start_quiz(payload: QuizRequest, db: Session = Depends(get_db)):
    """Hand back questions. Correct answers are not included."""
    if payload.kind == "test" and payload.concept_slugs:
        raise HTTPException(status_code=400,
                            detail="A test covers every topic; practise a topic instead.")
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

    student_id = active_student_id()
    if payload.repeat_of is not None:
        questions = _same_questions(db, student_id, payload.repeat_of)
    else:
        questions = select_questions(
            db,
            concept_ids=concept_ids or None,
            count=payload.count,
            difficulty=payload.difficulty,
            exclude=payload.exclude or None,
            student_id=student_id,
        )
    session = session_service.start_session(db, student_id, questions, kind=payload.kind)
    # get_db does not commit. Without this the session id goes to the
    # browser and the row it names is thrown away when the request ends.
    db.commit()

    return {
        "session_id": str(session.id) if session else None,
        "kind": payload.kind,
        "count": len(questions),
        "questions": [_question_out(q) for q in questions],
    }


def _same_questions(db: Session, student_id: uuid.UUID, quiz_id: uuid.UUID) -> list[Question]:
    """An earlier quiz's questions, in its order, leaving out any retired
    since: a reattempt never brings back a question found faulty."""
    from apps.api.quiz import QuestionStatus, QuizSession, QuizSessionQuestion

    earlier = db.get(QuizSession, quiz_id)
    if earlier is None or earlier.student_id != student_id:
        raise HTTPException(status_code=404, detail="No such quiz to repeat.")
    rows = db.scalars(select(QuizSessionQuestion).where(QuizSessionQuestion.session_id == quiz_id)
                      .order_by(QuizSessionQuestion.position)).all()
    out = []
    for row in rows:
        q = db.get(Question, row.question_id)
        if q is not None and q.status == QuestionStatus.approved and not q.faulty:
            out.append(q)
    return out


def _question_out(q: Question) -> dict:
    return {
        "id": str(q.id),
        "stem": q.stem,
        "difficulty": q.difficulty,
        "concept": q.concept_id and str(q.concept_id),
        # Shuffled, and is_correct is deliberately absent.
        "options": [{"id": str(o.id), "text": o.text} for o in shuffled(q)],
    }


def _session_call(db: Session, fn, session_id: uuid.UUID) -> dict:
    try:
        out = fn(db, active_student_id(), session_id)
    except session_service.SessionError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message) from err
    db.commit()
    return out


@router.post("/sessions/{session_id}/leave")
def leave_test(session_id: uuid.UUID, db: Session = Depends(get_db)):
    """Leave a test part-way; it stays in the history to continue later."""
    return _session_call(db, session_service.leave, session_id)


@router.post("/sessions/{session_id}/end")
def end_quiz(session_id: uuid.UUID, db: Session = Depends(get_db)):
    """End a quiz early. Answers given count; the rest are left out."""
    return _session_call(db, session_service.end, session_id)


@router.post("/sessions/{session_id}/resume")
def resume_test(session_id: uuid.UUID, db: Session = Depends(get_db)):
    """Continue a left test: the clock restarts and every question returns;
    answered ones carry their result, to look at but not answer again."""
    from apps.api.services.quiz_service import explain

    out = _session_call(db, session_service.resume, session_id)
    questions = []
    for q, answer in out["questions"]:
        item = _question_out(q)
        if answer is not None:
            item["selected_option_id"] = answer[0] and str(answer[0])
            item["result"] = explain(q, answer[0])
        questions.append(item)
    out["questions"] = questions
    return out


@router.post("/answer")
def answer(payload: AnswerRequest, db: Session = Depends(get_db)):
    try:
        result = session_service.answer(
            db,
            student_id=active_student_id(),
            session_id=payload.session_id,
            question_id=payload.question_id,
            selected_option_id=payload.selected_option_id,
            seconds_taken=payload.seconds_taken,
        )
    except session_service.SessionError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message) from err
    db.commit()
    return result


class FlagRequest(BaseModel):
    question_id: uuid.UUID
    reason: str
    note: str | None = None


@router.post("/flag")
def flag_question(payload: FlagRequest, db: Session = Depends(get_db)):
    """'This question looks wrong.' The question leaves every quiz until
    it is reviewed (python -m scripts.review_flags)."""
    try:
        flag_service.flag(
            db,
            student_id=active_student_id(),
            question_id=payload.question_id,
            reason=payload.reason,
            note=payload.note,
        )
    except flag_service.FlagError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    db.commit()
    return {"flagged": True}


@router.get("/flag-reasons")
def flag_reasons():
    return [{"key": k, "label": v} for k, v in FLAG_REASONS.items()]


@router.get("/sessions")
def past_sessions(db: Session = Depends(get_db)):
    """Recent quizzes for the student this server is recording, newest
    first."""
    return session_service.history(db, active_student_id())


@router.get("/concepts")
def concepts_with_questions(db: Session = Depends(get_db)):
    """Concepts a quiz can actually be built from."""
    from sqlalchemy import func

    from apps.api.quiz import QuestionStatus

    rows = db.execute(
        select(Concept.slug, Concept.name, func.count(Question.id))
        .join(Question, Question.concept_id == Concept.id)
        # Faulty questions are never served, so they must not be counted:
        # the list once promised more questions than a quiz could give.
        .where(Question.status == QuestionStatus.approved, ~Question.faulty)
        .group_by(Concept.slug, Concept.name)
        .order_by(func.count(Question.id).desc())
    ).all()
    return [{"slug": s, "name": n, "questions": c} for s, n, c in rows]


@router.get("/groups")
def groups_with_questions(db: Session = Depends(get_db)):
    """Topic groups a quiz can be built from, chapters first: the same
    groups the chat understands, so "Capacitors" means the same thing in
    the list as typed."""
    from apps.api.services.tutor_service import concepts_with_questions
    from apps.api.topic_groups import GROUPS

    available = concepts_with_questions(db)
    by_slug = {c.slug: c for c in db.scalars(select(Concept)).all()}
    out = []
    for g in sorted(GROUPS, key=lambda g: not g.chapter):
        members = [by_slug[m] for m in g.members if m in by_slug and available.get(by_slug[m].id)]
        if members:
            out.append({"slug": g.slug, "name": g.name, "aliases": list(g.aliases),
                        "chapter": g.chapter,
                        "concept_slugs": [c.slug for c in members],
                        "questions": sum(available[c.id] for c in members)})
    return out
