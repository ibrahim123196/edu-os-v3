"""Flagging a question that looks wrong, and deciding what to do about it."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.quiz import FLAG_REASONS, Attempt, Question, QuestionFlag, QuestionStatus
from apps.api.services.mastery_service import recompute


class FlagError(Exception):
    pass


# Rejection reasons that mean the answer key cannot be trusted. Not
# repeats, and not wrong-concept: those questions were answerable, and an
# answer to them still says something about the student.
_FAULTY_PREFIXES = (
    "faulty",                        # retired by review, or flagged and confirmed
    "check: key says",               # the checker answered differently
    "check: false premise",
    "check: no option is correct",
)


def reason_means_faulty(reason: str | None) -> bool:
    if not reason:
        return False
    return reason.startswith(_FAULTY_PREFIXES) or (
        reason.startswith("check:") and "options are defensible" in reason
    )


def flag(
    db: Session, *, student_id: uuid.UUID, question_id: uuid.UUID, reason: str, note: str | None
) -> QuestionFlag:
    if reason not in FLAG_REASONS:
        raise FlagError(f"unknown reason {reason!r}")
    if db.get(Question, question_id) is None:
        raise FlagError("no such question")
    # A second click is the same flag, not another one: counting it twice
    # would make one doubt look like two people's.
    existing = db.scalar(
        select(QuestionFlag).where(
            QuestionFlag.question_id == question_id,
            QuestionFlag.student_id == student_id,
            QuestionFlag.resolved_at.is_(None),
        )
    )
    if existing is not None:
        return existing
    note = (note or "").strip()[:500] or None
    f = QuestionFlag(question_id=question_id, student_id=student_id, reason=reason, note=note)
    db.add(f)
    db.flush()
    return f


def open_flags(db: Session) -> list[QuestionFlag]:
    """One row per flagged question, oldest first: the review list."""
    flags = db.scalars(
        select(QuestionFlag).where(QuestionFlag.resolved_at.is_(None)).order_by(QuestionFlag.created_at)
    ).all()
    seen, first = set(), []
    for f in flags:
        if f.question_id not in seen:
            seen.add(f.question_id)
            first.append(f)
    return first


def mark_faulty(db: Session, question: Question, reason: str) -> int:
    """Retire a question as faulty, and recompute mastery for everyone who
    answered it -- their answers to it stop counting. Returns how many
    students' mastery changed."""
    question.status = QuestionStatus.rejected
    question.faulty = True
    question.rejection_reason = reason[:500]
    db.flush()
    students = db.scalars(
        select(Attempt.student_id).where(Attempt.question_id == question.id).distinct()
    ).all()
    for student_id in students:
        recompute(db, student_id, question.concept_id)
    return len(students)


def resolve(db: Session, question_id: uuid.UUID, *, retire: bool) -> int:
    """Close every open flag on a question; retire it or put it back."""
    question = db.get(Question, question_id)
    flags = db.scalars(
        select(QuestionFlag).where(
            QuestionFlag.question_id == question_id, QuestionFlag.resolved_at.is_(None)
        )
    ).all()
    now = datetime.now(timezone.utc)
    for f in flags:
        f.resolved_at, f.resolution = now, ("retired" if retire else "restored")
    if not retire:
        return 0
    reasons = sorted({FLAG_REASONS[f.reason] for f in flags})
    return mark_faulty(db, question, "faulty (flagged): " + "; ".join(reasons))
