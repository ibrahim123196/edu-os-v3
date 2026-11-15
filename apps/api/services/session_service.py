"""Quiz sessions: start one, answer inside it, look back at past ones.

Every answer now belongs to a session, and the session decides what may be
answered. Before this the answer endpoint accepted any question id at all,
answered any number of times -- so a double click counted a question twice,
in the score and in mastery, and nothing could tell.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from apps.api.quiz import Attempt, Question, QuizSession, QuizSessionQuestion
from apps.api.services.quiz_service import grade


class SessionError(Exception):
    """An answer the session refuses. Carries the HTTP status to send."""

    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class SessionNotFound(SessionError):
    # Also used for another student's session: saying "that exists but is
    # not yours" would tell a stranger which session ids are real.
    status_code = 404


class NotInSession(SessionError):
    status_code = 400


class UnknownOption(SessionError):
    status_code = 400


class AlreadyAnswered(SessionError):
    status_code = 409


def start_session(
    db: Session, student_id: uuid.UUID, questions: list[Question]
) -> QuizSession | None:
    """Record which questions were handed out together.

    No questions, no session: an empty quiz would sit in the history as an
    unfinished 0 out of 0 forever.
    """
    if not questions:
        return None
    session = QuizSession(student_id=student_id)
    session.questions = [
        QuizSessionQuestion(question_id=q.id, position=i) for i, q in enumerate(questions)
    ]
    db.add(session)
    db.flush()  # the id is sent to the browser, so it must exist now
    return session


def answer(
    db: Session,
    *,
    student_id: uuid.UUID,
    session_id: uuid.UUID,
    question_id: uuid.UUID,
    selected_option_id: uuid.UUID | None,
    seconds_taken: float | None = None,
) -> dict:
    """Grade one answer inside a session, refusing anything the session
    did not hand out."""
    session = db.get(QuizSession, session_id)
    if session is None or session.student_id != student_id:
        raise SessionNotFound("No such quiz.")

    if question_id not in {sq.question_id for sq in session.questions}:
        raise NotInSession("That question is not part of this quiz.")

    question = db.get(Question, question_id)

    # An option from another question used to be graded as simply wrong,
    # recording a mistake the student never made.
    if selected_option_id is not None and selected_option_id not in {
        o.id for o in question.options
    }:
        raise UnknownOption("That answer does not belong to this question.")

    already = db.scalar(
        select(Attempt.id).where(
            Attempt.session_id == session_id, Attempt.question_id == question_id
        )
    )
    if already is not None:
        raise AlreadyAnswered("This question has already been answered in this quiz.")

    try:
        result = grade(
            db,
            question,
            selected_option_id,
            seconds_taken,
            student_id=student_id,
            session_id=session_id,
        )
        db.flush()
    except IntegrityError:
        # Two requests for the same question arrived together and both
        # passed the check above; the unique index stopped the second.
        db.rollback()
        raise AlreadyAnswered("This question has already been answered in this quiz.")

    score = _score(db, session_id)
    if score["answered"] == score["total"] and session.finished_at is None:
        session.finished_at = datetime.now(timezone.utc)
    score["finished"] = session.finished_at is not None

    result["session"] = score
    return result


def _score(db: Session, session_id: uuid.UUID) -> dict:
    """Counted from the attempts every time, never stored, so it cannot
    disagree with them."""
    row = db.execute(
        text(
            """
            SELECT count(*)                                  AS total,
                   count(a.id)                               AS answered,
                   count(a.id) FILTER (WHERE a.is_correct)   AS correct
            FROM quiz_session_questions sq
            LEFT JOIN attempts a
                   ON a.session_id = sq.session_id
                  AND a.question_id = sq.question_id
            WHERE sq.session_id = :sid
            """
        ),
        {"sid": session_id},
    ).one()
    return {"total": row.total, "answered": row.answered, "correct": row.correct}


def history(db: Session, student_id: uuid.UUID, limit: int = 20) -> list[dict]:
    """Past quizzes, newest first, with the topics each one covered.

    Attempts from before sessions existed belong to no session and do not
    appear here. They still count towards mastery.
    """
    rows = db.execute(
        text(
            """
            SELECT s.id, s.started_at, s.finished_at,
                   count(sq.question_id)                     AS total,
                   count(a.id)                               AS answered,
                   count(a.id) FILTER (WHERE a.is_correct)   AS correct,
                   array_agg(DISTINCT c.name ORDER BY c.name) AS concepts
            FROM quiz_sessions s
            JOIN quiz_session_questions sq ON sq.session_id = s.id
            JOIN questions q ON q.id = sq.question_id
            JOIN concepts c ON c.id = q.concept_id
            LEFT JOIN attempts a
                   ON a.session_id = s.id AND a.question_id = sq.question_id
            WHERE s.student_id = :sid
            GROUP BY s.id
            ORDER BY s.started_at DESC
            LIMIT :limit
            """
        ),
        {"sid": student_id, "limit": limit},
    ).all()
    return [
        {
            "id": str(r.id),
            "started_at": r.started_at.isoformat(),
            "finished": r.finished_at is not None,
            "total": r.total,
            "answered": r.answered,
            "correct": r.correct,
            "concepts": list(r.concepts),
        }
        for r in rows
    ]
