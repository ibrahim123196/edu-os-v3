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


class NotActive(SessionError):
    """Answering a quiz that was ended, or a test that was left and not
    yet continued."""

    status_code = 409


KINDS = ("practice", "test")

# The longest one stretch of a test may count, between two answers. A test
# whose tab was simply closed keeps its clock "running" until it is next
# touched; without a cap it would come back as hours long.
MAX_STRETCH_SECONDS = 10 * 60


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _bank(session: QuizSession, now: datetime, *, keep_running: bool) -> None:
    """Add the running stretch to the banked seconds."""
    if session.timing_since is not None:
        stretch = (now - session.timing_since).total_seconds()
        session.active_seconds += int(min(max(stretch, 0), MAX_STRETCH_SECONDS))
    session.timing_since = now if keep_running else None


def elapsed_seconds(session: QuizSession, now: datetime | None = None) -> int:
    """Time spent on a test so far, counting the stretch still running."""
    now = now or _now()
    running = 0
    if session.timing_since is not None:
        running = min(max((now - session.timing_since).total_seconds(), 0), MAX_STRETCH_SECONDS)
    return session.active_seconds + int(running)


def _own(db: Session, student_id: uuid.UUID, session_id: uuid.UUID) -> QuizSession:
    session = db.get(QuizSession, session_id)
    if session is None or session.student_id != student_id:
        raise SessionNotFound("No such quiz.")
    return session


def start_session(
    db: Session, student_id: uuid.UUID, questions: list[Question], kind: str = "practice"
) -> QuizSession | None:
    """Record which questions were handed out together.

    No questions, no session: an empty quiz would sit in the history as an
    unfinished 0 out of 0 forever. A test starts its clock now.
    """
    if not questions:
        return None
    if kind not in KINDS:
        raise ValueError(f"Unknown quiz kind {kind!r}")
    session = QuizSession(student_id=student_id, kind=kind,
                          timing_since=_now() if kind == "test" else None)
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
    session = _own(db, student_id, session_id)
    if session.stopped == "ended":
        raise NotActive("This quiz has ended.")
    if session.stopped == "left":
        raise NotActive("This test was left. Continue it to answer.")

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
    now = _now()
    done = score["answered"] == score["total"]
    if session.kind == "test":
        # Banked at every answer, so a closed tab loses at most one stretch.
        _bank(session, now, keep_running=not done)
    if done and session.finished_at is None:
        session.finished_at = now
    score["finished"] = session.finished_at is not None
    if session.kind == "test":
        score["elapsed_seconds"] = session.active_seconds

    result["session"] = score
    return result


def leave(db: Session, student_id: uuid.UUID, session_id: uuid.UUID) -> dict:
    """Leave a test part-way: the clock stops, and it can be continued."""
    session = _own(db, student_id, session_id)
    if session.kind != "test":
        raise NotActive("Only a test can be left and continued; end a quiz instead.")
    if session.finished_at is not None:
        raise NotActive("This test is already finished.")
    _bank(session, _now(), keep_running=False)
    session.stopped = "left"
    return _summary(db, session)


def end(db: Session, student_id: uuid.UUID, session_id: uuid.UUID) -> dict:
    """End a quiz early. The answers given count; the rest are not marked
    wrong, they are simply not part of the score."""
    session = _own(db, student_id, session_id)
    if session.kind != "practice":
        raise NotActive("A test is left, not ended, so it can be continued.")
    if session.finished_at is None:
        session.stopped = "ended"
    return _summary(db, session)


def resume(db: Session, student_id: uuid.UUID, session_id: uuid.UUID) -> dict:
    """Continue a test: the clock restarts and every question comes back,
    in its original order, the answered ones with what was chosen -- a test
    can go back to look at them, and must not be able to answer them again."""
    session = _own(db, student_id, session_id)
    if session.kind != "test":
        raise NotActive("Only a test can be continued.")
    if session.finished_at is not None:
        raise NotActive("This test is already finished.")
    now = _now()
    # A test closed without "Leave" still has a stretch running; it is
    # banked (capped) before the new one starts.
    _bank(session, now, keep_running=True)
    session.stopped = None
    chosen = {
        a.question_id: a.selected_option_id
        for a in db.scalars(select(Attempt).where(Attempt.session_id == session.id))
    }
    out = _summary(db, session)
    # (question, None) when unanswered; (question, (option id,)) when
    # answered -- a tuple, because "answered with no option" is possible.
    out["questions"] = [
        (db.get(Question, sq.question_id),
         (chosen[sq.question_id],) if sq.question_id in chosen else None)
        for sq in session.questions
    ]
    return out


def _summary(db: Session, session: QuizSession) -> dict:
    out = _score(db, session.id)
    out.update(
        session_id=str(session.id),
        kind=session.kind,
        stopped=session.stopped,
        finished=session.finished_at is not None,
    )
    if session.kind == "test":
        out["elapsed_seconds"] = elapsed_seconds(session)
    return out


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
            SELECT s.id, s.started_at, s.finished_at, s.kind, s.stopped,
                   s.active_seconds, s.timing_since,
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
    now = _now()

    def seconds(r) -> int | None:
        if r.kind != "test":
            return None          # quizzes are not timed
        running = 0
        if r.timing_since is not None:
            running = min(max((now - r.timing_since).total_seconds(), 0), MAX_STRETCH_SECONDS)
        return r.active_seconds + int(running)

    return [
        {
            "id": str(r.id),
            "started_at": r.started_at.isoformat(),
            "finished": r.finished_at is not None,
            "total": r.total,
            "answered": r.answered,
            "correct": r.correct,
            "concepts": list(r.concepts),
            "kind": r.kind,
            "stopped": r.stopped,
            "seconds": seconds(r),
            # Any unfinished test, including one whose tab was closed
            # without "Leave": nothing is lost by offering to continue it.
            "can_continue": r.kind == "test" and r.finished_at is None,
        }
        for r in rows
    ]
