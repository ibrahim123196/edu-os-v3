"""Students and what they know.

`student_id` exists from the first version even though there is no login
yet, and every attempt carries it. The alternative -- adding it when
accounts arrive -- means finding every query that assumed a single student,
and a mastery query missing its filter returns a plausible number that is
silently wrong. That is the failure mode this project has hit repeatedly,
so the column is here before it is needed.

Since accounts (24 Nov), a request from the page belongs to whoever is
logged in: the login middleware (`apps/api/auth.py`) puts that student into
`_current`, and `active_student_id()` reads it. Outside a request -- scripts,
tests, the doctor -- nobody is logged in, and the ACTIVE_STUDENT setting
picks one of the two stand-ins, as before.

The two stand-ins stay. "local" is the student everything was recorded
against before accounts, so it becomes Ibrahim's own account (it is given
an email and password with `python -m scripts.accounts claim local`), and
"test" stays the place for click-through testing.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from apps.api.db import Base

# The stand-in used until accounts exist. A fixed UUID so it survives
# restarts and so every attempt lands on the same student.
LOCAL_STUDENT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")

# Where clicking through a quiz to test a feature goes. Twenty-one guesses
# once landed on the real student and the mastery model believed them: to
# the code a guess is indistinguishable from an answer, so the only defence
# is to keep testing on a different student entirely.
TEST_STUDENT_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")

STUDENTS: dict[str, tuple[uuid.UUID, str]] = {
    "local": (LOCAL_STUDENT_ID, "This device"),
    "test": (TEST_STUDENT_ID, "Testing (not a real student)"),
}

# The logged-in student of the request being handled; None outside one.
# Set by the login middleware for the length of the request only.
_current: ContextVar[uuid.UUID | None] = ContextVar("eduos_student", default=None)


def active_student_id() -> uuid.UUID:
    """The student this request belongs to: the one logged in.

    The one place routes ask. Outside a request (scripts, tests) it is the
    stand-in that ACTIVE_STUDENT names. A request with nobody logged in
    never reaches a route -- the middleware answers 401 first -- unless
    REQUIRE_LOGIN is off, which only the local setup allows.
    """
    current = _current.get()
    if current is not None:
        return current
    from apps.api.config import get_settings

    return STUDENTS[get_settings().active_student][0]


def current_or_none() -> uuid.UUID | None:
    """The logged-in student, or None outside a logged-in request."""
    return _current.get()


class Student(Base):
    __tablename__ = "students"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    grade: Mapped[str | None] = mapped_column(String(20))
    # Login (24 Nov). Null until an account is made for this student, so
    # the two stand-ins exist without being able to log in. The email is
    # stored lower-cased; unique among those that have one.
    email: Mapped[str | None] = mapped_column(String(320))
    password_hash: Mapped[str | None] = mapped_column(String(300))
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    mastery: Mapped[list["ConceptMastery"]] = relationship(
        back_populates="student", cascade="all, delete-orphan"
    )

    # Named as init_db names it for an older database, so both get one.
    __table_args__ = (Index("uq_students_email", "email", unique=True),)


class ConceptMastery(Base):
    """A cached estimate, recomputed whenever an attempt lands.

    Caching rather than computing on read: a study plan asks about every
    concept at once, and recomputing 43 estimates from raw attempts on every
    page load is work done repeatedly for an answer that only changes when
    someone answers a question.

    The raw attempts remain the source of truth. This table can be rebuilt
    from them at any time, which matters because the estimator will change.
    """

    __tablename__ = "concept_mastery"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"), nullable=False
    )
    concept_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False
    )

    mastery: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    correct: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    student: Mapped[Student] = relationship(back_populates="mastery")

    __table_args__ = (
        Index("ix_mastery_student_concept", "student_id", "concept_id", unique=True),
    )
