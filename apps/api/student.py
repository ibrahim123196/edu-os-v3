"""Students and what they know.

`student_id` exists from the first version even though there is no login
yet, and every attempt carries it. The alternative -- adding it when
accounts arrive -- means finding every query that assumed a single student,
and a mastery query missing its filter returns a plausible number that is
silently wrong. That is the failure mode this project has hit repeatedly,
so the column is here before it is needed.

Until there is authentication, everything belongs to one placeholder
student. Making that explicit is the point: a row named "local" is obviously
a stand-in, where an absent column is an assumption nobody remembers making.
"""

from __future__ import annotations

import uuid
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


def active_student_id() -> uuid.UUID:
    """The student this running server records answers against.

    The one place routes ask. When login arrives this becomes "the student
    in the session", and nothing that calls it has to change.
    """
    from apps.api.config import get_settings

    return STUDENTS[get_settings().active_student][0]


class Student(Base):
    __tablename__ = "students"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    grade: Mapped[str | None] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    mastery: Mapped[list["ConceptMastery"]] = relationship(
        back_populates="student", cascade="all, delete-orphan"
    )


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
