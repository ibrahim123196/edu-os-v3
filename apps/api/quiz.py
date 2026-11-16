"""The question bank.

A question is stored, not generated on demand. Generating one per request
would be slower, more expensive, non-reproducible, and impossible to review:
a bad question would reach a student once and vanish, so nobody could fix it.
Storing them means a question can be validated before anyone sees it, reused
across students, and corrected when it turns out to be wrong.

Every question points at the passage it came from. That is what makes a wrong
answer explainable -- "you said B, the material says A, here it is on page
14" -- and what lets a reviewer check the question against its source rather
than against their own memory of the physics.

Questions carry a status because generated questions are not trustworthy by
default. `draft` has been generated but not checked; `approved` has passed
validation; `rejected` failed and is kept so the same mistake is visible
rather than silently regenerated.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from apps.api.db import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class QuestionKind(str, enum.Enum):
    multiple_choice = "multiple_choice"
    numerical = "numerical"
    short_answer = "short_answer"


class QuestionStatus(str, enum.Enum):
    draft = "draft"
    approved = "approved"
    rejected = "rejected"


class Difficulty(int, enum.Enum):
    """Coarse on purpose.

    Five bands a teacher could agree on beats a continuous score nobody can
    interpret. Calibration against real attempt data comes later, once there
    is any.
    """

    recall = 1
    understanding = 2
    application = 3
    analysis = 4
    challenge = 5


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)

    concept_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # The passage this came from. Null only for questions typed in by hand.
    source_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="SET NULL")
    )

    kind: Mapped[QuestionKind] = mapped_column(
        Enum(QuestionKind, name="question_kind"), nullable=False
    )
    stem: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)

    difficulty: Mapped[int] = mapped_column(Integer, nullable=False, default=2, index=True)
    status: Mapped[QuestionStatus] = mapped_column(
        Enum(QuestionStatus, name="question_status"),
        nullable=False,
        default=QuestionStatus.draft,
        index=True,
    )
    rejection_reason: Mapped[str | None] = mapped_column(Text)

    # For numerical questions.
    numeric_answer: Mapped[float | None] = mapped_column(Float)
    tolerance: Mapped[float | None] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(40))

    generated_by: Mapped[str | None] = mapped_column(String(80))
    # The model that answered this question blind and agreed with its key.
    # Null for questions from before the check existed: they were never
    # verified, and a review of them found faulty keys. Added after the
    # table shipped, so init_db adds it to existing databases.
    checked_by: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    options: Mapped[list["QuestionOption"]] = relationship(
        back_populates="question",
        cascade="all, delete-orphan",
        order_by="QuestionOption.position",
    )
    attempts: Mapped[list["Attempt"]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_question_concept_status", "concept_id", "status"),)


class Misconception(Base):
    """A named mistake students make with one concept.

    Loaded from content/misconceptions.json. The point is a shared name:
    when every question described its wrong answers in its own words, the
    same mistake made on two questions never counted as a repeat.
    """

    __tablename__ = "misconceptions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    # The id from the JSON file, e.g. "coulombs-law/inverse-r". Stable, so a
    # reworded label never breaks the questions already pointing here.
    slug: Mapped[str] = mapped_column(String(160), unique=True, nullable=False)
    concept_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(Text, nullable=False)


class QuestionOption(Base):
    __tablename__ = "question_options"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    question_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("questions.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Why this wrong answer is tempting. A distractor that corresponds to a
    # specific mistake is what turns a wrong answer into a diagnosis.
    misconception: Mapped[str | None] = mapped_column(Text)

    # Which named mistake this wrong option stands for. The free text above
    # says why it is tempting in this particular question; this says which
    # mistake it is, in words shared with every other question. Null where
    # none of the concept's named mistakes fits. Added after the table
    # shipped, so init_db adds it to existing databases.
    misconception_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("misconceptions.id", ondelete="SET NULL"),
        index=True,
    )
    named_misconception: Mapped[Misconception | None] = relationship()

    question: Mapped[Question] = relationship(back_populates="options")

    __table_args__ = (
        Index("ix_option_question_position", "question_id", "position", unique=True),
    )


class Attempt(Base):
    """One answer to one question, by one student."""

    __tablename__ = "attempts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("students.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("questions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    selected_option_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("question_options.id", ondelete="SET NULL")
    )
    answer_text: Mapped[str | None] = mapped_column(Text)
    numeric_answer: Mapped[float | None] = mapped_column(Float)

    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    seconds_taken: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # The quiz this answer was given in. Nullable because attempts recorded
    # before sessions existed belong to none, and they are still real
    # evidence for mastery. Added after the table shipped, so init_db adds
    # it to existing databases -- create_all never would.
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("quiz_sessions.id", ondelete="SET NULL"),
        index=True,
    )

    question: Mapped[Question] = relationship(back_populates="attempts")

    __table_args__ = (
        # One answer per question per quiz, enforced by the database. A
        # double click sends two requests; without this both are stored and
        # the question counts twice in the score and in mastery. Postgres
        # treats NULLs as distinct, so attempts without a session are free.
        Index("uq_attempt_session_question", "session_id", "question_id", unique=True),
    )


class QuizSession(Base):
    """One sitting: the questions handed out together, and how it went.

    Without this the server forgot which questions belonged together the
    moment it sent them, so "7 out of 10 on Tuesday" existed only in the
    browser until the tab closed.

    The score is not stored. It is counted from the attempts whenever it is
    asked for, so it can never disagree with them -- the same reason mastery
    is recomputed rather than nudged.
    """

    __tablename__ = "quiz_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("students.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Set when the last question is answered. A quiz left halfway stays
    # unfinished rather than being scored as if the rest were wrong.
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    questions: Mapped[list["QuizSessionQuestion"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="QuizSessionQuestion.position",
    )


class QuizSessionQuestion(Base):
    """Which questions a session handed out, in order.

    Needed for the total ("out of 10" includes the ones never answered) and
    so an answer can be refused when its question was never part of the quiz.
    """

    __tablename__ = "quiz_session_questions"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("quiz_sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("questions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    session: Mapped[QuizSession] = relationship(back_populates="questions")
