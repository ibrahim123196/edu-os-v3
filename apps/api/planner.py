"""A student's study plan: what they set, and the sessions given to them.

Only what cannot be worked out again is stored. The settings -- exam, date,
chapters, minutes per weekday -- and each day's sessions once that day
arrives, so a student's day does not reshuffle while they work through it,
and what was done or missed stays on record. Days still to come are not
stored: they are a forecast, rebuilt from the latest results every time.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from apps.api.db import Base


class StudyPlan(Base):
    """One per student: the exam they are working towards."""

    __tablename__ = "study_plans"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"),
        nullable=False, unique=True,
    )
    exam_name: Mapped[str] = mapped_column(String(200), nullable=False)
    exam_date: Mapped[date] = mapped_column(Date, nullable=False)
    chapter_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # Minutes for each day of the week, Monday first.
    minutes: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # The day whose sessions were last given out. Cleared when the settings
    # change, so today is planned again under the new ones.
    planned_day: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    sessions: Mapped[list["PlanSession"]] = relationship(
        back_populates="plan", cascade="all, delete-orphan",
        order_by=lambda: (PlanSession.day, PlanSession.position))


class PlanSession(Base):
    """One session on one day, once that day has come."""

    __tablename__ = "plan_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plan_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("study_plans.id", ondelete="CASCADE"), nullable=False)
    day: Mapped[date] = mapped_column(Date, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    kind: Mapped[str] = mapped_column(String(20), nullable=False)   # quiz | test
    concept_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    label: Mapped[str] = mapped_column(String(300), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    question_count: Mapped[int | None] = mapped_column(Integer)

    # todo, done (its quiz finished); a past day's todo is shown as missed.
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="todo")
    done_how: Mapped[str | None] = mapped_column(String(20))
    done_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # The quiz it opened -- the latest, after a reattempt. Finishing it is
    # what ticks the session off.
    link_kind: Mapped[str | None] = mapped_column(String(20))
    link_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    # Every quiz taken for it, first to latest: the best score is kept.
    quiz_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list,
                                           server_default="[]")

    plan: Mapped[StudyPlan] = relationship(back_populates="sessions")

    __table_args__ = (Index("ix_plan_sessions_plan_day", "plan_id", "day"),)

