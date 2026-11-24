"""A student saying a tutor answer looks wrong.

Quiz questions could already be reported (QuestionFlag); answers from the
Tutor tab, a Full lesson or Work it out could not, so a wrong one was only
found if Ibrahim happened to read it. A report keeps the message it is about
-- the question asked, the answer, the pages cited -- so it can be judged
later without asking the student anything. Reviewed with
`python -m scripts.review_answers`.

One report per student per answer: sending again replaces the reason.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from apps.api.db import Base

# Shown in this order. Short, in a student's words.
REASONS: dict[str, str] = {
    "not_in_book": "It says something the textbook doesn't",
    "wrong_fact": "A fact, formula or number is wrong",
    "wrong_page": "The page it points to doesn't say this",
    "missed_question": "It didn't answer my question",
    "confusing": "It's confusing or badly explained",
    "other": "Something else",
}

# How a report was closed.
OUTCOMES = {"fixed": "the answer was wrong; something was changed",
            "fine": "the answer was right after all"}


class AnswerReport(Base):
    __tablename__ = "answer_reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True,
                                          default=uuid.uuid4)
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"),
        nullable=False, index=True)
    reason: Mapped[str] = mapped_column(String(40), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
    # Which part of a lesson, e.g. "Worked example"; empty for a plain answer.
    part: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    outcome: Mapped[str | None] = mapped_column(String(20))
    review_note: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("uq_answer_report_message_student", "message_id", "student_id", unique=True),
    )
