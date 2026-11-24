"""How much each student may ask the AI, and how much EduOS may spend a day.

Anyone can make an account now, so one student (or one script) could ask
thousands of questions on Ibrahim's API key. Two limits:

1. Per student: requests that call the AI, in the last hour and the last
   24 hours. Counted per request -- one question, one lesson step, one
   Work-it-out reply -- not per model call, so the number means something
   to a student ("30 questions an hour").
2. For everyone together: what today's calls cost, at list prices. Sign-up
   lets one person make several accounts, so per-student limits alone do
   not cap the bill; this does. "Today" is India's day.

Every request that may call the AI gets a row in `ai_usage` before it runs;
each model call during it adds its tokens and cost to that row. The rows
are also the record of who used what.

Setting a limit to 0 turns it off (the automated tests do, see
tests/conftest.py).
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    func,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from ai.generation import usage
from apps.api.db import Base

log = logging.getLogger(__name__)

INDIA = timezone(timedelta(hours=5, minutes=30))


class AiUsage(Base):
    """One request that called (or was allowed to call) the AI."""

    __tablename__ = "ai_usage"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True,
                                          default=uuid.uuid4)
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"),
        nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(30), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # A model missing from the price table: its calls are counted, but the
    # daily cap cannot see their cost.
    unpriced: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


# The ai_usage row of the request being handled, for the model calls in it.
_current: ContextVar[uuid.UUID | None] = ContextVar("eduos_ai_usage", default=None)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _start_of_india_day(now: datetime) -> datetime:
    local = now.astimezone(INDIA)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


def _at(moment: datetime) -> str:
    """A time as a student in India would read it: 21:45."""
    return moment.astimezone(INDIA).strftime("%H:%M")


def reserve(db, student_id: uuid.UUID, kind: str, settings) -> uuid.UUID:
    """Check the limits; if there is room, record the request and return its
    row. HTTPException 429 if not."""
    now = _now()

    for limit, window, words in (
        (settings.ai_requests_per_hour, timedelta(hours=1), "in the last hour"),
        (settings.ai_requests_per_day, timedelta(days=1), "in the last 24 hours"),
    ):
        if limit <= 0:
            continue
        since = now - window
        times = db.scalars(
            select(AiUsage.created_at)
            .where(AiUsage.student_id == student_id, AiUsage.created_at > since)
            .order_by(AiUsage.created_at)
        ).all()
        if len(times) >= limit:
            # Room again once the oldest that counts falls out of the window.
            free_at = times[len(times) - limit] + window
            raise HTTPException(status_code=429, detail=(
                f"You have asked the tutor {len(times)} time{'' if len(times) == 1 else 's'} "
                f"{words}, the most for now. "
                f"You can ask again at {_at(free_at)}. Quizzes and your plan still work."))

    if settings.ai_daily_budget_usd > 0:
        start = _start_of_india_day(now)
        spent = db.scalar(select(func.coalesce(func.sum(AiUsage.cost_usd), 0.0))
                          .where(AiUsage.created_at >= start))
        if spent >= settings.ai_daily_budget_usd:
            log.warning("Daily AI budget of $%.2f reached ($%.3f spent)",
                        settings.ai_daily_budget_usd, spent)
            raise HTTPException(status_code=429, detail=(
                "EduOS has answered as many questions as it can today. It starts "
                "again after midnight. Quizzes and your plan still work."))

    row = AiUsage(student_id=student_id, kind=kind, created_at=now)
    db.add(row)
    db.commit()
    return row.id


def allowance(kind: str):
    """A route dependency: this request may call the AI, within the limits.

    Async on purpose: the row id it sets must be seen by the route, and a
    value set in an async dependency is (a sync one runs in a thread whose
    changes are thrown away).
    """

    async def check() -> None:
        from starlette.concurrency import run_in_threadpool

        from apps.api.config import get_settings
        from apps.api.db import SessionLocal
        from apps.api.student import active_student_id

        student_id = active_student_id()

        def run():
            with SessionLocal() as db:
                return reserve(db, student_id, kind, get_settings())

        _current.set(await run_in_threadpool(run))

    return Depends(check)


def _on_model_call(model: str, input_tokens: int | None, output_tokens: int | None) -> None:
    """Add a model call's tokens and cost to the request's ai_usage row."""
    row_id = _current.get()
    if row_id is None:  # a script, or a route with no limit
        return
    from apps.api.db import SessionLocal

    i, o = input_tokens or 0, output_tokens or 0
    price = usage.PRICES.get(model)
    values = {
        "calls": AiUsage.calls + 1,
        "input_tokens": AiUsage.input_tokens + i,
        "output_tokens": AiUsage.output_tokens + o,
    }
    if price is None:
        values["unpriced"] = True
    else:
        values["cost_usd"] = AiUsage.cost_usd + (i * price[0] + o * price[1]) / 1_000_000
    try:
        with SessionLocal() as db:
            db.execute(update(AiUsage).where(AiUsage.id == row_id).values(**values))
            db.commit()
    except Exception:  # never fail an answer over its bookkeeping
        log.warning("Could not record AI usage", exc_info=True)


usage.add_listener(_on_model_call)
