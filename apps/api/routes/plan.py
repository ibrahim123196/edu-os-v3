"""The study plan: set it up, see today and the days ahead, tick sessions off."""

from __future__ import annotations

import uuid
import datetime as dt

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from apps.api.db import get_db
from apps.api.services import plan_service
from apps.api.student import active_student_id

router = APIRouter(prefix="/plan", tags=["plan"])


class ExamSettings(BaseModel):
    name: str = Field(default="", max_length=200)
    date: dt.date
    chapter_ids: list[uuid.UUID] = Field(min_length=1)


class PlanSettings(BaseModel):
    # One or more exams -- a Physics and a Chemistry one, say -- sharing
    # the same weekly hours.
    exams: list[ExamSettings] = Field(min_length=1, max_length=plan_service.MAX_EXAMS)
    # Minutes for each day, Monday first.
    minutes: list[int] = Field(min_length=7, max_length=7)


class LinkRequest(BaseModel):
    quiz_session_id: uuid.UUID


def _call(db: Session, fn, *args):
    try:
        out = fn(db, active_student_id(), *args)
    except plan_service.PlanError as err:
        db.rollback()
        raise HTTPException(status_code=err.status_code, detail=err.message) from err
    db.commit()
    return out


@router.get("")
def get_plan(db: Session = Depends(get_db)):
    """The plan as it stands: today's sessions, the next two weeks, the
    last week's record. Rebuilt from the latest results on every call."""
    out = plan_service.view(db, active_student_id())
    db.commit()           # today's sessions, the first time today is asked for
    return out


@router.put("/settings")
def save_settings(payload: PlanSettings, db: Session = Depends(get_db)):
    _call(db, plan_service.save_settings, [e.model_dump() for e in payload.exams],
          payload.minutes)
    return get_plan(db)


@router.post("/sessions/{session_id}/link")
def link(session_id: uuid.UUID, payload: LinkRequest, db: Session = Depends(get_db)):
    """The quiz a session opened, or a reattempt of it. Finishing it ticks
    the session off."""
    s = _call(db, plan_service.link, session_id, payload.quiz_session_id)
    return {"id": str(s.id), "status": s.status}
