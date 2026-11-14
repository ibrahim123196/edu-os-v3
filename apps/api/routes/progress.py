"""What the student knows, and what to do about it."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from apps.api.db import get_db
from apps.api.services.mastery_service import progress_report
from apps.api.student import active_student_id

router = APIRouter(prefix="/progress", tags=["progress"])


@router.get("")
def progress(db: Session = Depends(get_db)):
    """Mastery per concept, with the likely cause where one can be named.

    Until there is a login this reports the placeholder student, which is
    everything answered on this device.
    """
    return progress_report(db, active_student_id())
