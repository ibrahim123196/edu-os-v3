"""What the student knows, and what to do about it."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from apps.api.db import get_db
from apps.api.services.mastery_service import progress_report
from apps.api.config import get_settings
from apps.api.student import STUDENTS, active_student_id

router = APIRouter(prefix="/progress", tags=["progress"])


@router.get("")
def progress(db: Session = Depends(get_db)):
    """Mastery per concept, with the likely cause where one can be named.

    Until there is a login this reports the placeholder student, which is
    everything answered on this device.
    """
    report = progress_report(db, active_student_id())
    # Said on the page, not just in settings: a progress page full of test
    # clicks looks exactly like a real one.
    key = get_settings().active_student
    report["student"] = {"key": key, "label": STUDENTS[key][1]}
    return report
