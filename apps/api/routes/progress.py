"""What the student knows, and what to do about it."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from apps.api.db import get_db
from apps.api.services.mastery_service import progress_report
from apps.api.services.misconception_service import mistake_summary
from apps.api.student import STUDENTS, TEST_STUDENT_ID, Student, active_student_id

router = APIRouter(prefix="/progress", tags=["progress"])


@router.get("")
def progress(db: Session = Depends(get_db)):
    """Mastery per concept, with the likely cause where one can be named.

    For the logged-in student.
    """
    student_id = active_student_id()
    report = progress_report(db, student_id)
    report["mistakes"] = mistake_summary(db, student_id)
    # Said on the page, not just in settings: a progress page full of test
    # clicks looks exactly like a real one.
    key = "test" if student_id == TEST_STUDENT_ID else "local"
    student = db.get(Student, student_id)
    report["student"] = {"key": key, "label": student.display_name if student else STUDENTS[key][1]}
    return report
