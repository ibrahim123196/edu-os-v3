"""Mark questions already retired for a faulty key as faulty. One-off.

    python -m scripts.mark_faulty_questions           # show what would change
    python -m scripts.mark_faulty_questions --apply   # do it

Questions were retired as faulty before the mark existed, so answers given
to them still count toward mastery. This finds them by the reason they were
retired with and marks them, recomputing mastery for anyone who answered.
Repeats and wrong-concept rejections are left alone: those questions were
answerable, and an answer to them still says something about the student.
"""

from __future__ import annotations

import sys

from sqlalchemy import func, select

import apps.api.models  # noqa: F401  (registers every table the flush may touch)
from apps.api.db import session_scope
from apps.api.quiz import Attempt, Question, QuestionStatus
from apps.api.services.flag_service import mark_faulty, reason_means_faulty


def main(apply: bool) -> int:
    with session_scope() as db:
        retired = db.scalars(
            select(Question).where(Question.status == QuestionStatus.rejected, ~Question.faulty)
        ).all()
        faulty = [q for q in retired if reason_means_faulty(q.rejection_reason)]
        answered = [
            q for q in faulty
            if db.scalar(select(func.count()).select_from(Attempt).where(Attempt.question_id == q.id))
        ]
        print(f"{len(faulty)} retired question(s) were faulty; {len(answered)} of them have answers")
        if not apply:
            print("Nothing changed. Re-run with --apply to mark them.")
            return 0
        students = sum(mark_faulty(db, q, q.rejection_reason) for q in faulty)
        print(f"Marked. Mastery recomputed {students} time(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main(apply="--apply" in sys.argv[1:]))
