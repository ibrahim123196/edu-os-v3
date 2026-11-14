"""Move every attempt from the real student to the test student.

For the one-off cleanup after a day of click-through testing landed on the
real student. Moved rather than deleted: the attempts are still useful as
test data, and a move can be reversed where a delete cannot.

    python -m scripts.move_attempts_to_test           # show what would happen
    python -m scripts.move_attempts_to_test --apply   # do it

Only run --apply when *every* attempt on the real student is a test click.
Attempts carry no marker separating real answers from guesses, which is
exactly why this script cannot be selective and why testing now happens on
its own student.
"""

from __future__ import annotations

import sys

from sqlalchemy import delete, func, select, update

from apps.api.db import ensure_students, session_scope
from apps.api.quiz import Attempt, Question
from apps.api.services.mastery_service import recompute
from apps.api.student import LOCAL_STUDENT_ID, TEST_STUDENT_ID, ConceptMastery


def _summary(db, student_id) -> str:
    attempts = db.scalar(
        select(func.count()).select_from(Attempt).where(Attempt.student_id == student_id)
    )
    rows = db.scalar(
        select(func.count())
        .select_from(ConceptMastery)
        .where(ConceptMastery.student_id == student_id)
    )
    return f"{attempts} attempt(s), {rows} mastery row(s)"


def main(apply: bool) -> None:
    with session_scope() as db:
        print("Before")
        print(f"  real student: {_summary(db, LOCAL_STUDENT_ID)}")
        print(f"  test student: {_summary(db, TEST_STUDENT_ID)}")

        concept_ids = set(
            db.scalars(
                select(Question.concept_id)
                .join(Attempt, Attempt.question_id == Question.id)
                .where(Attempt.student_id == LOCAL_STUDENT_ID)
            ).all()
        )
        if not concept_ids:
            print("\nNothing to move.")
            return

        if not apply:
            print(f"\nWould move them, touching {len(concept_ids)} concept(s).")
            print("Nothing changed. Re-run with --apply to do it.")
            return

        # The server creates the test student when it starts; this script
        # may run before the server ever has. Same transaction, so a failure
        # below undoes this too.
        ensure_students(db)

        db.execute(
            update(Attempt)
            .where(Attempt.student_id == LOCAL_STUDENT_ID)
            .values(student_id=TEST_STUDENT_ID)
        )

        # Deleted, not recomputed to zero. A mastery row means "has been
        # attempted"; a zero-attempt row would make untouched concepts look
        # like answered ones to anything reading the table.
        db.execute(
            delete(ConceptMastery).where(ConceptMastery.student_id == LOCAL_STUDENT_ID)
        )

        db.flush()  # the moved attempts must be visible to recompute
        for concept_id in concept_ids:
            recompute(db, TEST_STUDENT_ID, concept_id)
        db.flush()

        print("\nAfter")
        print(f"  real student: {_summary(db, LOCAL_STUDENT_ID)}")
        print(f"  test student: {_summary(db, TEST_STUDENT_ID)}")
    # session_scope commits on leaving the block, or rolls back on any error.


if __name__ == "__main__":
    main(apply="--apply" in sys.argv[1:])
