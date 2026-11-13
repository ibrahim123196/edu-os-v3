"""Recompute every mastery estimate from the stored attempts.

Mastery is cached, and a cache can drift. Deleting questions takes their
attempts with them, but nothing recomputes the estimates that counted those
attempts -- so the stored numbers keep describing answers that no longer
exist. The doctor notices the mismatch; this fixes it.

Also the way to adopt a change to the estimator. The attempts are the source
of truth and every stored value is derived from them, so when the formula
changes the whole table can simply be rebuilt.

    python -m scripts.rebuild_mastery
"""

from __future__ import annotations

import sys

from sqlalchemy import func, select

from apps.api.db import init_db, session_scope
from apps.api.quiz import Attempt, Question
from apps.api.services.mastery_service import recompute
from apps.api.student import ConceptMastery, Student


def main() -> int:
    init_db()

    with session_scope() as db:
        students = db.scalars(select(Student)).all()
        if not students:
            print("No students.")
            return 1

        total_rebuilt = 0
        for student in students:
            # Only concepts this student has actually attempted. Rebuilding
            # the rest would store a prior-only estimate for every concept
            # in the curriculum and make "attempted" meaningless.
            concept_ids = list(
                db.scalars(
                    select(Question.concept_id)
                    .join(Attempt, Attempt.question_id == Question.id)
                    .where(Attempt.student_id == student.id)
                    .distinct()
                ).all()
            )

            # Clear estimates for concepts with no attempts left, which is
            # what happens when the questions behind them are deleted.
            stale = db.scalars(
                select(ConceptMastery).where(
                    ConceptMastery.student_id == student.id,
                    ConceptMastery.concept_id.notin_(concept_ids)
                    if concept_ids
                    else True,
                )
            ).all()
            for row in stale:
                db.delete(row)

            for concept_id in concept_ids:
                recompute(db, student.id, concept_id)

            attempts = db.scalar(
                select(func.count())
                .select_from(Attempt)
                .where(Attempt.student_id == student.id)
            ) or 0

            print(
                f"  {student.display_name}: {len(concept_ids)} concept(s) "
                f"from {attempts} attempt(s)"
                + (f", {len(stale)} stale row(s) removed" if stale else "")
            )
            total_rebuilt += len(concept_ids)

    print(f"\n{total_rebuilt} estimate(s) rebuilt")
    return 0


if __name__ == "__main__":
    sys.exit(main())