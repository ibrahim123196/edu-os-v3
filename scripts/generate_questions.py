"""Build the question bank from the indexed passages.

Questions are generated once and stored, not produced per request: a student
waiting on a model call before they can answer anything is a worse
experience than a smaller bank, and a stored question can be validated,
reviewed and corrected.

Rejected questions are kept. Without them the same bad question is generated
again next week and nobody notices it is a pattern.

    python -m scripts.generate_questions --per-concept 2
    python -m scripts.generate_questions --concept coulombs-law --difficulty 3

Costs a fraction of a cent per question. Generating two for every concept in
the corpus is pennies.
"""

from __future__ import annotations

import argparse
import sys
import time

from sqlalchemy import func, select

from apps.api.curriculum import Concept, chunk_concepts
from apps.api.db import init_db, session_scope
from apps.api.quiz import Question, QuestionStatus
from apps.api.services.quiz_service import generate_for_concept


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate quiz questions.")
    parser.add_argument("--concept", help="one concept slug (default: all with passages)")
    parser.add_argument("--per-concept", type=int, default=1)
    parser.add_argument("--difficulty", type=int, default=2, choices=range(1, 6))
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="leave concepts that already have approved questions alone",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    init_db()

    with session_scope() as db:
        query = select(Concept).where(
            Concept.id.in_(select(chunk_concepts.c.concept_id))
        )
        if args.concept:
            query = query.where(Concept.slug == args.concept)
        concepts = db.scalars(query.order_by(Concept.name)).all()

        if not concepts:
            print("No concept has any passages. Re-ingest, or check the slug.")
            return 1

        if args.skip_existing:
            already = {
                cid
                for (cid,) in db.execute(
                    select(Question.concept_id)
                    .where(Question.status == QuestionStatus.approved)
                    .group_by(Question.concept_id)
                    .having(func.count(Question.id) >= args.per_concept)
                ).all()
            }
            concepts = [c for c in concepts if c.id not in already]
            if not concepts:
                print("Every concept already has enough questions.")
                return 0

        print(f"{len(concepts)} concept(s), {args.per_concept} question(s) each\n")

        approved = rejected = refused = 0
        for concept in concepts:
            outcome = generate_for_concept(
                db, concept, count=args.per_concept, difficulty=args.difficulty
            )
            approved += outcome.approved
            rejected += outcome.rejected
            refused += outcome.refused

            flag = "ok " if outcome.approved else "-- "
            print(
                f"  {flag} {concept.name[:38]:<38} "
                f"{outcome.approved} approved, {outcome.rejected} rejected"
            )
            for reason in outcome.reasons[:2]:
                print(f"        {reason}")

    elapsed = time.perf_counter() - started
    print(
        f"\n{approved} approved, {rejected} rejected, {refused} refused "
        f"({elapsed:.0f}s)"
    )
    if rejected:
        print(
            "Rejected questions are kept. Review them with:\n"
            "  SELECT stem, rejection_reason FROM questions "
            "WHERE status = 'rejected';"
        )
    return 0 if approved else 1


if __name__ == "__main__":
    sys.exit(main())