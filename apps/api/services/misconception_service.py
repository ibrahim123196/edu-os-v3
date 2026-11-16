"""Which named mistakes a student keeps making.

A mistake counts as repeated when it was made on at least two *different*
questions. The same question answered wrongly twice may only mean the
first wrong answer was remembered; the same mistake turning up in two
questions written about different things is evidence it is the student's.
"""

from __future__ import annotations

import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session

# Two different questions. Lower and a single slip would be reported as a
# pattern; higher and with a few dozen answers nothing would ever show.
REPEAT_THRESHOLD = 2


def mistake_summary(db: Session, student_id: uuid.UUID) -> dict:
    rows = db.execute(
        text(
            """
            SELECT m.slug, m.label,
                   c.name AS concept, c.slug AS concept_slug,
                   count(*)                      AS times,
                   count(DISTINCT a.question_id) AS questions,
                   max(a.created_at)             AS last_seen
            FROM attempts a
            JOIN question_options o ON o.id = a.selected_option_id
            JOIN misconceptions m   ON m.id = o.misconception_id
            JOIN concepts c         ON c.id = m.concept_id
            JOIN questions q        ON q.id = a.question_id
            WHERE a.student_id = :sid
              AND NOT a.is_correct
              AND NOT q.faulty      -- a mistake against a faulty key is not one
            GROUP BY m.id, c.id
            ORDER BY count(DISTINCT a.question_id) DESC, max(a.created_at) DESC
            """
        ),
        {"sid": student_id},
    ).all()

    # How much there was to go on, said alongside the result: "no repeated
    # mistakes" means something different after 3 wrong answers than after 300.
    wrong, named = db.execute(
        text(
            """
            SELECT count(*),
                   count(o.misconception_id)
            FROM attempts a
            JOIN questions q ON q.id = a.question_id
            LEFT JOIN question_options o ON o.id = a.selected_option_id
            WHERE a.student_id = :sid AND NOT a.is_correct AND NOT q.faulty
            """
        ),
        {"sid": student_id},
    ).one()

    return {
        "wrong_answers": wrong,
        "wrong_answers_named": named,
        "repeat_threshold": REPEAT_THRESHOLD,
        "repeated": [
            {
                "mistake": r.label,
                "slug": r.slug,
                "concept": r.concept,
                "concept_slug": r.concept_slug,
                "times": r.times,
                "questions": r.questions,
                "last_seen": r.last_seen.isoformat(),
            }
            for r in rows
            if r.questions >= REPEAT_THRESHOLD
        ],
    }
