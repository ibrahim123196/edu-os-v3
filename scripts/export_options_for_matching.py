"""Write every wrong option, with its free-text mistake, to a file. Read-only.

    python -m scripts.export_options_for_matching

Questions made before named mistakes existed describe each wrong option in
their own words. This exports them so each can be matched to the concept's
list once; scripts.apply_misconception_matches then stores the matches.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from apps.api.curriculum import Concept
from apps.api.db import session_scope
from apps.api.quiz import Question

OUT = Path("options_to_match.json")


def main() -> None:
    with session_scope() as db:
        questions = db.scalars(
            select(Question)
            .options(selectinload(Question.options))
            .order_by(Question.concept_id, Question.created_at)
        ).all()
        concepts = {c.id: c.slug for c in db.scalars(select(Concept)).all()}

        rows = []
        for q in questions:
            correct = next((o.text for o in q.options if o.is_correct), None)
            for o in q.options:
                if o.is_correct:
                    continue
                rows.append(
                    {
                        "option_id": str(o.id),
                        "concept": concepts.get(q.concept_id),
                        "status": q.status.value,
                        "stem": q.stem,
                        "correct_answer": correct,
                        "wrong_answer": o.text,
                        "why_tempting": o.misconception,
                        "already_matched": o.misconception_id is not None,
                    }
                )

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} wrong option(s) from {len(questions)} question(s) -> {OUT.resolve()}")


if __name__ == "__main__":
    main()
