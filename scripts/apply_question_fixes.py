"""Retire faulty or repeated questions, and move misfiled ones.

    python -m scripts.apply_question_fixes           # check, change nothing
    python -m scripts.apply_question_fixes --apply   # do it

Reads content/question_fixes.json. Retiring marks a question rejected,
with the reason: it stops being handed out, and answers already given to it
stay in the history. Moving changes its concept, and every student's mastery
for both concepts is then recomputed, since those answers now count toward
the other topic. All or nothing: one refused line and nothing changes.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from sqlalchemy import select

import apps.api.models  # noqa: F401  (registers every table the flush may touch)
from apps.api.curriculum import Concept
from apps.api.db import session_scope
from apps.api.quiz import Attempt, Question, QuestionOption, QuestionStatus
from apps.api.services.flag_service import mark_faulty, reason_means_faulty
from apps.api.services.mastery_service import recompute

FILE = Path(__file__).resolve().parents[1] / "content" / "question_fixes.json"


def _question(db, option_id: str) -> Question | None:
    option = db.get(QuestionOption, uuid.UUID(option_id))
    return db.get(Question, option.question_id) if option else None


def check(db, fixes: dict) -> tuple[list, list, list[str]]:
    concepts = {c.slug: c for c in db.scalars(select(Concept)).all()}
    retire, move, problems = [], [], []
    seen: set[uuid.UUID] = set()

    for item in fixes["retire"]:
        q = _question(db, item["option_id"])
        if q is None:
            problems.append(f"retire {item['option_id']}: no such question")
        elif q.id in seen:
            problems.append(f"retire {item['option_id']}: listed twice")
        else:
            seen.add(q.id)
            retire.append((q, item["reason"]))

    for item in fixes["move"]:
        q = _question(db, item["option_id"])
        target = concepts.get(item["to"])
        if q is None:
            problems.append(f"move {item['option_id']}: no such question")
        elif q.id in seen:
            problems.append(f"move {item['option_id']}: also being retired")
        elif target is None:
            problems.append(f"move {item['option_id']}: no concept {item['to']!r}")
        else:
            # A question that has moved since the review was written is
            # refused rather than moved again on stale information.
            current = db.get(Concept, q.concept_id)
            if current.slug != item["from"]:
                problems.append(
                    f"move {item['option_id']}: is under {current.slug!r}, review said {item['from']!r}"
                )
            else:
                seen.add(q.id)
                move.append((q, current, target))

    return retire, move, problems


def main(apply: bool) -> int:
    fixes = json.loads(FILE.read_text(encoding="utf-8"))
    with session_scope() as db:
        retire, move, problems = check(db, fixes)
        already = sum(1 for q, _ in retire if q.status != QuestionStatus.approved)
        print(f"{len(retire)} to retire ({already} already not in use), {len(move)} to move")
        for p in problems:
            print(f"  REFUSED {p}")
        if problems:
            print("\nNothing changed: fix the refused lines first.")
            return 1
        if not apply:
            print("\nNothing changed. Re-run with --apply to do it.")
            return 0

        touched = apply_fixes(db, retire, move)
        print(f"\nDone. Mastery recomputed for {touched} student-concept pair(s).")
    return 0


def apply_fixes(db, retire: list, move: list) -> int:
    for q, reason in retire:
        if reason_means_faulty(reason):
            # Its answers stop counting, not only the question.
            mark_faulty(db, q, reason)
        else:
            q.status = QuestionStatus.rejected
            q.rejection_reason = reason

    touched: set[tuple[uuid.UUID, uuid.UUID]] = set()
    for q, old, new in move:
        q.concept_id = new.id
        students = db.scalars(
            select(Attempt.student_id).where(Attempt.question_id == q.id).distinct()
        ).all()
        touched |= {(s, old.id) for s in students} | {(s, new.id) for s in students}

    db.flush()  # the moved questions must be visible to recompute
    for student_id, concept_id in touched:
        recompute(db, student_id, concept_id)
    return len(touched)


if __name__ == "__main__":
    sys.exit(main(apply="--apply" in sys.argv[1:]))
