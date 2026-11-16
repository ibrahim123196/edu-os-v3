"""Review questions someone flagged as looking wrong.

    python -m scripts.review_flags                 # list them; change nothing
    python -m scripts.review_flags --retire 1,3    # confirmed wrong: retire as faulty
    python -m scripts.review_flags --restore 2     # fine after all: back into quizzes

A flagged question is out of every quiz until it is decided here. Retiring
marks it faulty, so answers already given to it stop counting toward
mastery and mistakes -- someone "wrong" against a wrong key was right.
Numbers refer to the list as printed; anything unclear changes nothing.
"""

from __future__ import annotations

import sys

from sqlalchemy import select
from sqlalchemy.orm import Session

import apps.api.models  # noqa: F401  (registers every table the flush may touch)
from apps.api.curriculum import Concept
from apps.api.db import session_scope
from apps.api.quiz import FLAG_REASONS, Question, QuestionFlag
from apps.api.services.flag_service import open_flags, resolve


def describe(db: Session, n: int, first: QuestionFlag) -> list[str]:
    q = db.get(Question, first.question_id)
    concept = db.get(Concept, q.concept_id)
    flags = [f for f in q_flags(db, q) if f.resolved_at is None]
    # In this computer's time zone: the database keeps UTC, and "14:30" for
    # a flag sent at 20:00 in India read as somebody else's.
    when = first.created_at.astimezone()
    lines = [f"{n}. {concept.name}  (flagged {when:%d %b %H:%M})", f"   Q: {q.stem}"]
    for o in sorted(q.options, key=lambda o: o.position):
        lines.append(f"   {'KEY ' if o.is_correct else '    '} {o.text}")
    for f in flags:
        note = f" -- {f.note}" if f.note else ""
        lines.append(f"   flag: {FLAG_REASONS.get(f.reason, f.reason)}{note}")
    return lines


def q_flags(db: Session, q: Question) -> list[QuestionFlag]:
    return list(db.scalars(select(QuestionFlag).where(QuestionFlag.question_id == q.id)).all())


def _numbers(args: list[str], flag: str) -> set[int]:
    if flag not in args:
        return set()
    try:
        return {int(x) for x in args[args.index(flag) + 1].split(",")}
    except (IndexError, ValueError):
        sys.exit(f"{flag} needs numbers from the list, e.g. {flag} 1,3")


def main(args: list[str]) -> int:
    retire, restore = _numbers(args, "--retire"), _numbers(args, "--restore")
    with session_scope() as db:
        pending = open_flags(db)
        if not pending:
            print("No flagged questions.")
            return 0

        if not retire and not restore:
            for n, f in enumerate(pending, 1):
                print("\n".join(describe(db, n, f)) + "\n")
            print(f"{len(pending)} flagged. Decide with --retire and --restore.")
            return 0

        both = retire & restore
        unknown = sorted(n for n in retire | restore if not 1 <= n <= len(pending))
        if both or unknown:
            # Refused whole: a typo must not retire the question it meant
            # to restore.
            problem = f"in both lists: {sorted(both)}" if both else f"not in the list: {unknown}"
            print(f"Nothing changed -- numbers {problem}.")
            return 1

        changed = 0
        for n in sorted(retire | restore):
            question_id = pending[n - 1].question_id
            stem = db.get(Question, question_id).stem
            changed += resolve(db, question_id, retire=n in retire)
            print(f"{'retired ' if n in retire else 'restored'} {n}: {stem[:70]}")
        print(f"\nDone. Mastery recomputed for {changed} student(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
