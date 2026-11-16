"""Store matches between existing wrong options and named mistakes.

    python -m scripts.apply_misconception_matches           # check, change nothing
    python -m scripts.apply_misconception_matches --apply   # store them

Reads content/misconception_matches.json: option id -> mistake id, or null
where none of the concept's mistakes fits. All or nothing: if any match is
refused, nothing is stored, so a half-applied file never leaves the bank in
a state nobody chose.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from sqlalchemy import select

from apps.api.db import session_scope
from apps.api.quiz import Misconception, Question, QuestionOption

FILE = Path(__file__).resolve().parents[1] / "content" / "misconception_matches.json"


def check(db, matches: dict[str, str | None]) -> tuple[list[tuple], list[str]]:
    """Every match that can be stored, and every reason one cannot."""
    by_slug = {m.slug: m for m in db.scalars(select(Misconception)).all()}
    ready, problems = [], []

    for option_id, slug in matches.items():
        option = db.get(QuestionOption, uuid.UUID(option_id))
        if option is None:
            problems.append(f"{option_id}: no such option (question deleted?)")
            continue
        if option.is_correct:
            problems.append(f"{option_id}: is a correct answer, not a mistake")
            continue
        if slug is None:
            ready.append((option, None))
            continue
        misconception = by_slug.get(slug)
        if misconception is None:
            problems.append(f"{option_id}: unknown mistake {slug!r}")
            continue
        # A mistake from another concept would put the error under the
        # wrong topic on the progress page.
        question = db.get(Question, option.question_id)
        if misconception.concept_id != question.concept_id:
            problems.append(f"{option_id}: {slug!r} belongs to a different concept")
            continue
        ready.append((option, misconception))

    return ready, problems


def main(apply: bool) -> int:
    matches = json.loads(FILE.read_text(encoding="utf-8"))["matches"]
    with session_scope() as db:
        ready, problems = check(db, matches)
        named = sum(1 for _, m in ready if m is not None)
        print(f"{len(matches)} match(es) in the file")
        print(f"  {named} to a named mistake, {len(ready) - named} to none of them")
        for p in problems:
            print(f"  REFUSED {p}")

        if problems:
            print("\nNothing stored: fix the refused lines first.")
            return 1
        if not apply:
            print("\nNothing changed. Re-run with --apply to store them.")
            return 0

        for option, misconception in ready:
            option.misconception_id = misconception.id if misconception else None
        print("\nStored.")
    return 0


if __name__ == "__main__":
    sys.exit(main(apply="--apply" in sys.argv[1:]))
