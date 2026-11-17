"""Fix or retire approved questions whose stem points at something not shown.

    python -m scripts.retire_unseen_references           # list what would change; change nothing
    python -m scripts.retire_unseen_references --apply   # do it

"In the solution shown, the field at A points right..." was written from a
passage holding a worked example, and asked on its own it refers to a
solution the student never sees. The validator now refuses such stems for
new questions; this deals with the ones approved before it did.

Most only name the example ("In Example 1.2, a cup of water ...") and carry
every number they need: those are reworded, by hand below, with the
reference taken out and the physics unchanged. The rest depend on a figure
or a working the student cannot see, and are retired as faulty, so answers
already given to them stop counting toward mastery.

All or nothing: a rewording that matches more than one question, or no
longer fits its own, or still points at something unseen, stops the run before anything changes.
"""

from __future__ import annotations

import sys

from sqlalchemy import select

import apps.api.models  # noqa: F401  (registers every table the flush may touch)
from ai.quiz.validation import refers_to_unseen
from apps.api.curriculum import Concept
from apps.api.db import session_scope
from apps.api.quiz import Question, QuestionStatus
from apps.api.services.flag_service import mark_faulty

REASON = "faulty (review): stem refers to a figure, example or solution the student cannot see"

# Reviewed one by one (17 Nov 2026). Each: words that pick out the question,
# then the replacements that make it stand on its own.
REWRITES = [
    ("In the network described in Example 2.9,",
     [("In the network described in Example 2.9, capacitors", "In a network, capacitors")]),
    ("In Example 1.1, electrons move",
     [("In Example 1.1, electrons move", "Electrons move")]),
    ("In Example 1.2, a cup of water",
     [("In Example 1.2, a cup of water", "A cup of water")]),
    ("In the solution, the electric field",
     [("In the solution, the electric field ", "The electric field "),
      (" at point A due to the positive charge", " at a point A due to a positive point charge")]),
    ("In Example 1.7, the acceleration of the proton",
     [("In Example 1.7, the acceleration of the proton in the electric field is calculated to be",
       "A proton in a uniform electric field E has an acceleration of")]),
    ("In Example 1.10, the electric field E_x = ax",
     [("In Example 1.10, the electric field E_x = ax has different magnitudes at the left face "
       "(x = a) and right face (x = 2a) of the cube.",
       "A cube of side a lies in an electric field E_x = ax, with its left face at x = a and its "
       "right face at x = 2a. The field has different magnitudes at the two faces.")]),
    ("In the platinum resistance thermometer of Example 3.4,",
     [("In the platinum resistance thermometer of Example 3.4,",
       "In a platinum resistance thermometer,")]),
    ("In Example 4.2, the magnetic field",
     [("In Example 4.2, the magnetic field", "A magnetic field")]),
]


def plan(questions):
    """(question, new stem) to reword, questions to retire, and problems."""
    reword, problems = [], []
    by_id = {}
    for marker, changes in REWRITES:
        matches = [q for q in questions if marker in q.stem]
        if not matches:
            continue                      # reworded already, or not in this bank
        if len(matches) > 1:
            problems.append(f"{marker!r}: {len(matches)} questions match, expected 1")
            continue
        q = matches[0]
        stem = q.stem
        for old, new in changes:
            if old not in stem:
                problems.append(f"{marker!r}: {old!r} not in the stem")
                break
            stem = stem.replace(old, new, 1)
        else:
            if refers_to_unseen(stem):
                problems.append(f"{marker!r}: still points at something unseen: {stem}")
            else:
                reword.append((q, stem))
                by_id[q.id] = q
    retire = [q for q in questions if q.id not in by_id]
    return reword, retire, problems


def main(apply: bool) -> int:
    with session_scope() as db:
        names = {c.id: c.name for c in db.scalars(select(Concept)).all()}
        approved = db.scalars(
            select(Question).where(Question.status == QuestionStatus.approved, ~Question.faulty)
        ).all()
        found = [q for q in approved if refers_to_unseen(q.stem)]
        print(f"{len(found)} of {len(approved)} approved question(s) point at something not shown.\n")
        if not found:
            return 0
        reword, retire, problems = plan(found)
        if problems:
            print("Stopped, nothing changed:")
            for p in problems:
                print(f"  {p}")
            return 1

        print(f"Reword {len(reword)} (the reference taken out, the physics unchanged):\n")
        for q, stem in reword:
            print(f"  [{names.get(q.concept_id, '?')}]\n    was: {q.stem}\n    now: {stem}\n")
        print(f"Retire {len(retire)} (they need a figure or working the student cannot see):\n")
        for q in retire:
            print(f"  [{names.get(q.concept_id, '?')}] {q.stem}\n")

        if not apply:
            print("Nothing changed. Re-run with --apply to do it.")
            return 0
        for q, stem in reword:
            q.stem = stem
        students = sum(mark_faulty(db, q, REASON) for q in retire)
        print(f"Reworded {len(reword)}, retired {len(retire)}. Mastery recomputed {students} time(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main("--apply" in sys.argv[1:]))
