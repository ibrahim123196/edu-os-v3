"""Load content/misconceptions.json into the database. Safe to re-run.

    python -m scripts.seed_misconceptions

Run it after seed_curriculum (the mistakes hang off concepts), and again
whenever the file is edited. Reworded labels are updated in place. A
mistake removed from the file is deleted only if no question uses it;
otherwise it is kept and named, because deleting it would quietly strip
the label from every option pointing at it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from apps.api.curriculum import Concept
from apps.api.db import session_scope
from apps.api.quiz import Misconception, QuestionOption

FILE = Path(__file__).resolve().parents[1] / "content" / "misconceptions.json"


@dataclass
class LoadReport:
    added: int = 0
    updated: int = 0
    removed: int = 0
    kept_in_use: list[str] = field(default_factory=list)


class UnknownConcepts(Exception):
    pass


def load_misconceptions(db: Session, data: dict) -> LoadReport:
    concepts = {c.slug: c for c in db.scalars(select(Concept)).all()}

    # Refuse the whole file rather than load part of it: a typo in a
    # concept slug would otherwise drop that concept's mistakes silently.
    unknown = sorted(set(data["concepts"]) - set(concepts))
    if unknown:
        raise UnknownConcepts(
            f"not in the curriculum: {', '.join(unknown)} -- run seed_curriculum first?"
        )

    report = LoadReport()
    existing = {m.slug: m for m in db.scalars(select(Misconception)).all()}
    wanted: set[str] = set()

    for concept_slug, items in data["concepts"].items():
        concept = concepts[concept_slug]
        for item in items:
            slug, label = item["id"], item["label"].strip()
            wanted.add(slug)
            current = existing.get(slug)
            if current is None:
                db.add(Misconception(slug=slug, concept_id=concept.id, label=label))
                report.added += 1
            elif current.label != label or current.concept_id != concept.id:
                current.label, current.concept_id = label, concept.id
                report.updated += 1

    for slug, misconception in existing.items():
        if slug in wanted:
            continue
        in_use = db.scalar(
            select(func.count())
            .select_from(QuestionOption)
            .where(QuestionOption.misconception_id == misconception.id)
        )
        if in_use:
            report.kept_in_use.append(f"{slug} (used by {in_use} option(s))")
        else:
            db.delete(misconception)
            report.removed += 1

    db.flush()
    return report


def main() -> None:
    data = json.loads(FILE.read_text(encoding="utf-8"))
    with session_scope() as db:
        report = load_misconceptions(db, data)
        total = db.scalar(select(func.count()).select_from(Misconception))
    print(
        f"misconceptions: {total} "
        f"({report.added} added, {report.updated} updated, {report.removed} removed)"
    )
    for line in report.kept_in_use:
        print(f"  kept, no longer in the file but still in use: {line}")


if __name__ == "__main__":
    main()
