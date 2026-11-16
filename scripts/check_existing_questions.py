"""Run the independent check over questions approved before it existed.

    python -m scripts.check_existing_questions --limit 10  # check 10; write a report; change nothing
    python -m scripts.check_existing_questions             # check all not yet checked
    python -m scripts.check_existing_questions --apply     # store the verdicts from that report
    python -m scripts.check_existing_questions --apply --keep 2,4   # ...overruling numbers 2 and 4

Each check is a paid model call. --limit checks a few first, so the real
cost can be read in the Anthropic Console before checking the rest. After
--apply those count as checked, and the next run moves on to the others.

The first run asks the checker model about every approved question that
has never been checked, and writes question_check_report.md (to read) and
question_check_results.json (what --apply stores). Nothing in the database
changes until --apply, and --apply never calls the model: it stores exactly
the verdicts in the report that was read.

Repeats are judged only against questions written earlier, so of two
copies it is the later one that is flagged, never both.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import selectinload

import apps.api.models  # noqa: F401  (registers every table the flush may touch)
from ai.generation.llm import get_llm
from ai.quiz.checking import check_question
from apps.api.config import settings
from apps.api.curriculum import Concept
from apps.api.db import session_scope
from apps.api.models import DocumentChunk
from apps.api.quiz import Question, QuestionStatus

RESULTS = Path("question_check_results.json")
REPORT = Path("question_check_report.md")


def run_checks(limit: int | None = None) -> None:
    checker = get_llm(settings.check_model) if settings.check_model else get_llm()
    results, failed = [], []

    with session_scope() as db:
        concepts = {c.id: c.name for c in db.scalars(select(Concept)).all()}
        approved = db.scalars(
            select(Question)
            .options(selectinload(Question.options))
            .where(Question.status == QuestionStatus.approved)
            .order_by(Question.concept_id, Question.created_at)
        ).all()
        unchecked = [q for q in approved if q.checked_by is None]
        if limit is not None:
            unchecked = unchecked[:limit]
        print(f"checking {len(unchecked)} of {len(approved)} approved question(s) with {checker.model}")

        for n, q in enumerate(unchecked, 1):
            options = sorted(q.options, key=lambda o: o.position)
            correct = next(i for i, o in enumerate(options) if o.is_correct)
            chunk = db.get(DocumentChunk, q.source_chunk_id) if q.source_chunk_id else None
            earlier = [
                o.stem for o in approved
                if o.concept_id == q.concept_id and o.created_at < q.created_at
            ]
            result = check_question(
                checker,
                concept=concepts[q.concept_id],
                passage=chunk.content if chunk else "",
                stem=q.stem,
                options=[o.text for o in options],
                correct_index=correct,
                existing_stems=earlier,
            )
            results.append({"question_id": str(q.id), "ok": result.ok, "reason": result.reason})
            if not result.ok:
                failed.append((concepts[q.concept_id], q, options, correct, result.reason))
            print(f"  {n}/{len(unchecked)} {'ok' if result.ok else 'FAILED'}")

    RESULTS.write_text(
        json.dumps({"checked_by": checker.model, "results": results}, indent=1), encoding="utf-8"
    )
    lines = [
        "# Independent check of existing questions", "",
        f"Checked by {checker.model}: {len(results)} question(s), {len(failed)} failed. "
        "Nothing has changed yet. If you disagree with a verdict, say which number "
        "before running --apply: the checker is a model and can be wrong too.", "",
    ]
    for k, (concept, q, options, correct, reason) in enumerate(failed, 1):
        lines += [f"**{k}. {concept}**  ", f"*Question:* {q.stem}  "]
        lines += [f"*{'Key' if i == correct else 'Wrong'}:* {o.text}  " for i, o in enumerate(options)]
        lines += [f"*Checker:* {reason}", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{len(failed)} failed. Read {REPORT.resolve()}, then run with --apply.")


def apply_results(keep: set[int] | None = None) -> int:
    """Store the saved verdicts. `keep` holds report numbers a person read
    and overruled: those stay in use, marked as kept on review."""
    keep = keep or set()
    if not RESULTS.exists():
        print("No results yet: run without --apply first.")
        return 1
    saved = json.loads(RESULTS.read_text(encoding="utf-8"))
    # Report numbers count the failures in the order they were written.
    failures = [r for r in saved["results"] if not r["ok"]]
    unknown = sorted(n for n in keep if not 1 <= n <= len(failures))
    if unknown:
        # A mistyped number must not quietly retire the question it meant.
        print(f"No failure numbered {', '.join(map(str, unknown))} in the report: nothing stored.")
        return 1
    overruled = {failures[n - 1]["question_id"] for n in keep}
    with session_scope() as db:
        changed = 0
        for r in saved["results"]:
            q = db.get(Question, uuid.UUID(r["question_id"]))
            # Skip anything that changed since the report was written,
            # rather than store a verdict about a question nobody re-read.
            if q is None or q.status != QuestionStatus.approved or q.checked_by is not None:
                continue
            if r["question_id"] in overruled:
                q.checked_by = f"{saved['checked_by']}, kept on review"[:80]
                changed += 1
                continue
            q.checked_by = saved["checked_by"]
            if not r["ok"]:
                q.status = QuestionStatus.rejected
                q.rejection_reason = r["reason"]
            changed += 1
    print(f"Stored {changed} verdict(s), {len(overruled)} kept on review.")
    return 0


def _keep(args: list[str]) -> set[int]:
    if "--keep" not in args:
        return set()
    try:
        return {int(n) for n in args[args.index("--keep") + 1].split(",")}
    except (IndexError, ValueError):
        sys.exit("--keep needs report numbers, e.g. --keep 2,4")


def _limit(args: list[str]) -> int | None:
    if "--limit" not in args:
        return None
    try:
        return int(args[args.index("--limit") + 1])
    except (IndexError, ValueError):
        # Refused rather than ignored: a mistyped limit falling back to
        # "check everything" would spend what the limit was there to save.
        sys.exit("--limit needs a number, e.g. --limit 10")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--apply" in args:
        sys.exit(apply_results(_keep(args)))
    from ai.generation import usage

    # Printed even when a run stops part-way: what it spent was still spent.
    try:
        run_checks(_limit(args))
    finally:
        if usage.summary():
            print("\n" + usage.summary())
