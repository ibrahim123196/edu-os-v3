"""Review tutor answers that students said look wrong.

    python -m scripts.review_answers                         # list them; change nothing
    python -m scripts.review_answers --fixed 1,3 --note "prompt now says..."
    python -m scripts.review_answers --fine 2                # right after all
    python -m scripts.review_answers --all                   # closed ones too

Each report shows what the student asked, the answer they got, the pages
it cited, and why they think it is wrong. Closing one records what you
decided; it changes nothing the student sees. Numbers refer to the list as
printed; anything unclear changes nothing.
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime, timezone

from sqlalchemy import select

import apps.api.models  # noqa: F401  (registers every table)
from apps.api.answer_reports import REASONS, AnswerReport
from apps.api.db import init_db, session_scope
from apps.api.models import Message
from apps.api.student import Student


def _wrap(text: str, indent: str, limit: int = 900) -> list[str]:
    text = " ".join((text or "").split())
    if len(text) > limit:
        text = text[:limit] + " …"
    return textwrap.wrap(text, width=96, initial_indent=indent,
                         subsequent_indent=" " * len(indent)) or [indent]


def describe(db, n: int, report: AnswerReport) -> list[str]:
    answer = db.get(Message, report.message_id)
    student = db.get(Student, report.student_id)
    # The question is the student's message just before the answer.
    asked = db.scalar(
        select(Message.content)
        .where(Message.conversation_id == answer.conversation_id, Message.role == "user",
               Message.created_at <= answer.created_at)
        .order_by(Message.created_at.desc()).limit(1))
    when = report.created_at.astimezone()
    who = student.display_name if student else "?"
    status = "" if report.resolved_at is None else f"  [closed: {report.outcome}]"
    lines = [f"{n}. {who}, {when:%d %b %H:%M}{status}"]
    lines += _wrap(asked or "(question not found)", "   Asked:  ")
    if report.part:
        lines.append(f"   Part:   {report.part}")
    lines += _wrap(answer.content, "   Answer: ")
    pages = []
    for c in answer.citations or []:
        label = c.get("page_label") or ""
        pages.append(f"{c.get('document_title') or '?'} {label}".strip())
    if pages:
        lines += _wrap("; ".join(dict.fromkeys(pages)), "   Cited:  ")
    reason = REASONS.get(report.reason, report.reason)
    lines.append(f"   Report: {reason}" + (f" -- {report.note}" if report.note else ""))
    if report.review_note:
        lines.append(f"   You:    {report.review_note}")
    return lines


def _numbers(args: list[str], flag: str) -> set[int]:
    if flag not in args:
        return set()
    try:
        return {int(x) for x in args[args.index(flag) + 1].split(",")}
    except (IndexError, ValueError):
        sys.exit(f"{flag} needs numbers from the list, e.g. {flag} 1,3")


def _note(args: list[str]) -> str | None:
    if "--note" not in args:
        return None
    try:
        return args[args.index("--note") + 1]
    except IndexError:
        sys.exit('--note needs some words, e.g. --note "fixed the prompt"')


def main(args: list[str]) -> int:
    fixed, fine = _numbers(args, "--fixed"), _numbers(args, "--fine")
    if fixed & fine:
        sys.exit(f"{sorted(fixed & fine)} cannot be both fixed and fine.")
    show_all = "--all" in args
    init_db()
    with session_scope() as db:
        query = select(AnswerReport).order_by(AnswerReport.created_at)
        if not show_all:
            query = query.where(AnswerReport.resolved_at.is_(None))
        reports = db.scalars(query).all()
        if not reports:
            print("No reported answers." if show_all else "No open reports.")
            return 0
        chosen = fixed | fine
        unknown = sorted(n for n in chosen if not 1 <= n <= len(reports))
        if unknown:
            sys.exit(f"Not in the list: {unknown}. Nothing changed.")
        if not chosen:
            for n, r in enumerate(reports, 1):
                print("\n".join(describe(db, n, r)), end="\n\n")
            print("Close with --fixed N or --fine N (add --note \"...\" to say what you did).")
            return 0
        note = _note(args)
        now = datetime.now(timezone.utc)
        for n in sorted(chosen):
            r = reports[n - 1]
            r.resolved_at, r.outcome = now, "fixed" if n in fixed else "fine"
            if note:
                r.review_note = note
            print(f"{n}. closed as {r.outcome}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
