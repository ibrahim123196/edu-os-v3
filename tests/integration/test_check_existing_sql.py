"""Checking questions approved before the check existed.

Local database only (see conftest).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, select

from apps.api.curriculum import Concept
from apps.api.db import init_db, session_scope
from apps.api.quiz import Question, QuestionKind, QuestionOption, QuestionStatus
from scripts import check_existing_questions as script

pytestmark = pytest.mark.integration

SLUG = f"existing-{uuid.uuid4().hex[:6]}"


class FakeChecker:
    """Agrees with the key, except: a stem containing BADKEY gets option 2,
    and one containing COPY repeats the first earlier question."""

    model = "fake-checker"

    def complete(self, system, messages, max_tokens):
        prompt = messages[0]["content"]
        stem = prompt.split("Question: ", 1)[1].split("\n", 1)[0]
        pick = 1 if "BADKEY" in stem else 0
        return json.dumps({
            "options": [{"correct": i == pick, "why": "-"} for i in range(2)],
            "false_premise": None, "tests_concept": True, "concept_note": "",
            "repeats": 1 if "COPY" in stem else None,
        })


@pytest.fixture
def bank(monkeypatch, tmp_path):
    init_db()
    monkeypatch.chdir(tmp_path)          # the report files land here
    monkeypatch.setattr(script, "get_llm", lambda *a, **k: FakeChecker())
    start = datetime.now(timezone.utc)
    with session_scope() as db:
        concept = Concept(slug=SLUG, name="Existing")
        db.add(concept)
        db.flush()
        for n, stem in enumerate(["GOOD original question?", "BADKEY question?", "COPY of the original?"]):
            q = Question(concept_id=concept.id, kind=QuestionKind.multiple_choice, stem=stem,
                         explanation="b", status=QuestionStatus.approved,
                         created_at=start + timedelta(seconds=n))
            q.options = [QuestionOption(position=0, text="keyed", is_correct=True),
                         QuestionOption(position=1, text="other", is_correct=False)]
            db.add(q)
    yield tmp_path
    with session_scope() as db:
        db.execute(delete(Concept).where(Concept.slug == SLUG))


def _questions():
    with session_scope() as db:
        concept = db.scalar(select(Concept).where(Concept.slug == SLUG))
        return {q.stem.split()[0]: (q.status, q.checked_by, q.rejection_reason)
                for q in db.scalars(select(Question).where(Question.concept_id == concept.id))}


def test_checking_writes_a_report_and_changes_nothing(bank):
    script.run_checks()
    assert all(status == QuestionStatus.approved and checked is None
               for status, checked, _ in _questions().values())
    report = (bank / "question_check_report.md").read_text(encoding="utf-8")
    assert "*Question:* BADKEY" in report and "*Question:* COPY" in report
    # The earlier question is never flagged as a copy of the later one.
    assert "*Question:* GOOD" not in report


def test_apply_stores_exactly_the_reported_verdicts(bank):
    script.run_checks()
    script.apply_results()
    q = _questions()
    assert q["GOOD"] == (QuestionStatus.approved, "fake-checker", None)
    assert q["BADKEY"][0] == QuestionStatus.rejected and "checker says 2" in q["BADKEY"][2]
    assert q["COPY"][0] == QuestionStatus.rejected and "repeats" in q["COPY"][2]


def test_apply_skips_a_question_changed_since_the_report(bank):
    script.run_checks()
    with session_scope() as db:
        concept = db.scalar(select(Concept).where(Concept.slug == SLUG))
        q = db.scalar(select(Question).where(Question.concept_id == concept.id,
                                             Question.stem.like("BADKEY%")))
        q.status = QuestionStatus.draft
    script.apply_results()
    assert _questions()["BADKEY"] == (QuestionStatus.draft, None, None)


def test_a_limit_checks_only_that_many(bank):
    """--limit is how the cost is measured before spending the rest."""
    script.run_checks(limit=1)
    saved = json.loads((bank / "question_check_results.json").read_text(encoding="utf-8"))
    assert len(saved["results"]) == 1
