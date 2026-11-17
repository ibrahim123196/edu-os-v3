"""scripts.retire_unseen_references against a real database. Local only."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete

from apps.api.curriculum import Concept
from apps.api.db import init_db, session_scope
from apps.api.quiz import Question, QuestionKind, QuestionOption, QuestionStatus
from scripts.retire_unseen_references import REASON, main

pytestmark = pytest.mark.integration

TAG = "t" + uuid.uuid4().hex[:5]


@pytest.fixture
def bank():
    init_db()
    with session_scope() as db:
        concept = Concept(slug=f"unseen-{TAG}", name=f"Unseen {TAG}")
        db.add(concept)
        db.flush()
        ids = {}
        for key, stem in (("bad", f"In the solution shown, why does the field at A point right {TAG}?"),
                          ("named", f"In Example 4.2, the magnetic field is along y {TAG}. Why?"),
                          ("good", f"Why do field lines never cross {TAG}?")):
            q = Question(concept_id=concept.id, kind=QuestionKind.multiple_choice, stem=stem,
                         explanation="Because the field has one direction at each point.",
                         status=QuestionStatus.approved)
            q.options = [QuestionOption(position=0, text="A", is_correct=True),
                         QuestionOption(position=1, text="B", is_correct=False)]
            db.add(q)
            db.flush()
            ids[key] = q.id
        ids["concept"] = concept.id
    yield ids
    with session_scope() as db:
        db.execute(delete(Concept).where(Concept.id == ids["concept"]))


def _state(qid):
    with session_scope() as db:
        q = db.get(Question, qid)
        return q.status, q.faulty, q.rejection_reason


def test_listing_changes_nothing(bank, capsys):
    main(apply=False)
    assert f"In the solution shown, why does the field at A point right {TAG}?" in capsys.readouterr().out
    assert _state(bank["bad"])[0] == QuestionStatus.approved


def test_apply_rewords_or_retires_only_the_questions_pointing_at_the_unseen(bank):
    main(apply=True)
    assert _state(bank["bad"]) == (QuestionStatus.rejected, True, REASON)
    assert _state(bank["good"])[:2] == (QuestionStatus.approved, False)
    # Only naming the example: reworded and kept.
    assert _state(bank["named"])[:2] == (QuestionStatus.approved, False)
    with session_scope() as db:
        assert db.get(Question, bank["named"]).stem == f"A magnetic field is along y {TAG}. Why?"


def test_a_second_run_finds_nothing_left(bank, capsys):
    main(apply=True)
    capsys.readouterr()
    main(apply=False)
    assert TAG not in capsys.readouterr().out
