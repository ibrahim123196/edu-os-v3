"""Retiring and moving questions against a real database.

Local database only (see conftest): these create and delete rows.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete, select

from apps.api.curriculum import Concept
from apps.api.db import init_db, session_scope
from apps.api.quiz import Question, QuestionKind, QuestionOption, QuestionStatus
from apps.api.services.quiz_service import grade, select_questions
from apps.api.student import ConceptMastery, TEST_STUDENT_ID
from scripts.apply_question_fixes import apply_fixes, check

pytestmark = pytest.mark.integration

TAG = uuid.uuid4().hex[:6]
FROM, TO = f"fix-from-{TAG}", f"fix-to-{TAG}"


@pytest.fixture
def two_concepts():
    init_db()
    with session_scope() as db:
        db.add_all([Concept(slug=FROM, name="From"), Concept(slug=TO, name="To")])
    yield
    with session_scope() as db:
        ids = select(Concept.id).where(Concept.slug.in_([FROM, TO]))
        db.execute(delete(Question).where(Question.concept_id.in_(ids)))
        db.execute(delete(Concept).where(Concept.slug.in_([FROM, TO])))


def _question(db, slug: str) -> tuple[Question, str]:
    """A question and the id of its wrong option -- how the review names it."""
    concept = db.scalar(select(Concept).where(Concept.slug == slug))
    q = Question(concept_id=concept.id, kind=QuestionKind.multiple_choice,
                 stem=f"Q {uuid.uuid4().hex[:4]}?", explanation="b",
                 status=QuestionStatus.approved)
    q.options = [QuestionOption(position=0, text="right", is_correct=True),
                 QuestionOption(position=1, text="wrong", is_correct=False)]
    db.add(q)
    db.flush()
    return q, str(q.options[1].id)


def _mastery(db, slug: str) -> ConceptMastery | None:
    concept = db.scalar(select(Concept).where(Concept.slug == slug))
    return db.scalar(select(ConceptMastery).where(
        ConceptMastery.student_id == TEST_STUDENT_ID,
        ConceptMastery.concept_id == concept.id))


def test_a_retired_question_is_no_longer_handed_out(two_concepts):
    with session_scope() as db:
        q, option = _question(db, FROM)
        retire, move, problems = check(db, {"retire": [{"option_id": option, "reason": "faulty: x"}],
                                            "move": []})
        assert problems == []
        apply_fixes(db, retire, move)
        concept = db.scalar(select(Concept).where(Concept.slug == FROM))
        assert q.status == QuestionStatus.rejected
        assert q.rejection_reason == "faulty: x"
        assert q not in select_questions(db, concept_ids=[concept.id], count=10,
                                         student_id=TEST_STUDENT_ID)


def test_a_moved_question_takes_its_answers_with_it(two_concepts):
    """And leaves no zero row behind: an empty row would make the old
    concept look attempted."""
    with session_scope() as db:
        q, option = _question(db, FROM)
        grade(db, q, q.options[0].id, student_id=TEST_STUDENT_ID)
        db.flush()  # sessions do not autoflush; read what was just written
        assert _mastery(db, FROM).attempts == 1 and _mastery(db, TO) is None

        retire, move, problems = check(db, {"retire": [], "move": [
            {"option_id": option, "from": FROM, "to": TO}]})
        assert problems == []
        apply_fixes(db, retire, move)
        db.flush()

        assert _mastery(db, FROM) is None
        assert _mastery(db, TO).attempts == 1


def test_stale_and_conflicting_lines_are_refused(two_concepts):
    with session_scope() as db:
        _, a = _question(db, FROM)
        _, b = _question(db, FROM)
        _, c = _question(db, TO)
        _, _, problems = check(db, {
            "retire": [{"option_id": a, "reason": "r"}, {"option_id": a, "reason": "r"}],
            "move": [
                {"option_id": a, "from": FROM, "to": TO},          # also being retired
                {"option_id": b, "from": FROM, "to": "nowhere"},   # no such concept
                {"option_id": c, "from": FROM, "to": TO},          # already elsewhere
                {"option_id": str(uuid.uuid4()), "from": FROM, "to": TO},
            ],
        })
        assert len(problems) == 5
        assert any("listed twice" in p for p in problems)
        assert any("also being retired" in p for p in problems)
        assert any("no concept" in p for p in problems)
        assert any("review said" in p for p in problems)
        assert any("no such question" in p for p in problems)
