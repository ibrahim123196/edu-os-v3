"""Keep the mastery estimate current, and answer questions about it.

Mastery is recomputed from the raw attempts each time one lands, rather than
nudged. Nudging a cached number drifts: a bug in the update, a lost write, a
change to the estimator, and the stored value no longer reflects what the
student actually did, with nothing to reconcile it against. Recomputing from
attempts costs one indexed query and keeps the attempts as the only source
of truth -- which matters because the estimator will change, and every
stored value will need rebuilding when it does.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai.mastery.estimate import MasteryEstimate, diagnose, estimate, practice_weight
from apps.api.curriculum import Concept
from apps.api.quiz import Attempt, Question
from apps.api.student import ConceptMastery

log = logging.getLogger(__name__)


def _outcomes(db: Session, student_id: uuid.UUID, concept_id: uuid.UUID) -> list[bool]:
    """A student's answers for one concept, oldest first."""
    return list(
        db.scalars(
            select(Attempt.is_correct)
            .join(Question, Question.id == Attempt.question_id)
            .where(
                Attempt.student_id == student_id,
                Question.concept_id == concept_id,
            )
            .order_by(Attempt.created_at)
        ).all()
    )


def recompute(
    db: Session, student_id: uuid.UUID, concept_id: uuid.UUID
) -> MasteryEstimate:
    """Recompute and store one concept's estimate."""
    est = estimate(_outcomes(db, student_id, concept_id))

    row = db.scalar(
        select(ConceptMastery).where(
            ConceptMastery.student_id == student_id,
            ConceptMastery.concept_id == concept_id,
        )
    )
    if row is None:
        row = ConceptMastery(student_id=student_id, concept_id=concept_id)
        db.add(row)

    row.mastery = est.mastery
    row.confidence = est.confidence
    row.attempts = est.attempts
    row.correct = est.correct
    return est


def record_attempt(db: Session, student_id: uuid.UUID, question: Question) -> None:
    """Update mastery after an answer. Called once the attempt is saved."""
    db.flush()  # the new attempt must be visible to the recompute query
    recompute(db, student_id, question.concept_id)


def estimates_for(
    db: Session, student_id: uuid.UUID, concept_ids: list[uuid.UUID] | None = None
) -> dict[uuid.UUID, MasteryEstimate]:
    """Stored estimates, keyed by concept.

    Concepts with no row are absent rather than zero: never attempted and
    answered badly are different states, and collapsing them would have a
    planner treat an untouched topic as a failed one.
    """
    query = select(ConceptMastery).where(ConceptMastery.student_id == student_id)
    if concept_ids:
        query = query.where(ConceptMastery.concept_id.in_(concept_ids))

    return {
        row.concept_id: MasteryEstimate(
            mastery=row.mastery,
            confidence=row.confidence,
            attempts=row.attempts,
            correct=row.correct,
        )
        for row in db.scalars(query).all()
    }


def progress_report(db: Session, student_id: uuid.UUID) -> dict:
    """Everything known about this student, with causes where they exist."""
    concepts = db.scalars(select(Concept)).all()
    by_id = {c.id: c for c in concepts}
    stored = estimates_for(db, student_id)

    rows = []
    for concept in sorted(concepts, key=lambda c: c.name):
        est = stored.get(concept.id, estimate([]))

        prerequisites = {
            by_id[p.id].name: stored.get(p.id, estimate([]))
            for p in concept.prerequisites
            if p.id in by_id
        }
        finding = diagnose(concept.name, est, prerequisites)

        rows.append(
            {
                "concept": concept.name,
                "slug": concept.slug,
                "mastery": est.mastery,
                "confidence": est.confidence,
                "attempts": est.attempts,
                "correct": est.correct,
                "reliable": est.is_reliable,
                "weak": est.is_weak,
                "summary": est.describe(),
                "likely_cause": finding.blamed_prerequisite,
                "explanation": finding.explanation,
            }
        )

    attempted = [r for r in rows if r["attempts"]]
    weak = [r for r in attempted if r["weak"] and r["reliable"]]

    return {
        "concepts_total": len(rows),
        "concepts_attempted": len(attempted),
        "questions_answered": sum(r["attempts"] for r in rows),
        "overall_mastery": (
            round(sum(r["mastery"] for r in attempted) / len(attempted), 3)
            if attempted
            else None
        ),
        # Weakest first, but only where there is enough evidence to say so.
        "weakest": sorted(weak, key=lambda r: r["mastery"])[:5],
        "untouched": [r["concept"] for r in rows if not r["attempts"]][:10],
        "concepts": rows,
    }


def selection_weights(
    db: Session, student_id: uuid.UUID, concept_ids: list[uuid.UUID]
) -> dict[uuid.UUID, float]:
    """How strongly to favour each concept when choosing questions."""
    stored = estimates_for(db, student_id, concept_ids)
    return {
        concept_id: practice_weight(stored.get(concept_id, estimate([])))
        for concept_id in concept_ids
    }
