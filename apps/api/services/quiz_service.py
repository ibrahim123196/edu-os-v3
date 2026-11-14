"""Generate, store and select questions.

Generation picks a passage, asks for a question about it, validates the
result, and stores it either approved or rejected. A rejected question is
kept rather than thrown away: without it the same bad question is generated
again next week and nobody notices it is a pattern.

Selection never generates. A student waiting on a model call before they can
answer anything is a worse experience than a smaller bank, and a question
that has been through validation is worth more than a fresh one that has
not.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ai.generation.llm import LLMError, get_llm
from ai.quiz.generation import GenerationRefused, generate_question
from apps.api.curriculum import Concept, chunk_concepts
from apps.api.models import Document, DocumentChunk
from apps.api.quiz import (
    Attempt,
    Difficulty,
    Question,
    QuestionKind,
    QuestionOption,
    QuestionStatus,
)

log = logging.getLogger(__name__)

# Passages below this are usually a heading, a caption or a stray line, and
# produce questions about nothing.
MIN_PASSAGE_WORDS = 60


@dataclass
class GenerationOutcome:
    concept: str
    approved: int = 0
    rejected: int = 0
    refused: int = 0
    reasons: list[str] = None

    def __post_init__(self) -> None:
        self.reasons = self.reasons or []


def candidate_passages(
    db: Session, concept_id: uuid.UUID, limit: int = 8
) -> list[tuple[DocumentChunk, str, str]]:
    """Passages that teach a concept, most substantial first.

    Ordering by length alone picked the longest chunk tagged with the
    concept, which for Coulomb's law is the biographical note about
    Coulomb's torsion balance -- and the questions that came back asked why
    a sensitive instrument was needed in 1785. Structurally valid, useless
    for revision.

    Notation is the better signal for physics: the passage carrying the
    formula is the one worth asking about. Length still breaks ties, and
    still rules out figure captions, which produce exactly the questions
    you would expect.
    """
    rows = db.execute(
        select(DocumentChunk, Document.title)
        .join(chunk_concepts, chunk_concepts.c.chunk_id == DocumentChunk.id)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            chunk_concepts.c.concept_id == concept_id,
            DocumentChunk.word_count >= MIN_PASSAGE_WORDS,
        )
        .order_by(
            DocumentChunk.notation_score.desc(),
            DocumentChunk.word_count.desc(),
        )
        .limit(limit)
    ).all()

    return [(chunk, title, chunk.section or "") for chunk, title in rows]


def generate_for_concept(
    db: Session,
    concept: Concept,
    count: int = 1,
    difficulty: int = Difficulty.understanding.value,
) -> GenerationOutcome:
    """Generate up to `count` questions about one concept."""
    outcome = GenerationOutcome(concept=concept.name)
    passages = candidate_passages(db, concept.id)

    if not passages:
        outcome.reasons.append("no passage long enough teaches this concept")
        return outcome

    llm = get_llm()

    # Everything already written for this concept, so the model is not
    # asked to invent a second question in ignorance of the first.
    existing_stems = list(
        db.scalars(
            select(Question.stem).where(Question.concept_id == concept.id)
        ).all()
    )

    for chunk, title, section in passages:
        if outcome.approved >= count:
            break

        location = f"page {chunk.page_start}" if chunk.page_start else ""
        if section:
            location = f"{location}, {section}" if location else section

        try:
            generated = generate_question(
                llm,
                content=chunk.content,
                concept=concept.name,
                title=title,
                location=location,
                difficulty=difficulty,
                existing_stems=existing_stems,
            )
        except GenerationRefused as exc:
            outcome.refused += 1
            outcome.reasons.append(str(exc)[:90])
            continue
        except LLMError as exc:
            log.warning("Generation failed for %s: %s", concept.name, exc)
            outcome.reasons.append(f"model error: {exc}"[:90])
            break

        question = Question(
            concept_id=concept.id,
            source_chunk_id=chunk.id,
            kind=QuestionKind.multiple_choice,
            stem=generated.stem,
            explanation=generated.explanation,
            difficulty=difficulty,
            generated_by=llm.model,
            status=(
                QuestionStatus.approved
                if generated.validation.ok
                else QuestionStatus.rejected
            ),
            rejection_reason=(
                None if generated.validation.ok else generated.validation.reason
            ),
        )
        for position, text in enumerate(generated.options):
            question.options.append(
                QuestionOption(
                    position=position,
                    text=text,
                    is_correct=(position == generated.correct_index),
                    misconception=(
                        generated.misconceptions[position]
                        if position < len(generated.misconceptions)
                        else None
                    ),
                )
            )
        db.add(question)

        existing_stems.append(generated.stem)

        if generated.validation.ok:
            outcome.approved += 1
        else:
            outcome.rejected += 1
            outcome.reasons.append(generated.validation.reason[:90])

    return outcome


def select_questions(
    db: Session,
    concept_ids: list[uuid.UUID] | None = None,
    count: int = 5,
    difficulty: int | None = None,
    exclude: list[uuid.UUID] | None = None,
    student_id: uuid.UUID | None = None,
) -> list[Question]:
    """Pick approved questions to put in front of a student.

    With a student, weak and untouched concepts are favoured. The weighting
    is probabilistic rather than a strict weakest-first ordering: sorting by
    mastery hands the student the same quiz every time and stops them ever
    revisiting something they have learnt, which is how knowledge decays
    unnoticed.

    Weakness and ignorance both earn practice. A concept with no attempts
    ranks just below the weakest known one -- finding out is nearly as
    valuable as remediating, and a topic never seen can never be shown to
    be weak.
    """
    query = select(Question).where(Question.status == QuestionStatus.approved)

    if concept_ids:
        query = query.where(Question.concept_id.in_(concept_ids))
    if difficulty is not None:
        query = query.where(Question.difficulty == difficulty)
    if exclude:
        query = query.where(Question.id.notin_(exclude))

    if student_id is None:
        return list(db.scalars(query.order_by(func.random()).limit(count)).all())

    # Oversample, then weight in Python. Expressing the weighting in SQL
    # would need the mastery table joined and multiplied against random(),
    # which is harder to read and harder to change than this.
    pool = list(db.scalars(query.order_by(func.random()).limit(count * 6)).all())
    if len(pool) <= count:
        return pool

    from apps.api.services.mastery_service import selection_weights

    weights = selection_weights(
        db, student_id, list({q.concept_id for q in pool})
    )

    chosen: list[Question] = []
    remaining = list(pool)
    while remaining and len(chosen) < count:
        scores = [weights.get(q.concept_id, 1.0) for q in remaining]
        total = sum(scores) or 1.0
        pick = random.random() * total
        running = 0.0
        for index, score in enumerate(scores):
            running += score
            if running >= pick:
                chosen.append(remaining.pop(index))
                break
        else:
            chosen.append(remaining.pop())
    return chosen


def grade(
    db: Session,
    question: Question,
    selected_option_id: uuid.UUID | None,
    seconds_taken: float | None = None,
    *,
    student_id: uuid.UUID,
) -> dict:
    """Record an answer and explain it.

    The explanation is returned whether the answer was right or wrong. A
    student who guessed correctly needs it as much as one who did not, and
    withholding it on a correct answer rewards guessing.
    """
    correct_option = next((o for o in question.options if o.is_correct), None)
    is_correct = bool(
        correct_option and selected_option_id and correct_option.id == selected_option_id
    )

    chosen = next(
        (o for o in question.options if o.id == selected_option_id), None
    )

    # No default student. A fallback here is how test clicks once landed on
    # the real student: a caller that forgets to say whose answer this is
    # should fail loudly, not guess.
    db.add(
        Attempt(
            student_id=student_id,
            question_id=question.id,
            selected_option_id=selected_option_id,
            is_correct=is_correct,
            seconds_taken=seconds_taken,
        )
    )

    # Mastery is recomputed from the attempts rather than nudged, so the
    # stored estimate can never drift away from what the student did.
    from apps.api.services.mastery_service import record_attempt

    record_attempt(db, student_id, question)

    return {
        "correct": is_correct,
        "correct_option_id": str(correct_option.id) if correct_option else None,
        "explanation": question.explanation,
        # What this particular wrong answer suggests the student was
        # thinking. This is the whole reason distractors are written
        # deliberately rather than filled with plausible noise.
        "misconception": (
            chosen.misconception if chosen and not is_correct else None
        ),
        "source_chunk_id": (
            str(question.source_chunk_id) if question.source_chunk_id else None
        ),
    }


def shuffled(question: Question) -> list[QuestionOption]:
    """Options in a random order, so position carries no information."""
    options = list(question.options)
    random.shuffle(options)
    return options
