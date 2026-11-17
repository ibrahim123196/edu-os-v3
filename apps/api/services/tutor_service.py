"""Learning modes: turning a guessed mode into what the chat does.

Practice and test are doorways to the quiz; review is an answer plus the
student's own progress; learn and clarify are answers. Nothing here makes a
quiz or estimates mastery -- it finds the topic and hands over to what
already does.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ai.tutor.modes import Mode
from apps.api.curriculum import Concept, chunk_concepts
from apps.api.quiz import Question, QuestionStatus

log = logging.getLogger(__name__)

PRACTICE_COUNT = 5
TEST_COUNT = 10


@dataclass
class TopicMatch:
    concept: Concept
    how: str          # "name" or "search", for the trace


@dataclass
class ModeReply:
    """What the chat sends back for a quiz doorway: no textbook answer."""

    text: str
    understood_as: str
    action: dict | None = None            # what the page's button does
    trace: dict = field(default_factory=dict)


# --- finding the topic -------------------------------------------------------

def _norm(text: str) -> str:
    """Comparable form: "Coulomb’s law", "coulombs law" and "Coulomb's Law"
    are the same topic to a student typing on a phone."""
    text = text.lower().replace("’", "'").replace("'", "").replace("-", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text)).strip()


def find_concept(db: Session, topic: str) -> TopicMatch | None:
    """The concept a student's words refer to, or None.

    A concept named in the message wins: "quiz me on gauss's law" means
    Gauss's law, and searching could rank its applications above it. The
    longest name wins, so "applications of gauss's law" is not read as
    "Gauss's law". Otherwise the textbook is searched and the concept
    tagged on the best passages is taken -- that is how "capacitors"
    finds Capacitance without a list of synonyms.

    Whether the concept has questions is the caller's business. Choosing
    only among concepts with questions turned "quiz me on X" into a quiz on
    whatever other topic the search ranked next.
    """
    if not topic.strip():
        return None
    concepts = db.scalars(select(Concept)).all()

    wanted = f" {_norm(topic)} "
    named = [c for c in concepts if f" {_norm(c.name)} " in wanted]
    if named:
        return TopicMatch(max(named, key=lambda c: len(c.name)), "name")

    found = _concept_by_search(db, topic)
    return TopicMatch(found, "search") if found else None


def _concept_by_search(db: Session, topic: str) -> Concept | None:
    from ai.embeddings.provider import get_embedding_provider
    from ai.retrieval.hybrid import RetrievalFilters, hybrid_search
    from ai.retrieval.query import to_lexical

    try:
        hits = hybrid_search(
            db,
            query=to_lexical(topic),
            query_embedding=get_embedding_provider().embed_query(topic),
            filters=RetrievalFilters(),
        )[:8]
    except Exception:
        log.warning("Topic search failed for %r", topic, exc_info=True)
        return None
    # Only passages that contain the student's words. A vector search
    # always returns its nearest passages, however far: "photosynthesis"
    # would otherwise vote for whichever physics topic sat closest, and the
    # student would be handed a quiz on something they never asked about.
    hits = [h for h in hits if h.keyword_rank is not None]
    if not hits:
        return None

    rank = {h.chunk_id: i for i, h in enumerate(hits)}
    rows = db.execute(
        select(chunk_concepts.c.chunk_id, chunk_concepts.c.concept_id)
        .where(chunk_concepts.c.chunk_id.in_(list(rank)))
    ).all()
    # Higher passages count for more: the best match says most about the
    # topic, and the eighth may be a neighbouring section.
    votes: dict[uuid.UUID, float] = {}
    for chunk_id, concept_id in rows:
        votes[concept_id] = votes.get(concept_id, 0.0) + 1.0 / (1 + rank[chunk_id])
    if not votes:
        return None
    return db.get(Concept, max(votes, key=votes.get))


def concepts_with_questions(db: Session) -> dict[uuid.UUID, int]:
    """Concept -> number of questions a quiz could use."""
    rows = db.execute(
        select(Question.concept_id, func.count())
        .where(Question.status == QuestionStatus.approved, ~Question.faulty)
        .group_by(Question.concept_id)
    ).all()
    return {cid: n for cid, n in rows}


# --- practice and test: doorways to the quiz --------------------------------

def quiz_reply(db: Session, mode: Mode, topic: str) -> ModeReply:
    """Offer a quiz. The page's button starts it through the quiz that
    already exists, so it is scored and saved like any other."""
    available = concepts_with_questions(db)
    count = TEST_COUNT if mode == Mode.test else PRACTICE_COUNT
    kind = "test" if mode == Mode.test else "practice questions"
    trace: dict = {"topic": topic}

    if not topic:
        total = sum(available.values())
        if not total:
            return ModeReply(
                text="There are no practice questions yet.",
                understood_as=f"{kind.capitalize()} across all topics", trace=trace)
        n = min(count, total)
        return ModeReply(
            text=(f"A {n}-question {'test' if mode == Mode.test else 'quiz'} across "
                  f"all topics, weighted towards the ones you are weakest at. "
                  f"It is scored and saved in your quiz history."),
            understood_as=f"{kind.capitalize()} across all topics",
            action={"kind": "quiz", "concept_slug": None, "concept": None, "count": n},
            trace=trace,
        )

    match = find_concept(db, topic)
    if match is None:
        trace["matched"] = None
        return ModeReply(
            text=(f"I couldn't match \"{topic}\" to a topic in your course. "
                  f"Pick one from the list in the Practise tab."),
            understood_as=f"{kind.capitalize()} on \"{topic}\"",
            action={"kind": "open_practice"},
            trace=trace,
        )

    concept = match.concept
    trace.update(matched=concept.slug, how=match.how)
    if not available.get(concept.id):
        return ModeReply(
            text=(f"There are no practice questions on {concept.name} yet. "
                  f"Pick another topic in the Practise tab."),
            understood_as=f"{kind.capitalize()} on {concept.name}",
            action={"kind": "open_practice"},
            trace=trace,
        )
    n = min(count, available[concept.id])
    short = (f" There are only {available[concept.id]} on this topic so far."
             if available[concept.id] < count else "")
    return ModeReply(
        text=(f"A {n}-question {'test' if mode == Mode.test else 'quiz'} on "
              f"{concept.name}. It is scored and saved in your quiz history.{short}"),
        understood_as=f"{kind.capitalize()} on {concept.name}",
        action={"kind": "quiz", "concept_slug": concept.slug,
                "concept": concept.name, "count": n},
        trace=trace,
    )


# --- review: an answer plus the student's own record -------------------------

REVIEW_REQUEST = (
    "Give me a short revision summary of {topic}: the key ideas, definitions "
    "and formulas I need for the exam, as a short list."
)


def progress_note(db: Session, student_id: uuid.UUID, concept: Concept) -> str:
    """One line from the student's record. Written here, not by the model:
    a number the model was asked to repeat is a number it can change."""
    from apps.api.services.mastery_service import estimates_for

    est = estimates_for(db, student_id, [concept.id]).get(concept.id)
    if est is None:
        return f"Your progress on {concept.name}: not practised yet."
    note = f"Your progress on {concept.name}: {est.describe()}"
    if est.is_reliable and est.is_weak:
        note += " — one to work on."
    return note + "."
