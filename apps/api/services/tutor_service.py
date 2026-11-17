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
    """What a student's words refer to: one concept, or a group of them."""

    name: str                 # as the student will see it: "Capacitors"
    concepts: list[Concept]   # in the textbook's order
    how: str                  # "name" or "search", for the trace
    group: str | None = None  # the group's slug, when it is one

    @property
    def slugs(self) -> list[str]:
        return [c.slug for c in self.concepts]


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


def find_topic(db: Session, topic: str) -> TopicMatch | None:
    """The concept or topic group a student's words refer to, or None.

    Names first: every concept name and every group name and alias in
    topic_groups. The longest name found in the message wins, so "parallel
    plate capacitor" is that one concept, while "capacitors" is the group.
    Where a group and a concept share a name -- "Capacitance", "Electric
    field" -- the group wins: a student naming the subject means all of it.

    Otherwise the textbook is searched and the concept tagged on the best
    passages is taken, which finds topics nobody listed a name for.

    Whether there are questions is the caller's business. Choosing only
    among topics with questions turned "quiz me on X" into a quiz on
    whatever other topic the search ranked next.
    """
    from apps.api.topic_groups import CONCEPT_ALIASES, GROUPS

    if not topic.strip():
        return None
    # Typos first, with the same corrector questions get: "capactiors" was
    # once looked up as typed, matched nothing, and the student was told
    # their course had no such topic.
    from ai.retrieval.spelling import correct_query

    topic, _ = correct_query(db, topic)
    concepts = db.scalars(select(Concept)).all()
    by_slug = {c.slug: c for c in concepts}
    wanted = f" {_norm(topic)} "

    # (length of the name matched, group beats concept on a tie, the match)
    found: list[tuple[int, int, TopicMatch]] = []
    for c in concepts:
        for name in (c.name, *CONCEPT_ALIASES.get(c.slug, ())):
            label = _norm(name)
            if f" {label} " in wanted:
                found.append((len(label), 0, TopicMatch(c.name, [c], "name")))
    for g in GROUPS:
        members = [by_slug[m] for m in g.members if m in by_slug]
        if not members:
            continue
        for name in (g.name, *g.aliases):
            label = _norm(name)
            if f" {label} " in wanted:
                found.append((len(label), 1, TopicMatch(g.name, members, "name", g.slug)))
    if found:
        return max(found, key=lambda f: (f[0], f[1]))[2]

    concept = _concept_by_search(db, topic)
    return TopicMatch(concept.name, [concept], "search") if concept else None


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
            action=quiz_action([], None, n, test=mode == Mode.test),
            trace=trace,
        )

    match = find_topic(db, topic)
    if match is None:
        trace["matched"] = None
        return ModeReply(
            text=(f"I couldn't match \"{topic}\" to a topic in your course. "
                  f"Pick one from the list in the Practise tab."),
            understood_as=f"{kind.capitalize()} on \"{topic}\"",
            action={"kind": "open_practice"},
            trace=trace,
        )

    members = [c for c in match.concepts if available.get(c.id)]
    trace.update(matched=match.group or match.slugs[0], how=match.how,
                 covers=[c.slug for c in members])
    if not members:
        return ModeReply(
            text=(f"There are no practice questions on {match.name} yet. "
                  f"Pick another topic in the Practise tab."),
            understood_as=f"{kind.capitalize()} on {match.name}",
            action={"kind": "open_practice"},
            trace=trace,
        )
    total = sum(available[c.id] for c in members)
    n = min(count, total)
    covers = f" It covers {_listing([c.name for c in members])}." if len(members) > 1 else ""
    short = f" There are only {total} on this topic so far." if total < count else ""
    return ModeReply(
        text=(f"A {n}-question {'test' if mode == Mode.test else 'quiz'} on "
              f"{match.name}.{covers} It is scored and saved in your quiz "
              f"history.{short}"),
        understood_as=f"{kind.capitalize()} on {match.name}",
        action=quiz_action(members, match.name, n),
        trace=trace,
    )


def _listing(names: list[str]) -> str:
    """"A", "A and B", "A, B and C"."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def quiz_action(members: list[Concept], label: str | None, count: int,
                test: bool = False) -> dict:
    """What the page's Start button sends to the quiz. test: a timed Test
    me sitting, every topic, rather than a quiz."""
    return {"kind": "quiz", "concept_slugs": [c.slug for c in members],
            "concept": label, "count": count, "test": test}


# --- review: an answer plus the student's own record -------------------------

REVIEW_REQUEST = (
    "Give me a short revision summary of {topic}: the key ideas, definitions "
    "and formulas I need for the exam, as a short list."
)


def progress_note(db: Session, student_id: uuid.UUID, match: TopicMatch) -> str:
    """The student's record on the topic. Written here, not by the model:
    a number the model was asked to repeat is a number it can change."""
    from apps.api.services.mastery_service import estimates_for

    found = estimates_for(db, student_id, [c.id for c in match.concepts])
    if len(match.concepts) == 1:
        est = found.get(match.concepts[0].id)
        if est is None:
            return f"Your progress on {match.name}: not practised yet."
        return f"Your progress on {match.name}: {_described(est)}."

    done = [f"{c.name} {_described(found[c.id])}" for c in match.concepts if c.id in found]
    left = [c.name for c in match.concepts if c.id not in found]
    if not done:
        return f"Your progress on {match.name}: none of its {len(left)} topics practised yet."
    note = f"Your progress on {match.name}: " + "; ".join(done)
    if left:
        note += f". Not practised yet: {_listing(left)}"
    return note + "."


def _described(est) -> str:
    text = est.describe()
    return text + " — one to work on" if est.is_reliable and est.is_weak else text
