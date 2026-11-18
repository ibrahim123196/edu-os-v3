"""The syllabus structure, read once and kept for a few minutes.

Concepts, the chapters that teach them and their prerequisites change only
when the curriculum is seeded. Read on every request, they were four round
trips to a database across the internet before a page could start on the
student's own data -- the Plan and Progress tabs were the slow ones (18
Nov), each paying those trips again on every open.

Kept as plain values, not database objects, so nothing here is tied to a
session that has closed. A re-seed shows up within TTL_SECONDS; a test, or
a script that changes the syllabus, can call clear().
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from apps.api.curriculum import Chapter, Concept, Topic

TTL_SECONDS = 300


@dataclass(frozen=True)
class ChapterInfo:
    id: uuid.UUID
    subject: str
    number: int
    part: str | None
    name: str


@dataclass(frozen=True)
class ConceptInfo:
    id: uuid.UUID
    slug: str
    name: str
    prerequisite_ids: tuple
    home: ChapterInfo | None        # the chapter that first teaches it
    first_topic: str | None         # its first topic number in that chapter, "1.6"


@dataclass(frozen=True)
class Graph:
    concepts: tuple                 # ConceptInfo, by name
    chapters: tuple                 # ChapterInfo with concepts, in syllabus order
    concepts_per_chapter: dict

    def by_id(self) -> dict:
        return {c.id: c for c in self.concepts}


_lock = threading.Lock()
_cached: tuple[float, Graph] | None = None


def clear() -> None:
    global _cached
    with _lock:
        _cached = None


def graph(db: Session) -> Graph:
    global _cached
    with _lock:
        if _cached is not None and time.monotonic() - _cached[0] < TTL_SECONDS:
            return _cached[1]
    built = _build(db)
    with _lock:
        _cached = (time.monotonic(), built)
    return built


def _build(db: Session) -> Graph:
    concepts = db.scalars(select(Concept).options(
        selectinload(Concept.prerequisites),
        selectinload(Concept.topics).selectinload(Topic.chapter).selectinload(Chapter.subject),
    )).all()
    chapters: dict = {}
    per_chapter: dict = {}
    out = []
    for c in sorted(concepts, key=lambda c: c.name):
        homes = [t.chapter for t in c.topics]
        home = min(homes, key=lambda ch: (ch.part or "", ch.number), default=None)
        info = None
        first = None
        if home is not None:
            info = chapters.get(home.id) or ChapterInfo(home.id, home.subject.name, home.number,
                                                        home.part, home.name)
            chapters[home.id] = info
            first = min((t.number for t in c.topics if t.chapter_id == home.id),
                        key=_topic_key, default=None)
        for t in c.topics:
            ch = t.chapter
            chapters.setdefault(ch.id, ChapterInfo(ch.id, ch.subject.name, ch.number, ch.part, ch.name))
            per_chapter.setdefault(ch.id, set()).add(c.id)
        out.append(ConceptInfo(c.id, c.slug, c.name, tuple(p.id for p in c.prerequisites),
                               info, first))
    ordered = tuple(sorted(chapters.values(), key=lambda ch: (ch.subject, ch.part or "", ch.number)))
    return Graph(tuple(out), ordered, {k: len(v) for k, v in per_chapter.items()})


def _topic_key(number: str | None) -> tuple:
    return tuple(int(p) if p.isdigit() else 0 for p in (number or "").split("."))
