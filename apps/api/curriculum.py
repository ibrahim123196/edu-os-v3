"""The curriculum graph.

    Exam ──┐
           ├── Subject ── Chapter ── Topic ──┐
    Exam ──┘                                 │
                                        Concept ── prerequisite ──> Concept

Concepts are shared, not duplicated per exam. NCERT's "Coulomb's law" and
JEE's "Coulomb's law" are the same physics, so they are one node that both
exams link to. The alternative -- a copy per exam -- splits a student's
mastery across identical concepts, so revising for boards teaches the system
nothing about their JEE readiness. Sharing costs an extra join and is very
hard to retrofit, which is why it is decided here rather than later.

A Topic is where a chapter talks about something; a Concept is the thing
itself. One topic usually covers several concepts, and one concept usually
appears in several topics across chapters and subjects, so the link between
them is many-to-many.

Prerequisites sit between concepts, not topics. That is what lets Milestone 4
say a weak Gauss's law score is really an electric field problem.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from apps.api.db import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


# --- association tables --------------------------------------------------

subject_exams = Table(
    "subject_exams",
    Base.metadata,
    Column("subject_id", UUID(as_uuid=True), ForeignKey("subjects.id", ondelete="CASCADE"), primary_key=True),
    Column("exam_id", UUID(as_uuid=True), ForeignKey("exams.id", ondelete="CASCADE"), primary_key=True),
)

topic_concepts = Table(
    "topic_concepts",
    Base.metadata,
    Column("topic_id", UUID(as_uuid=True), ForeignKey("topics.id", ondelete="CASCADE"), primary_key=True),
    Column("concept_id", UUID(as_uuid=True), ForeignKey("concepts.id", ondelete="CASCADE"), primary_key=True),
)

# Which exams assess a concept. This is the sharing: one concept, many exams.
concept_exams = Table(
    "concept_exams",
    Base.metadata,
    Column("concept_id", UUID(as_uuid=True), ForeignKey("concepts.id", ondelete="CASCADE"), primary_key=True),
    Column("exam_id", UUID(as_uuid=True), ForeignKey("exams.id", ondelete="CASCADE"), primary_key=True),
)

# A concept cannot be its own prerequisite; the check is in code, since a
# self-referential many-to-many cannot express it.
concept_prerequisites = Table(
    "concept_prerequisites",
    Base.metadata,
    Column("concept_id", UUID(as_uuid=True), ForeignKey("concepts.id", ondelete="CASCADE"), primary_key=True),
    Column("prerequisite_id", UUID(as_uuid=True), ForeignKey("concepts.id", ondelete="CASCADE"), primary_key=True),
)

# Which concepts a retrieved passage covers. Populated by tagging, and what
# lets retrieval filter by concept rather than by a flat subject string.
chunk_concepts = Table(
    "chunk_concepts",
    Base.metadata,
    Column("chunk_id", UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), primary_key=True),
    Column("concept_id", UUID(as_uuid=True), ForeignKey("concepts.id", ondelete="CASCADE"), primary_key=True),
    Column("confidence", Float, nullable=False, server_default="1.0"),
)


# --- nodes ---------------------------------------------------------------

class Exam(Base):
    """JEE Main, NEET, CBSE Class 12."""

    __tablename__ = "exams"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    code: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    grade: Mapped[str | None] = mapped_column(String(20))

    subjects: Mapped[list["Subject"]] = relationship(
        secondary=subject_exams, back_populates="exams"
    )
    concepts: Mapped[list["Concept"]] = relationship(
        secondary=concept_exams, back_populates="exams"
    )


class Subject(Base):
    __tablename__ = "subjects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    code: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)

    exams: Mapped[list[Exam]] = relationship(
        secondary=subject_exams, back_populates="subjects"
    )
    chapters: Mapped[list["Chapter"]] = relationship(
        back_populates="subject", cascade="all, delete-orphan", order_by="Chapter.number"
    )


class Chapter(Base):
    __tablename__ = "chapters"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    subject_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False
    )
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    # NCERT Physics is split into Part I and Part II.
    part: Mapped[str | None] = mapped_column(String(20))

    subject: Mapped[Subject] = relationship(back_populates="chapters")
    topics: Mapped[list["Topic"]] = relationship(
        back_populates="chapter", cascade="all, delete-orphan", order_by="Topic.number"
    )

    __table_args__ = (UniqueConstraint("subject_id", "number", "part", name="uq_chapter_number"),)


class Topic(Base):
    """A numbered section of a chapter: "1.5 Coulomb's Law"."""

    __tablename__ = "topics"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    chapter_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False
    )
    number: Mapped[str] = mapped_column(String(20), nullable=False)  # "1.5", "1.14.1"
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)

    chapter: Mapped[Chapter] = relationship(back_populates="topics")
    concepts: Mapped[list["Concept"]] = relationship(
        secondary=topic_concepts, back_populates="topics"
    )


class Concept(Base):
    """A single idea, shared across every exam and topic that uses it."""

    __tablename__ = "concepts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    topics: Mapped[list[Topic]] = relationship(
        secondary=topic_concepts, back_populates="concepts"
    )
    exams: Mapped[list[Exam]] = relationship(
        secondary=concept_exams, back_populates="concepts"
    )
    prerequisites: Mapped[list["Concept"]] = relationship(
        secondary=concept_prerequisites,
        primaryjoin=id == concept_prerequisites.c.concept_id,
        secondaryjoin=id == concept_prerequisites.c.prerequisite_id,
        backref="unlocks",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# --- provenance ----------------------------------------------------------

class SourceKind(str, enum.Enum):
    """Where a document came from, which decides who can search it."""

    student_upload = "student_upload"
    platform = "platform"


class License(str, enum.Enum):
    """The provenance ladder from the README, strictest first."""

    public_curriculum = "public_curriculum"   # NCERT and other government material
    permitted = "permitted"                   # cleared for use by its publisher
    licensed = "licensed"                     # paid for
    original = "original"                     # written for EduOS
    unknown = "unknown"                       # a student's own upload
