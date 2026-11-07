"""Relational schema for Milestone 1.

Scope note: this covers the RAG schema (Document, DocumentChunk, Source
provenance) plus a minimal conversation log. The full educational schema
(Exam / Subject / Chapter / Topic / Concept / Question / Mastery) arrives in
later milestones. Documents carry flat curriculum metadata for now so that
retrieval can already filter by subject and grade -- those columns become
foreign keys to the curriculum graph without changing the retrieval code.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from apps.api.config import settings
from apps.api.db import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class IngestStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class Document(Base):
    """A source the student uploaded. One row per file."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    content_type: Mapped[str] = mapped_column(String(120), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    checksum: Mapped[str | None] = mapped_column(String(64), index=True)

    # Flat curriculum metadata, kept for student uploads that are not mapped
    # into the graph. Platform documents use chapter_id instead.
    subject: Mapped[str | None] = mapped_column(String(120), index=True)
    grade: Mapped[str | None] = mapped_column(String(40), index=True)

    # Curriculum link. Null for an unmapped student upload.
    chapter_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chapters.id", ondelete="SET NULL"), index=True
    )

    # Provenance. "platform" documents are searchable by every student;
    # "student_upload" documents only by their owner. Licence travels with
    # the document so a citation can say what it is quoting and under what
    # terms, which the README treats as both a legal and a technical need.
    source_kind: Mapped[str] = mapped_column(
        String(40), default="student_upload", nullable=False, index=True
    )
    license: Mapped[str] = mapped_column(String(40), default="unknown", nullable=False)
    publisher: Mapped[str | None] = mapped_column(String(200))
    edition: Mapped[str | None] = mapped_column(String(80))
    source_url: Mapped[str | None] = mapped_column(String(1000))

    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)

    # Which version of the ingestion pipeline produced this document's
    # chunks. Lets a re-upload reprocess a file that was indexed by older code.
    pipeline_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    status: Mapped[IngestStatus] = mapped_column(
        Enum(IngestStatus, name="ingest_status"),
        default=IngestStatus.pending,
        nullable=False,
        index=True,
    )
    error: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class DocumentChunk(Base):
    """A retrievable passage with the page range needed to cite it."""

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    content: Mapped[str] = mapped_column(Text, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, default=0)

    # Provenance: what the citation points at.
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(400))

    # Denormalised from the parent so metadata filters stay on one table.
    subject: Mapped[str | None] = mapped_column(String(120), index=True)
    grade: Mapped[str | None] = mapped_column(String(40), index=True)

    # How much mathematical or chemical notation this passage holds (0-1).
    # Lets retrieval find the line with the formula on it when a student asks
    # for one, instead of the paragraph that merely mentions formulas.
    notation_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.embedding_dim))

    # Postgres maintains this automatically -- it is the lexical half of
    # hybrid retrieval.
    content_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunk_document_order", "document_id", "chunk_index", unique=True),
    )


class CorpusTerm(Base):
    """Every distinct word across all indexed passages, with its frequency.

    Used to correct misspelled questions against the student's own material
    rather than a general dictionary, which would not know "Kirchhoff" or
    "NCERT". Rebuilt after each ingestion.
    """

    __tablename__ = "corpus_terms"

    term: Mapped[str] = mapped_column(String(100), primary_key=True)
    frequency: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class Conversation(Base):
    """A study chat thread. Single-turn in Milestone 1, multi-turn in M2."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    title: Mapped[str | None] = mapped_column(String(400))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )


class Message(Base):
    """One turn. Assistant turns keep their citations and retrieval trace so
    that answers stay auditable after the fact."""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)

    citations: Mapped[list | None] = mapped_column(JSONB)
    retrieval_trace: Mapped[dict | None] = mapped_column(JSONB)
    grounded: Mapped[bool | None] = mapped_column()
    latency_ms: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
