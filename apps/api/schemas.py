"""Request and response contracts."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    filename: str
    subject: str | None = None
    grade: str | None = None
    page_count: int
    chunk_count: int
    byte_size: int
    status: str
    error: str | None = None
    created_at: datetime
    ingested_at: datetime | None = None


class CitationOut(BaseModel):
    marker: str
    chunk_id: str
    document_id: str
    document_title: str
    page_start: int | None = None
    page_end: int | None = None
    page_label: str
    section: str | None = None
    snippet: str
    source_kind: str = "student_upload"
    publisher: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    document_ids: list[uuid.UUID] = Field(
        default_factory=list,
        description="Restrict the search to these documents. Empty means all.",
    )
    subject: str | None = None
    grade: str | None = None
    conversation_id: uuid.UUID | None = None
    include_trace: bool = Field(
        default=False, description="Return the per-stage retrieval trace."
    )


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    grounded: bool
    used_source_count: int
    retrieved_count: int
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    latency_ms: float
    corrections: list[dict] = Field(default_factory=list)
    corrected_question: str | None = None
    # A follow-up rewritten into the full question that was searched.
    searched_as: str | None = None
    # The learning mode guessed from the message, and the one line the page
    # shows for it ("Practice questions on Gauss's law"). For a plain
    # question this is the follow-up rewrite, if there was one.
    mode: str = "clarify"
    understood_as: str | None = None
    # A button for the page: start a quiz, or open the Practise tab.
    action: dict | None = None
    # Review: the student's record on the topic, from the database.
    progress_note: str | None = None
    # Why the answer was pitched simpler or deeper, when it was.
    pitch_note: str | None = None
    # A known mistake the student's message stated, which the answer
    # corrects: {"slug": ..., "label": ...}.
    misconception: dict | None = None
    trace: dict | None = None


class HealthResponse(BaseModel):
    status: str
    database: str
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    llm_provider: str
    llm_model: str
    llm_ready: bool
    documents_ready: int
    chunks_indexed: int
    platform_documents: int = 0
