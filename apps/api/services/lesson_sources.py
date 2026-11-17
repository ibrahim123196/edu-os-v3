"""Passages a lesson was written from, loaded again by id.

Feedback on a student's own explanation is judged against the same passages
the lesson used -- not a fresh search on the student's words, which could
find different ones and contradict the lesson the student just read.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from ai.retrieval.hybrid import RetrievedChunk
from apps.api.models import Document, DocumentChunk


def load_chunks(db: Session, chunk_ids: list[uuid.UUID]) -> list[RetrievedChunk]:
    """In the order given; passages since deleted are skipped."""
    rows = db.execute(
        select(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.id.in_(chunk_ids))
    ).all() if chunk_ids else []
    by_id = {chunk.id: (chunk, doc) for chunk, doc in rows}
    out = []
    for cid in chunk_ids:
        if cid not in by_id:
            continue
        chunk, doc = by_id[cid]
        out.append(RetrievedChunk(
            chunk_id=chunk.id, document_id=doc.id, document_title=doc.title,
            content=chunk.content, page_start=chunk.page_start, page_end=chunk.page_end,
            section=chunk.section, chunk_index=chunk.chunk_index,
            notation_score=chunk.notation_score, source_kind=doc.source_kind,
            publisher=doc.publisher,
        ))
    return out
