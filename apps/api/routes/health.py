"""Health check that reports what is actually wired up.

Deliberately opinionated: it reports whether the LLM is reachable and how many
chunks are indexed, because the two most common "it's broken" reports are a
missing API key and an empty index.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from ai.generation.llm import get_llm
from apps.api.config import settings
from apps.api.db import get_db
from apps.api.models import Document, DocumentChunk, IngestStatus
from apps.api.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)):
    database = "ok"
    documents_ready = 0
    chunks_indexed = 0
    try:
        db.execute(text("SELECT 1"))
        documents_ready = db.scalar(
            select(func.count())
            .select_from(Document)
            .where(Document.status == IngestStatus.ready)
        )
        chunks_indexed = db.scalar(select(func.count()).select_from(DocumentChunk))
    except Exception as exc:
        database = f"error: {exc}"

    try:
        get_llm()
        llm_ready = True
    except Exception:
        llm_ready = False

    return HealthResponse(
        status="ok" if database == "ok" else "degraded",
        database=database,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        llm_ready=llm_ready,
        documents_ready=documents_ready or 0,
        chunks_indexed=chunks_indexed or 0,
    )
