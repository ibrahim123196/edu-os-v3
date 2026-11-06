"""Document endpoints: upload, list, inspect, delete."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ai.ingestion.extract import SUPPORTED_SUFFIXES
from apps.api.config import settings
from apps.api.db import get_db
from apps.api.models import Document, DocumentChunk, IngestStatus
from apps.api.schemas import DocumentOut
from apps.api.services.ingestion_service import compute_checksum, ingest_document

log = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

CHUNK_BYTES = 1024 * 1024


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    subject: str | None = Form(default=None),
    grade: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"{suffix or 'That file type'} is not supported. "
                f"Upload a PDF, .txt or .md file."
            ),
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    document_id = uuid.uuid4()
    destination = upload_dir / f"{document_id}{suffix}"

    # Stream to disk so a large PDF never sits fully in memory, and enforce
    # the size cap as we go rather than after the fact.
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with destination.open("wb") as out:
            while block := await file.read(CHUNK_BYTES):
                written += len(block)
                if written > limit:
                    out.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"That file is over the {settings.max_upload_mb} MB limit.",
                    )
                out.write(block)
    finally:
        await file.close()

    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="That file is empty.")

    checksum = compute_checksum(destination)
    duplicate = db.scalar(
        select(Document).where(
            Document.checksum == checksum, Document.status == IngestStatus.ready
        )
    )
    if duplicate is not None:
        destination.unlink(missing_ok=True)
        return duplicate

    document = Document(
        id=document_id,
        title=(title or Path(filename).stem).strip()[:500],
        filename=filename,
        content_type=file.content_type or "application/octet-stream",
        storage_path=str(destination),
        byte_size=written,
        checksum=checksum,
        subject=(subject or None),
        grade=(grade or None),
        status=IngestStatus.pending,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    background.add_task(ingest_document, document.id)
    return document


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    return db.scalars(select(Document).order_by(Document.created_at.desc())).all()


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="No such document.")
    return document


@router.post("/{document_id}/reingest", response_model=DocumentOut)
def reingest(
    document_id: uuid.UUID,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Re-run ingestion, e.g. after changing the chunking settings."""
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="No such document.")
    if not Path(document.storage_path).exists():
        raise HTTPException(status_code=410, detail="The original file is no longer on disk.")

    document.status = IngestStatus.pending
    document.error = None
    db.commit()
    db.refresh(document)

    background.add_task(ingest_document, document.id)
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="No such document.")

    Path(document.storage_path).unlink(missing_ok=True)
    db.delete(document)  # chunks cascade
    db.commit()


@router.get("/{document_id}/chunks")
def list_chunks(
    document_id: uuid.UUID,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """Inspect how a document was split. Useful when tuning chunk settings."""
    if db.get(Document, document_id) is None:
        raise HTTPException(status_code=404, detail="No such document.")

    total = db.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
    )
    rows = db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
        .limit(min(limit, 100))
        .offset(offset)
    ).all()

    return {
        "total": total,
        "chunks": [
            {
                "id": str(c.id),
                "index": c.chunk_index,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "section": c.section,
                "word_count": c.word_count,
                "content": c.content,
            }
            for c in rows
        ],
    }
