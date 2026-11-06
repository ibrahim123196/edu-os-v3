"""Ingestion pipeline: file on disk -> embedded, searchable chunks.

Runs as a background task. The upload endpoint returns immediately with a
document in `pending`, and the student watches it move to `ready`. Embedding a
200-page PDF takes long enough that doing it inside the request would time out
the browser.

Failures are recorded on the document row rather than raised into the void, so
the UI can tell the student what went wrong and what to do about it.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete

from ai.embeddings.provider import get_embedding_provider
from ai.ingestion.chunking import chunk_pages
from ai.ingestion.extract import EmptyDocument, UnsupportedFileType, extract
from ai.ingestion.notation import notation_score
from ai.retrieval.spelling import rebuild_vocabulary
from apps.api.config import settings
from apps.api.db import session_scope
from apps.api.models import Document, DocumentChunk, IngestStatus

log = logging.getLogger(__name__)


def compute_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_document(document_id) -> None:
    """Extract, chunk, embed and store. Safe to re-run: chunks are replaced."""
    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            log.error("Ingestion requested for unknown document %s", document_id)
            return
        doc.status = IngestStatus.processing
        doc.error = None
        db.commit()
        path = Path(doc.storage_path)
        subject, grade = doc.subject, doc.grade

    try:
        pages = extract(path)

        chunks = chunk_pages(
            pages,
            target_words=settings.chunk_target_words,
            overlap_words=settings.chunk_overlap_words,
            min_words=settings.chunk_min_words,
        )
        if not chunks:
            raise EmptyDocument("The file contained no text worth indexing.")

        provider = get_embedding_provider()
        vectors: list[list[float]] = []
        batch = settings.embedding_batch_size
        for i in range(0, len(chunks), batch):
            texts = [c.content for c in chunks[i : i + batch]]
            vectors.extend(provider.embed_documents(texts))

        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch: {len(vectors)} vectors for {len(chunks)} chunks"
            )

        with session_scope() as db:
            # Replace rather than append, so re-ingesting is idempotent.
            db.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )
            db.add_all(
                [
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk.index,
                        content=chunk.content,
                        word_count=chunk.word_count,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        section=chunk.section,
                        subject=subject,
                        grade=grade,
                        notation_score=notation_score(chunk.content),
                        embedding=vector,
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ]
            )
            doc = db.get(Document, document_id)
            doc.page_count = max((p.number for p in pages), default=0)
            doc.chunk_count = len(chunks)
            doc.status = IngestStatus.ready
            doc.ingested_at = datetime.now(timezone.utc)

            # New passages mean new words the student might misspell.
            rebuild_vocabulary(db)

        log.info("Ingested %s: %d pages, %d chunks", document_id, len(pages), len(chunks))

    except (UnsupportedFileType, EmptyDocument) as exc:
        _fail(document_id, str(exc))
    except Exception as exc:
        log.exception("Ingestion failed for %s", document_id)
        _fail(document_id, f"Could not process this file: {exc}")


def _fail(document_id, message: str) -> None:
    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is not None:
            doc.status = IngestStatus.failed
            doc.error = message[:2000]
