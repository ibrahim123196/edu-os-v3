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
from ai.ingestion import PIPELINE_VERSION
from ai.ingestion.boilerplate import strip_boilerplate
from ai.ingestion.corrections import apply_corrections, load_corrections
from ai.ingestion.extract import EmptyDocument, UnsupportedFileType, extract
from ai.ingestion.notation import notation_score
from ai.ingestion.tagging import tag_chunks
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
        doc_filename = doc.filename
        chapter_id = doc.chapter_id

    try:
        pages = extract(path)

        # Running headers and footers ("Reprint 2026-27", "8 Physics") repeat
        # on every page. Left in, they break sentences mid-flow, get mistaken
        # for section headings, and fill the spelling vocabulary with words no
        # student will ever ask about.
        pages, furniture = strip_boilerplate(pages)
        if furniture:
            log.info("Stripped %d repeated header/footer lines", len(furniture))

        # Hand-corrections for platform content. Extraction gets most of a
        # textbook right; the remainder is fixed once, by a person, and is
        # then correct for every student who reads it.
        corrections = load_corrections(Path(doc_filename).stem)
        if corrections:
            result = apply_corrections(pages, corrections)
            pages = result.pages
            log.info("Corrections for %s: %s", doc_filename, result.summary)

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
            doc.pipeline_version = PIPELINE_VERSION
            doc.ingested_at = datetime.now(timezone.utc)

            # The chunks above are still pending in the session, and this
            # session has autoflush off. rebuild_vocabulary reads
            # document_chunks with raw SQL, so without an explicit flush it
            # queries a table that does not yet contain them and builds an
            # empty word list -- which is exactly what happened: 111 passages
            # indexed, zero terms, and spelling correction silently doing
            # nothing with no error anywhere.
            db.flush()

            # New passages mean new words the student might misspell.
            rebuild_vocabulary(db)

            # Link passages to curriculum concepts. Only platform content
            # is mapped to a chapter; a student's own upload has no place
            # in the graph until someone says where it belongs.
            if chapter_id is not None:
                tagged = _link_concepts(db, document_id, chapter_id, chunks)
                log.info("Tagged %d passage-concept links", tagged)

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


def _link_concepts(db, document_id, chapter_id, chunks) -> int:
    """Attach each passage to the concepts its section covers."""
    from sqlalchemy import delete as sql_delete
    from sqlalchemy import select

    from apps.api.curriculum import Topic, chunk_concepts

    topics = db.scalars(select(Topic).where(Topic.chapter_id == chapter_id)).all()
    if not topics:
        log.warning("Chapter %s has no topics; nothing to tag against", chapter_id)
        return 0

    by_number = {topic.number: topic for topic in topics}
    tags = tag_chunks([c.section for c in chunks], set(by_number))
    if not tags:
        return 0

    stored = db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    ).all()
    by_index = {chunk.chunk_index: chunk for chunk in stored}

    db.execute(
        sql_delete(chunk_concepts).where(
            chunk_concepts.c.chunk_id.in_([c.id for c in stored])
        )
    )

    rows = []
    seen: set[tuple] = set()
    for tag in tags:
        chunk = by_index.get(tag.chunk_index)
        topic = by_number.get(tag.topic_number)
        if chunk is None or topic is None:
            continue
        for concept in topic.concepts:
            key = (chunk.id, concept.id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "chunk_id": chunk.id,
                    "concept_id": concept.id,
                    "confidence": tag.confidence,
                }
            )

    if rows:
        db.execute(chunk_concepts.insert(), rows)
    return len(rows)
