"""Re-apply curriculum tagging to passages already in the database.

Concept links are written during ingestion, so editing the curriculum graph
leaves them stale: attaching a concept to a topic, or correcting a section
number, changes nothing until something re-tags.

Re-ingesting would do it, but at the cost of re-extracting and re-embedding
every page -- minutes per chapter, and the embeddings have not changed.
Tagging reads only the section heading already stored on each passage, so it
can be redone in seconds.

    python -m scripts.retag

Nothing is extracted, embedded or re-chunked. Only the passage-to-concept
links are rewritten.
"""

from __future__ import annotations

import sys

from sqlalchemy import delete, select

from ai.ingestion.tagging import tag_chunks
from apps.api.curriculum import Topic, chunk_concepts
from apps.api.db import session_scope
from apps.api.models import Document, DocumentChunk, IngestStatus


def main() -> int:
    total_links = 0
    documents_done = 0

    with session_scope() as db:
        documents = db.scalars(
            select(Document).where(
                Document.status == IngestStatus.ready,
                Document.chapter_id.isnot(None),
            )
        ).all()

        if not documents:
            print("No documents are mapped to a chapter; nothing to tag.")
            return 1

        for document in documents:
            topics = db.scalars(
                select(Topic).where(Topic.chapter_id == document.chapter_id)
            ).all()
            if not topics:
                print(f"  {document.title[:50]}: chapter has no topics, skipped")
                continue

            by_number = {topic.number: topic for topic in topics}
            chunks = db.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document.id)
                .order_by(DocumentChunk.chunk_index)
            ).all()

            tags = tag_chunks([c.section for c in chunks], set(by_number))
            by_index = {c.chunk_index: c for c in chunks}

            db.execute(
                delete(chunk_concepts).where(
                    chunk_concepts.c.chunk_id.in_([c.id for c in chunks])
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

            covered = len({t.topic_number for t in tags})
            print(
                f"  {document.title[:46]:<46} "
                f"{len(tags)}/{len(chunks)} passages, "
                f"{covered}/{len(topics)} topics, {len(rows)} links"
            )
            total_links += len(rows)
            documents_done += 1

    print(f"\n{documents_done} document(s) re-tagged, {total_links} links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
