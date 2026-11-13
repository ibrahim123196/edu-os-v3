"""Check the live database for the problems that have actually bitten us.

Every check here exists because something went wrong once and was invisible
until it caused a confusing failure somewhere else: a vocabulary built from
an empty table, chunks written before a column existed, corrections silently
no longer matching, a document stuck half-ingested.

    python -m scripts.doctor

Nothing is modified. Each check prints OK, WARN or FAIL, and a FAIL names
what to do about it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import func, select, text

from ai.ingestion import PIPELINE_VERSION
from ai.ingestion.corrections import apply_corrections, load_corrections
from apps.api.curriculum import (
    Chapter,
    Concept,
    Subject,
    Topic,
    chunk_concepts,
)
from apps.api.db import session_scope
from apps.api.models import Document, DocumentChunk, IngestStatus

PASS, WARN, FAIL = "OK  ", "WARN", "FAIL"
_counts = {PASS: 0, WARN: 0, FAIL: 0}


def report(level: str, title: str, detail: str = "", fix: str = "") -> None:
    _counts[level] += 1
    print(f"  [{level}] {title}")
    if detail:
        print(f"         {detail}")
    if level == FAIL and fix:
        print(f"         fix: {fix}")


def check_documents(db) -> None:
    print("\nDocuments")
    total = db.scalar(select(func.count()).select_from(Document)) or 0
    if total == 0:
        report(FAIL, "No documents at all", fix="python -m scripts.ingest_ncert <folder>")
        return

    failed = db.scalars(
        select(Document).where(Document.status == IngestStatus.failed)
    ).all()
    if failed:
        report(
            FAIL,
            f"{len(failed)} document(s) failed to ingest",
            "; ".join(f"{d.title[:40]}: {(d.error or '')[:60]}" for d in failed[:3]),
            "fix the cause, then re-upload or re-ingest",
        )
    else:
        report(PASS, f"{total} document(s), none failed")

    stuck = db.scalar(
        select(func.count())
        .select_from(Document)
        .where(Document.status.in_([IngestStatus.pending, IngestStatus.processing]))
    )
    if stuck:
        report(
            WARN,
            f"{stuck} document(s) still pending or processing",
            "normal if an ingest is running; otherwise it died part-way",
        )

    stale = db.scalars(
        select(Document).where(
            Document.status == IngestStatus.ready,
            Document.pipeline_version < PIPELINE_VERSION,
        )
    ).all()
    if stale:
        report(
            FAIL,
            f"{len(stale)} document(s) built by an older pipeline "
            f"(now v{PIPELINE_VERSION})",
            "; ".join(f"{d.title[:36]} (v{d.pipeline_version})" for d in stale[:3]),
            "remove and re-upload, or POST /documents/{id}/reingest",
        )
    else:
        report(PASS, f"all ready documents built by pipeline v{PIPELINE_VERSION}")

    missing = [
        d for d in db.scalars(select(Document)).all()
        if d.storage_path and not Path(d.storage_path).exists()
    ]
    if missing:
        report(
            WARN,
            f"{len(missing)} document(s) whose source file is gone",
            "they stay searchable, but cannot be re-ingested",
        )


def check_chunks(db) -> None:
    print("\nPassages")
    total = db.scalar(select(func.count()).select_from(DocumentChunk)) or 0
    if total == 0:
        report(FAIL, "No passages indexed", fix="re-ingest your documents")
        return

    unembedded = db.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.embedding.is_(None))
    )
    if unembedded:
        report(
            FAIL,
            f"{unembedded} passage(s) have no embedding",
            "they are invisible to semantic search",
            "re-ingest the affected document",
        )
    else:
        report(PASS, f"{total} passages, all embedded")

    with_notation = db.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.notation_score >= 0.3)
    )
    if with_notation == 0:
        report(
            FAIL,
            "No passage carries notation",
            "formula questions cannot work",
            "re-ingest; this means chunks predate the notation detector",
        )
    else:
        report(PASS, f"{with_notation} passages carry formulas")

    empty_section = db.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.section.is_(None))
    )
    if empty_section > total * 0.5:
        report(
            WARN,
            f"{empty_section} of {total} passages have no section heading",
            "heading detection may be failing; curriculum tagging depends on it",
        )


def check_curriculum(db) -> None:
    print("\nCurriculum graph")
    subjects = db.scalar(select(func.count()).select_from(Subject)) or 0
    concepts = db.scalar(select(func.count()).select_from(Concept)) or 0
    topics = db.scalar(select(func.count()).select_from(Topic)) or 0

    if not concepts:
        report(FAIL, "The curriculum graph is empty", fix="python -m scripts.seed_curriculum")
        return
    report(PASS, f"{subjects} subject(s), {topics} topics, {concepts} concepts")

    untagged_topics = db.scalars(
        select(Topic).where(~Topic.concepts.any())
    ).all()
    if untagged_topics:
        report(
            WARN,
            f"{len(untagged_topics)} topic(s) link to no concept",
            ", ".join(f"{t.number} {t.name[:28]}" for t in untagged_topics[:4]),
        )

    orphan_chapters = db.scalars(select(Chapter).where(~Chapter.topics.any())).all()
    if orphan_chapters:
        report(WARN, f"{len(orphan_chapters)} chapter(s) have no topics")


def check_tagging(db) -> None:
    print("\nConcept tagging")
    links = db.scalar(select(func.count()).select_from(chunk_concepts)) or 0
    mapped_chunks = db.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(Document.chapter_id.isnot(None))
    ) or 0

    if mapped_chunks == 0:
        report(
            WARN,
            "No passage belongs to a chapter in the graph",
            "only platform content is tagged; student uploads are not",
        )
        return

    if links == 0:
        report(
            FAIL,
            "No passage is tagged with a concept",
            "quizzes and mastery cannot work",
            "re-ingest the platform chapters",
        )
        return

    tagged = db.scalar(
        select(func.count(func.distinct(chunk_concepts.c.chunk_id)))
    ) or 0
    coverage = 100 * tagged // mapped_chunks
    level = PASS if coverage >= 90 else WARN
    report(level, f"{tagged}/{mapped_chunks} passages tagged ({coverage}%), {links} links")

    # A concept nothing teaches is a hole in the corpus, not a bug, but the
    # quiz engine will ask for it and find nothing.
    empty = db.scalars(select(Concept).where(~Concept.id.in_(
        select(chunk_concepts.c.concept_id)
    ))).all()
    if empty:
        report(
            WARN,
            f"{len(empty)} concept(s) have no passages",
            ", ".join(c.name[:30] for c in empty[:5]),
        )

    # Tags must not cross chapters.
    stray = db.execute(text("""
        SELECT COUNT(*) FROM chunk_concepts cc
        JOIN document_chunks dc ON dc.id = cc.chunk_id
        JOIN documents d ON d.id = dc.document_id
        JOIN topic_concepts tc ON tc.concept_id = cc.concept_id
        JOIN topics t ON t.id = tc.topic_id
        WHERE d.chapter_id IS NOT NULL AND t.chapter_id <> d.chapter_id
    """)).scalar() or 0
    if stray:
        report(
            WARN,
            f"{stray} link(s) point at a concept taught in another chapter",
            "expected where a concept is shared, worth a look if it is large",
        )


def check_questions(db) -> None:
    print("\nQuestion bank")
    from apps.api.quiz import Question, QuestionStatus

    total = db.scalar(select(func.count()).select_from(Question)) or 0
    if total == 0:
        report(
            WARN,
            "No questions yet",
            fix="python -m scripts.generate_questions --per-concept 1",
        )
        return

    approved = db.scalar(
        select(func.count())
        .select_from(Question)
        .where(Question.status == QuestionStatus.approved)
    ) or 0
    rejected = total - approved
    report(PASS, f"{approved} approved, {rejected} rejected of {total}")

    if rejected > approved:
        report(
            WARN,
            "more questions were rejected than approved",
            "the prompt or the passages may need work; "
            "SELECT stem, rejection_reason FROM questions WHERE status='rejected'",
        )

    orphan = db.scalar(
        select(func.count())
        .select_from(Question)
        .where(Question.source_chunk_id.is_(None))
    ) or 0
    if orphan:
        report(
            WARN,
            f"{orphan} question(s) cite no passage",
            "their explanations cannot be checked against the material",
        )


def check_vocabulary(db) -> None:
    print("\nSpelling vocabulary")
    terms = db.execute(text("SELECT COUNT(*) FROM corpus_terms")).scalar() or 0
    chunks = db.scalar(select(func.count()).select_from(DocumentChunk)) or 0

    if chunks and terms == 0:
        report(
            FAIL,
            "The vocabulary is empty",
            "typo correction silently does nothing",
            "re-ingest, or run the INSERT in content/rebuild_vocabulary.sql",
        )
    elif terms < chunks:
        report(WARN, f"only {terms} terms for {chunks} passages", "unusually few")
    else:
        report(PASS, f"{terms} distinct terms")


def check_corrections(db) -> None:
    print("\nHand corrections")
    from ai.ingestion.boilerplate import strip_boilerplate
    from ai.ingestion.extract import Page

    documents = db.scalars(
        select(Document).where(Document.status == IngestStatus.ready)
    ).all()

    checked = stale = applied = 0
    for document in documents:
        key = Path(document.filename).stem
        corrections = load_corrections(key)
        if not corrections:
            continue
        checked += 1
        rows = db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.chunk_index)
        ).all()
        # Reconstruct page text from stored chunks: good enough to tell
        # whether a correction's target text still exists anywhere.
        joined = "\n".join(c.content for c in rows)
        for correction in corrections:
            if correction.replace and correction.replace in joined:
                applied += 1
            elif correction.find in joined:
                stale += 1

    if checked == 0:
        names = ", ".join(sorted({Path(d.filename).stem for d in documents})[:4])
        available = sorted(p.stem for p in Path("content/corrections").glob("*.json"))
        if available:
            report(
                WARN,
                "correction files exist but match no document",
                f"documents: {names or 'none'} | correction files: "
                f"{', '.join(available[:4])}",
            )
        else:
            report(PASS, "no correction files in use")
        return
    if stale:
        report(
            WARN,
            f"{stale} correction(s) appear not to have been applied",
            "the document may predate them; re-ingest to apply",
        )
    else:
        report(PASS, f"{applied} correction(s) present in the indexed text")


def main() -> int:
    print(f"EduOS doctor — pipeline v{PIPELINE_VERSION}")
    try:
        with session_scope() as db:
            check_documents(db)
            check_chunks(db)
            check_curriculum(db)
            check_tagging(db)
            check_questions(db)
            check_vocabulary(db)
            check_corrections(db)
    except Exception as exc:
        print(f"\n  [{FAIL}] Could not reach the database")
        print(f"         {exc}")
        print("         fix: check DATABASE_URL in .env")
        return 1

    print(
        f"\n{_counts[PASS]} ok, {_counts[WARN]} warning(s), {_counts[FAIL]} failure(s)"
    )
    return 1 if _counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
