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
    # Any level: warnings carry fixes too, and printing them only for
    # failures hid the very next step a warning was pointing at.
    if fix:
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

    # Approved before the independent check existed means never verified;
    # a review of such questions found wrong answer keys.
    unchecked = db.scalar(
        select(func.count())
        .select_from(Question)
        .where(Question.status == QuestionStatus.approved, Question.checked_by.is_(None))
    ) or 0
    if unchecked:
        report(
            WARN,
            f"{unchecked} of {approved} approved questions were never independently checked",
            "their answer keys have not been verified",
            fix="python -m scripts.check_existing_questions",
        )
    elif approved:
        report(PASS, f"all {approved} approved questions independently checked")

    # A flagged question is out of every quiz until someone reviews it, and
    # nothing reviews it automatically: without this line a forgotten flag
    # looks exactly like a question that was never there.
    from apps.api.quiz import QuestionFlag

    waiting = db.scalar(
        select(func.count(func.distinct(QuestionFlag.question_id))).where(
            QuestionFlag.resolved_at.is_(None)
        )
    ) or 0
    if waiting:
        report(
            WARN,
            f"{waiting} flagged question(s) waiting for review, out of every quiz until then",
            fix="python -m scripts.review_flags",
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


def check_misconceptions(db) -> None:
    print("\nNamed mistakes")
    from apps.api.curriculum import Concept
    from apps.api.quiz import Misconception, Question, QuestionOption, QuestionStatus

    total = db.scalar(select(func.count()).select_from(Misconception)) or 0
    if total == 0:
        report(
            WARN,
            "no named mistakes loaded",
            "wrong answers cannot be grouped into repeated mistakes",
            fix="python -m scripts.seed_misconceptions",
        )
        return

    covered = db.scalar(select(func.count(func.distinct(Misconception.concept_id)))) or 0
    concepts = db.scalar(select(func.count()).select_from(Concept)) or 0
    report(PASS, f"{total} named mistakes across {covered} of {concepts} concepts")
    if covered < concepts:
        report(
            WARN,
            f"{concepts - covered} concept(s) have no named mistakes",
            "new questions for them cannot name what a wrong answer means",
        )

    # Only approved questions are ever handed out, so only their options
    # decide whether a wrong answer can be traced to a mistake.
    wrong = db.scalar(
        select(func.count())
        .select_from(QuestionOption)
        .join(Question, Question.id == QuestionOption.question_id)
        .where(~QuestionOption.is_correct, Question.status == QuestionStatus.approved)
    ) or 0
    linked = db.scalar(
        select(func.count())
        .select_from(QuestionOption)
        .join(Question, Question.id == QuestionOption.question_id)
        .where(
            ~QuestionOption.is_correct,
            Question.status == QuestionStatus.approved,
            QuestionOption.misconception_id.is_not(None),
        )
    ) or 0
    if wrong == 0:
        return
    share = linked / wrong
    # Below half, most wrong answers cannot be traced to a mistake and the
    # repeated-mistakes list stays empty however much is answered.
    report(
        PASS if share >= 0.5 else WARN,
        f"{linked} of {wrong} wrong options in approved questions name a mistake ({share:.0%})",
        # After matching, most older options still name nothing: they were
        # never built on a real mistake. Only new questions change this.
        "" if share >= 0.5 else "most older wrong options are not built on a common mistake",
        fix="" if share >= 0.5 else "new questions name a mistake as they are generated",
    )


def check_mastery(db) -> None:
    print("\nStudent model")
    from apps.api.quiz import Attempt
    from apps.api.student import ConceptMastery, Student

    students = db.scalar(select(func.count()).select_from(Student)) or 0
    if students == 0:
        report(FAIL, "No student row", fix="start the server once to seed it")
        return

    # Which student is recording, said out loud. A forgotten setting looks
    # identical to a correct one everywhere else.
    from apps.api.config import get_settings
    from apps.api.student import STUDENTS

    active = get_settings().active_student
    report(PASS, f"answers are being recorded against: {active}")

    # The setting can name a student whose row was never created; the first
    # answer then fails on the database. This once read OK while the test
    # student did not exist.
    present = set(db.scalars(select(Student.id)).all())
    missing = [name for name, (sid, _) in STUDENTS.items() if sid not in present]
    if missing:
        report(
            FAIL,
            f"student row(s) missing: {', '.join(missing)}",
            fix="start the server once to create them",
        )

    for name, (student_id, _) in STUDENTS.items():
        n = db.scalar(
            select(func.count()).select_from(Attempt).where(Attempt.student_id == student_id)
        ) or 0
        print(f"         {name}: {n} attempt(s)")

    attempts = db.scalar(select(func.count()).select_from(Attempt)) or 0
    if attempts == 0:
        report(WARN, "No questions answered yet", "mastery has nothing to work from")
        return

    tracked = db.scalar(select(func.count()).select_from(ConceptMastery)) or 0
    report(PASS, f"{attempts} attempt(s) across {tracked} concept(s)")

    # Mastery is cached but derived; a count that disagrees with the
    # attempts means a recompute was missed and the estimate has drifted
    # from what the student actually did.
    counted = db.scalar(select(func.sum(ConceptMastery.attempts))) or 0
    if counted != attempts:
        report(
            WARN,
            f"mastery counts {counted} attempts but {attempts} are stored",
            "the cache has drifted; it can be rebuilt from the attempts",
        )

    reliable = db.scalar(
        select(func.count())
        .select_from(ConceptMastery)
        .where(ConceptMastery.confidence >= 0.5)
    ) or 0
    if tracked and not reliable:
        report(
            WARN,
            "no concept has enough attempts to act on",
            "about five answered questions per concept are needed",
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


def check_schema(db) -> bool:
    """Does the database have what the code expects?

    create_all never alters an existing table, so a column added to a model
    reaches a database only through an explicit ALTER in init_db -- and a
    forgotten one is invisible until a query touches it. Returns False when
    columns are missing, because every check after this would then crash
    with an error that points somewhere else entirely.
    """
    print("\nSchema")
    from sqlalchemy import inspect

    from apps.api import curriculum, models, quiz, student  # noqa: F401
    from apps.api.db import Base

    insp = inspect(db.get_bind())
    existing = set(insp.get_table_names())

    missing_tables, missing_columns, missing_links, missing_unique = [], [], [], []
    for table in Base.metadata.sorted_tables:
        if table.name not in existing:
            missing_tables.append(table.name)
            continue

        columns = {c["name"] for c in insp.get_columns(table.name)}
        missing_columns += [
            f"{table.name}.{c.name}" for c in table.columns if c.name not in columns
        ]

        # Links between tables ("every attempt belongs to a real student").
        # Without one, the database accepts rows pointing at nothing.
        links = {
            (tuple(fk["constrained_columns"]), fk["referred_table"])
            for fk in insp.get_foreign_keys(table.name)
        }
        for fk in table.foreign_key_constraints:
            key = (tuple(fk.column_keys), fk.referred_table.name)
            if key not in links:
                missing_links.append(
                    f"{table.name}.{', '.join(fk.column_keys)} -> {fk.referred_table.name}"
                )

        # Unique rules carry meaning ("one answer per question per quiz");
        # plain indexes only affect speed and are not checked.
        indexes = {ix["name"] for ix in insp.get_indexes(table.name)}
        missing_unique += [
            f"{table.name}: {ix.name}"
            for ix in table.indexes
            if ix.unique and ix.name not in indexes
        ]

    if missing_tables or missing_columns:
        report(
            FAIL,
            "the database is missing what the code expects",
            "; ".join(missing_tables + missing_columns),
            fix="start the server once; if this stays, init_db lacks an ALTER",
        )
        return False
    report(PASS, "every table and column the code expects is present")

    if missing_links:
        report(
            WARN,
            f"{len(missing_links)} link(s) between tables are not enforced",
            "; ".join(missing_links),
        )
    if missing_unique:
        report(
            WARN,
            f"{len(missing_unique)} uniqueness rule(s) missing",
            "; ".join(missing_unique),
        )
    return True


def main() -> int:
    print(f"EduOS doctor — pipeline v{PIPELINE_VERSION}")
    # Said out loud: the driver can switch by itself when Windows blocks
    # psycopg, and a silent switch is the kind that gets forgotten.
    from apps.api.db import engine

    print(f"database driver: {engine.dialect.driver}")
    try:
        with session_scope() as db:
            if not check_schema(db):
                print("\n  Stopped here: the checks below would fail for this reason.")
                print(
                    f"\n{_counts[PASS]} ok, {_counts[WARN]} warning(s), "
                    f"{_counts[FAIL]} failure(s)"
                )
                return 1
            check_documents(db)
            check_chunks(db)
            check_curriculum(db)
            check_tagging(db)
            check_questions(db)
            check_misconceptions(db)
            check_mastery(db)
            check_vocabulary(db)
            check_corrections(db)
    except Exception as exc:
        # Not always a connection problem: a check that hits a missing column
        # also lands here, and blaming DATABASE_URL sent people the wrong way.
        print(f"\n  [{FAIL}] The doctor stopped with an error")
        print(f"         {exc}")
        print("         fix: if this is a connection error, check DATABASE_URL in .env")
        return 1

    print(
        f"\n{_counts[PASS]} ok, {_counts[WARN]} warning(s), {_counts[FAIL]} failure(s)"
    )
    return 1 if _counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
