"""Database engine, session management and schema bootstrap."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from apps.api.config import settings

log = logging.getLogger(__name__)

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Standalone session for background workers and scripts."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def ensure_students(db) -> None:
    """Create the stand-in students if they are missing. Safe to repeat.

    One for real study, one for testing, until there is a login. Separate
    from init_db because init_db only runs when the server starts: a script
    run before that once found the test student missing and failed on it.
    Takes a Connection or a Session, so a script can do this inside its own
    transaction.
    """
    from apps.api.student import STUDENTS

    for student_id, name in STUDENTS.values():
        db.execute(
            text(
                "INSERT INTO students (id, display_name) "
                "VALUES (:id, :name) ON CONFLICT (id) DO NOTHING"
            ),
            {"id": str(student_id), "name": name},
        )


def init_db() -> None:
    """Create the pgvector extension, tables, and the search indexes.

    Milestone 1 uses create_all for speed. Once the schema starts changing
    under real data, replace this with Alembic migrations in
    infrastructure/migrations/.
    """
    from apps.api import (  # noqa: F401  (register mappers)
        curriculum,
        models,
        quiz,
        student,
    )

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        ensure_students(conn)

    dim = settings.embedding_dim
    with engine.begin() as conn:
        # Added after the first release; existing installs get it here rather
        # than having to drop their data. Re-ingest to populate it.
        conn.execute(
            text(
                "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS "
                "notation_score double precision NOT NULL DEFAULT 0"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS "
                "pipeline_version integer NOT NULL DEFAULT 0"
            )
        )
        for column, spec in (
            ("chapter_id", "uuid"),
            ("source_kind", "varchar(40) NOT NULL DEFAULT 'student_upload'"),
            ("license", "varchar(40) NOT NULL DEFAULT 'unknown'"),
            ("publisher", "varchar(200)"),
            ("edition", "varchar(80)"),
            ("source_url", "varchar(1000)"),
        ):
            conn.execute(
                text(f"ALTER TABLE documents ADD COLUMN IF NOT EXISTS {column} {spec}")
            )
        # HNSW gives good recall/latency for cosine distance without tuning
        # lists the way IVFFlat requires.
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_chunk_embedding_hnsw "
                "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_chunk_tsv "
                "ON document_chunks USING gin (content_tsv)"
            )
        )
        # Trigram index: makes fuzzy word matching fast enough to run on
        # every query, rather than scanning the whole vocabulary.
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_corpus_term_trgm "
                "ON corpus_terms USING gin (term gin_trgm_ops)"
            )
        )
    log.info("Database ready (embedding_dim=%s)", dim)
