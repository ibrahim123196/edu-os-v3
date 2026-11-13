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
    from apps.api.student import LOCAL_STUDENT_ID

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

    Base.metadata.create_all(bind=engine)

    # The stand-in student that owns everything until there is a login.
    # Seeded before any column referencing it is added, so a NOT NULL
    # default has something valid to point at.
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO students (id, display_name) "
                "VALUES (:id, :name) ON CONFLICT (id) DO NOTHING"
            ),
            {"id": str(LOCAL_STUDENT_ID), "name": "This device"},
        )

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
        # Questions follow the passage they came from. The original rule
        # set the reference to null, leaving questions that could not be
        # checked against anything.
        conn.execute(
            text(
                "ALTER TABLE questions DROP CONSTRAINT IF EXISTS "
                "questions_source_chunk_id_fkey"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE questions ADD CONSTRAINT "
                "questions_source_chunk_id_fkey FOREIGN KEY (source_chunk_id) "
                "REFERENCES document_chunks(id) ON DELETE CASCADE"
            )
        )

        # create_all never alters an existing table, so every column added
        # after a table first shipped needs naming here. attempts.student_id
        # was missed, and answering a question failed on a table that had
        # existed since the previous version.
        conn.execute(
            text(
                "ALTER TABLE attempts ADD COLUMN IF NOT EXISTS student_id uuid "
                f"NOT NULL DEFAULT '{LOCAL_STUDENT_ID}'"
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