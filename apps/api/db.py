"""Database engine, session management and schema bootstrap."""

from __future__ import annotations

import logging
import ssl
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import URL, create_engine, make_url, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from apps.api.config import settings

log = logging.getLogger(__name__)

def _psycopg_loads() -> bool:
    try:
        import psycopg  # noqa: F401
    except ImportError:
        return False
    return True


def engine_url_and_args(database_url: str) -> tuple[URL, dict]:
    """The URL and connection options to actually use.

    psycopg ships a compiled library, and Windows Smart App Control blocks
    unrecognised libraries -- one day it loaded, the next it did not, and
    nothing in the project had changed. pg8000 is plain Python, so there is
    nothing for Windows to block; it is used whenever psycopg cannot load.

    pg8000 does not understand libpq's sslmode=... in the URL, and passing
    it through fails the connection, so it is turned into a TLS context.
    The certificate is verified: Neon's is valid, and an unverified
    connection to a database across the internet is not worth having.
    """
    url = make_url(database_url)
    if url.drivername == "postgresql+psycopg" and not _psycopg_loads():
        log.warning("psycopg cannot load on this computer; using pg8000 instead")
        url = url.set(drivername="postgresql+pg8000")

    connect_args: dict = {}
    if url.drivername == "postgresql+pg8000":
        query = dict(url.query)
        sslmode = query.pop("sslmode", None)
        # libpq-only, like sslmode. pg8000 negotiates authentication itself.
        query.pop("channel_binding", None)
        url = url.set(query=query)
        if sslmode not in (None, "disable"):
            connect_args["ssl_context"] = ssl.create_default_context()
    return url, connect_args


_url, _connect_args = engine_url_and_args(settings.database_url)
engine = create_engine(
    _url,
    connect_args=_connect_args,
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
        # Quiz sessions. quiz_sessions is a new table, so create_all above
        # has already made it; the column on attempts is not, so it is added
        # here along with the indexes create_all would have given a new table.
        conn.execute(
            text(
                "ALTER TABLE attempts ADD COLUMN IF NOT EXISTS session_id uuid "
                "REFERENCES quiz_sessions(id) ON DELETE SET NULL"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_attempts_session_id "
                "ON attempts (session_id)"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_attempt_session_question "
                "ON attempts (session_id, question_id)"
            )
        )
        # Named misconceptions. The misconceptions table is new, so
        # create_all made it; the link on question_options is a column added
        # to a table that already existed.
        conn.execute(
            text(
                "ALTER TABLE question_options ADD COLUMN IF NOT EXISTS misconception_id uuid "
                "REFERENCES misconceptions(id) ON DELETE SET NULL"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_question_options_misconception_id "
                "ON question_options (misconception_id)"
            )
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
