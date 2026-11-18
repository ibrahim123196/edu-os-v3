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
        planner,
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
        # Test me: timed tests that can be left and continued, and quizzes
        # ended early. Columns on a table that already existed.
        for column, spec in (
            ("kind", "varchar(20) NOT NULL DEFAULT 'practice'"),
            ("stopped", "varchar(20)"),
            ("active_seconds", "integer NOT NULL DEFAULT 0"),
            ("timing_since", "timestamptz"),
        ):
            conn.execute(
                text(f"ALTER TABLE quiz_sessions ADD COLUMN IF NOT EXISTS {column} {spec}")
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
        # Which model independently checked a question.
        conn.execute(
            text("ALTER TABLE questions ADD COLUMN IF NOT EXISTS checked_by varchar(80)")
        )
        # Every quiz a plan session opened, for its best score. Rows from
        # before it get their one linked quiz.
        conn.execute(text(
            "ALTER TABLE plan_sessions ADD COLUMN IF NOT EXISTS quiz_ids jsonb "
            "NOT NULL DEFAULT '[]'::jsonb"))
        conn.execute(text(
            "UPDATE plan_sessions SET quiz_ids = jsonb_build_array(link_id::text) "
            "WHERE link_kind = 'quiz' AND link_id IS NOT NULL AND quiz_ids = '[]'::jsonb"))
        # Several exams per plan (19 Nov): each plan's single exam becomes
        # its first row in plan_exams, and the old columns may be empty.
        for column in ("exam_name", "exam_date", "chapter_ids"):
            conn.execute(text(f"ALTER TABLE study_plans ALTER COLUMN {column} DROP NOT NULL"))
        conn.execute(text("ALTER TABLE plan_exams ADD COLUMN IF NOT EXISTS set_on date"))
        conn.execute(text(
            "INSERT INTO plan_exams (id, plan_id, name, exam_date, chapter_ids, position, set_on) "
            "SELECT gen_random_uuid(), p.id, COALESCE(NULLIF(p.exam_name, ''), 'My exam'), "
            "p.exam_date, COALESCE(p.chapter_ids, '[]'::jsonb), 0, p.created_at::date "
            "FROM study_plans p "
            "WHERE p.exam_date IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM plan_exams e WHERE e.plan_id = p.id)"))
        # Questions confirmed faulty; their answers stop counting.
        conn.execute(
            text(
                "ALTER TABLE questions ADD COLUMN IF NOT EXISTS faulty boolean "
                "NOT NULL DEFAULT false"
            )
        )
        # Never filled before; rows from then get it once, here. Same rule
        # as mastery_service.recompute: answers to faulty questions are not
        # counted. Rows that already have it are left alone.
        conn.execute(
            text(
                "UPDATE concept_mastery m SET last_attempt_at = ("
                " SELECT max(a.created_at) FROM attempts a"
                " JOIN questions q ON q.id = a.question_id"
                " WHERE a.student_id = m.student_id AND q.concept_id = m.concept_id"
                " AND NOT q.faulty)"
                " WHERE m.last_attempt_at IS NULL"
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
    ensure_links()
    log.info("Database ready (embedding_dim=%s)", dim)


# Links the models declare but an older database may lack: each column was
# added to a table that already existed, and create_all never revisits one.
# Without them the database accepts an attempt by a student who does not
# exist -- which once let a script move answers to a missing student and
# fail only later, somewhere else.
#   (table, column, referenced table, on delete, constraint name)
LINKS = [
    ("attempts", "student_id", "students", "CASCADE", "attempts_student_id_fkey"),
    ("documents", "chapter_id", "chapters", "SET NULL", "documents_chapter_id_fkey"),
]


def ensure_links() -> None:
    """Add each missing link, then check existing rows obey it.

    Added NOT VALID: enforced for every new row, without checking old ones,
    so it cannot fail on rows that already break it -- a startup that
    crashed on old data would take the whole app down over one bad row.
    Validating the old rows is a separate step; if it fails the link still
    guards new rows, a warning is logged, and the doctor names the rows.
    """
    for table, column, target, on_delete, name in LINKS:
        with engine.begin() as conn:
            # Looked up by column, not by name: a database that has the
            # link under another name must not get a second copy.
            exists = conn.execute(
                text(
                    "SELECT 1 FROM pg_constraint c "
                    "JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey) "
                    "WHERE c.contype = 'f' AND c.conrelid = CAST(:t AS regclass) "
                    "AND a.attname = :col"
                ),
                {"t": table, "col": column},
            ).first()
            if exists:
                continue
            conn.execute(
                text(
                    f"ALTER TABLE {table} ADD CONSTRAINT {name} FOREIGN KEY ({column}) "
                    f"REFERENCES {target}(id) ON DELETE {on_delete} NOT VALID"
                )
            )
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} VALIDATE CONSTRAINT {name}"))
        except Exception as exc:  # existing rows break the link
            log.warning(
                "%s.%s: link to %s added for new rows, but existing rows break it "
                "(python -m scripts.doctor names them): %s",
                table, column, target, str(exc).splitlines()[0],
            )
