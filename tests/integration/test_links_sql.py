"""Adding missing links to a database that predates them.

Neon's attempts and documents tables lacked two links the models declare.
Local database only (see conftest): these drop constraints on purpose.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from apps.api.db import LINKS, engine, ensure_links, init_db
from scripts import doctor

pytestmark = pytest.mark.integration

# psycopg calls a broken link an IntegrityError, pg8000 a ProgrammingError;
# either way the message names the link that refused the row.
REFUSED = "attempts_student_id_fkey"


def _links_on(table: str, column: str) -> list[tuple[str, bool]]:
    with engine.connect() as conn:
        return [tuple(r) for r in conn.execute(text(
            "SELECT c.conname, c.convalidated FROM pg_constraint c "
            "JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey) "
            "WHERE c.contype = 'f' AND c.conrelid = CAST(:t AS regclass) AND a.attname = :c"
        ), {"t": table, "c": column})]


def _drop_links():
    """The state Neon was in: no link on either column."""
    with engine.begin() as conn:
        for table, column, *_ in LINKS:
            for name, _ in _links_on(table, column):
                conn.execute(text(f"ALTER TABLE {table} DROP CONSTRAINT {name}"))


def _orphan_attempt(conn) -> uuid.UUID:
    """An attempt by a student who does not exist, on a real question."""
    question = conn.execute(text("SELECT id FROM questions LIMIT 1")).scalar()
    if question is None:
        concept = conn.execute(text(
            "INSERT INTO concepts (id, slug, name) VALUES (gen_random_uuid(), :s, 'Links') RETURNING id"
        ), {"s": f"links-{uuid.uuid4().hex[:6]}"}).scalar()
        question = conn.execute(text(
            "INSERT INTO questions (id, concept_id, kind, difficulty, stem, explanation, status, faulty) "
            "VALUES (gen_random_uuid(), :c, 'multiple_choice', 2, 'Q?', 'b', 'approved', false) RETURNING id"
        ), {"c": concept}).scalar()
    return conn.execute(text(
        "INSERT INTO attempts (id, student_id, question_id, is_correct) "
        "VALUES (gen_random_uuid(), gen_random_uuid(), :q, false) RETURNING id"
    ), {"q": question}).scalar()


@pytest.fixture
def fresh():
    init_db()
    _drop_links()
    yield
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM attempts WHERE student_id NOT IN (SELECT id FROM students)"))
        conn.execute(text("DELETE FROM concepts WHERE slug LIKE 'links-%'"))
    ensure_links()


def test_missing_links_are_added_and_confirmed(fresh):
    ensure_links()
    for table, column, *_ in LINKS:
        assert [ok for _, ok in _links_on(table, column)] == [True]


def test_after_that_the_database_refuses_a_broken_row(fresh):
    ensure_links()
    with pytest.raises(DBAPIError, match=REFUSED):
        with engine.begin() as conn:
            _orphan_attempt(conn)


def test_old_broken_rows_do_not_stop_startup(fresh, capsys):
    """A crash here would take the whole app down over one bad row."""
    with engine.begin() as conn:
        _orphan_attempt(conn)

    ensure_links()                       # must not raise

    links = _links_on("attempts", "student_id")
    assert len(links) == 1 and links[0][1] is False   # added, not yet confirmed
    with pytest.raises(DBAPIError, match=REFUSED):              # but new bad rows are refused
        with engine.begin() as conn:
            _orphan_attempt(conn)

    from apps.api.db import session_scope
    with session_scope() as db:
        doctor.check_schema(db)
    out = capsys.readouterr().out
    assert "1 row(s) in attempts point at a students row that does not exist" in out


def test_running_twice_adds_nothing_twice(fresh):
    ensure_links()
    ensure_links()
    for table, column, *_ in LINKS:
        assert len(_links_on(table, column)) == 1
