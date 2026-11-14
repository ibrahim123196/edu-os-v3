"""Decide whether the integration tests may run, and against what.

Two reasons to skip:

1. The database is not up. A bare `pytest` should pass on a fresh clone.
   Erroring because Postgres is not running trains people to ignore red
   output, which is how a real failure gets missed.

2. The database is not a local one. These tests write to the tables they
   check and clean up after themselves -- and the spelling test's cleanup
   once emptied the whole vocabulary. Run against the real database (Neon),
   that would have silently switched spelling correction off for every
   student, with every test green. So they only ever run against a
   throwaway local database, whatever DATABASE_URL says.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

# localhost for a database on this machine, db for docker compose.
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "db"}


def _is_local_database() -> bool:
    from apps.api.config import get_settings

    return make_url(get_settings().database_url).host in LOCAL_HOSTS


def _database_available() -> bool:
    try:
        from apps.api.db import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def _skip_reason() -> str | None:
    # Host checked first: no reason to even connect to a remote database.
    if not _is_local_database():
        return (
            "DATABASE_URL points at a remote database; integration tests "
            "write to it, so they only run against a local one"
        )
    if not _database_available():
        return "needs Postgres with pgvector — run `docker compose up -d db`"
    return None


def pytest_collection_modifyitems(config, items):
    reason = _skip_reason()
    if reason is None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
