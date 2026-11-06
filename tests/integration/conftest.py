"""Skip integration tests cleanly when the database is not up.

A bare `pytest` should pass on a fresh clone. Erroring because Postgres is not
running trains people to ignore red output, which is how a real failure gets
missed.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text


def _database_available() -> bool:
    try:
        from apps.api.db import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _database_available():
        return
    skip = pytest.mark.skip(
        reason="needs Postgres with pgvector — run `docker compose up -d db`"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
