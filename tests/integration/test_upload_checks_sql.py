"""The upload route refuses what the checks refuse, and keeps nothing of it.

Local database only (see conftest). Only refusals are sent: an accepted
file would start ingestion, which needs the embedding model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from apps.api.config import settings
from apps.api.db import init_db, session_scope
from apps.api.main import app
from apps.api.models import Document

pytestmark = pytest.mark.integration


@pytest.fixture
def client(tmp_path, monkeypatch):
    init_db()
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    return TestClient(app)


def _documents() -> int:
    with session_scope() as db:
        return db.scalar(select(func.count()).select_from(Document))


def _send(client, name: str, data: bytes):
    return client.post("/documents", files={"file": (name, data, "application/pdf")})


def test_a_fake_pdf_is_refused_and_not_kept(client, tmp_path):
    before = _documents()
    r = _send(client, "notes.pdf", b"MZ\x90\x00 a program, not a PDF")
    assert r.status_code == 422
    assert "not really a PDF" in r.json()["detail"]
    assert _documents() == before
    assert list((tmp_path / "uploads").iterdir()) == []      # the file is gone too


def test_binary_text_is_refused(client):
    r = _send(client, "notes.txt", b"abc\x00\x01")
    assert r.status_code == 422


def test_a_student_with_too_many_files_is_told_to_remove_one(client, monkeypatch):
    monkeypatch.setattr(settings, "max_uploads_per_student", 0)
    r = _send(client, "notes.txt", b"Ohm's law")
    assert r.status_code == 409 and "Remove one" in r.json()["detail"]


def test_a_student_out_of_space_is_told_so(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_storage_mb", 0)
    r = _send(client, "notes.txt", b"Ohm's law")
    assert r.status_code == 409 and "MB in all" in r.json()["detail"]
    assert list((tmp_path / "uploads").iterdir()) == []
