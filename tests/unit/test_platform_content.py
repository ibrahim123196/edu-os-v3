"""Platform content is background knowledge, not files the student manages.

The NCERT chapters are loaded once and read by everyone. Listing them beside
a student's own uploads, each with a Remove button, invited any single user
to delete the shared corpus.
"""

from __future__ import annotations

import inspect

from apps.api.curriculum import License, SourceKind
from apps.api.routes import documents


def test_the_two_kinds_of_source_are_distinguished():
    assert SourceKind.student_upload.value == "student_upload"
    assert SourceKind.platform.value == "platform"


def test_the_document_list_excludes_platform_content_by_default():
    source = inspect.getsource(documents.list_documents)
    assert "include_platform" in source
    assert "SourceKind.student_upload" in source


def test_administrative_callers_can_still_see_platform_content():
    signature = inspect.signature(documents.list_documents)
    assert "include_platform" in signature.parameters
    assert signature.parameters["include_platform"].default is False


def test_platform_content_cannot_be_deleted():
    source = inspect.getsource(documents.delete_document)
    assert "SourceKind.platform" in source
    assert "403" in source


def test_ncert_content_is_marked_as_public_curriculum():
    """Licence travels with the document so a citation can say what it is
    quoting and under what terms."""
    assert License.public_curriculum.value == "public_curriculum"
    from scripts import ingest_ncert

    source = inspect.getsource(ingest_ncert.main)
    assert "License.public_curriculum" in source
    assert "SourceKind.platform" in source
    assert 'publisher="NCERT"' in source
