"""Hand-corrections for platform content.

Extraction gets roughly 99% of a textbook right, and a wrong formula looks
exactly like a correct one. For a student's own upload that is the ceiling.
For platform content it is not: the corpus is finite and every student reads
the same copy, so a formula fixed once is fixed for everyone.
"""

from __future__ import annotations

import json

from ai.ingestion.corrections import (
    Correction,
    apply_corrections,
    load_corrections,
)
from ai.ingestion.extract import Page


def pages() -> list[Page]:
    return [
        Page(6, "charge e. Since e = 1.6 x 10_13 C, a charge\nmore text here"),
        Page(7, "F = k (q_1q_2)/(r^2)\nalso 1.6 x 10_13 C appears here"),
    ]


def test_a_correction_is_applied():
    result = apply_corrections(
        pages(), [Correction(page=6, find="10_13 C", replace="10^-19 C")]
    )
    assert "10^-19 C" in result.pages[0].text
    assert len(result.applied) == 1


def test_a_page_number_limits_the_replacement():
    """The same wrong string often appears on several pages for different
    reasons, so a correction naming a page must not leak onto others."""
    result = apply_corrections(
        pages(), [Correction(page=6, find="10_13 C", replace="10^-19 C")]
    )
    assert "10_13 C" in result.pages[1].text


def test_a_correction_without_a_page_applies_everywhere():
    result = apply_corrections(
        pages(), [Correction(page=None, find="10_13 C", replace="10^-19 C")]
    )
    assert all("10_13" not in p.text for p in result.pages)


def test_a_stale_correction_is_reported_not_silently_skipped():
    """Extraction improves. A correction whose target no longer exists would
    otherwise sit there looking applied while doing nothing -- or hide a
    regression."""
    result = apply_corrections(
        pages(), [Correction(page=6, find="text that is not there", replace="x")]
    )
    assert len(result.unmatched) == 1
    assert "NO LONGER MATCH" in result.summary


def test_no_corrections_leaves_the_pages_untouched():
    original = pages()
    result = apply_corrections(original, [])
    assert [p.text for p in result.pages] == [p.text for p in original]


def test_a_missing_correction_file_is_not_an_error(tmp_path):
    assert load_corrections("no-such-document", directory=tmp_path) == []


def test_corrections_load_from_json(tmp_path):
    (tmp_path / "ch1.json").write_text(
        json.dumps(
            {
                "document": "ch1",
                "corrections": [
                    {"page": 6, "find": "a", "replace": "b", "note": "why"}
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_corrections("ch1", directory=tmp_path)
    assert len(loaded) == 1
    assert loaded[0].replace == "b"
    assert loaded[0].note == "why"


def test_malformed_json_is_ignored_rather_than_crashing_ingestion(tmp_path):
    (tmp_path / "ch1.json").write_text("{ not json", encoding="utf-8")
    assert load_corrections("ch1", directory=tmp_path) == []


# --- bulk loading --------------------------------------------------------

def test_bulk_ingest_rebuilds_the_vocabulary_once():
    """The spelling vocabulary is built by scanning every chunk in the
    database. Rebuilding it after each document makes a bulk import
    quadratic -- loading 150 chapters would rescan the whole corpus 150
    times."""
    import inspect

    from apps.api.services.ingestion_service import ingest_document
    from scripts import ingest_ncert

    assert "rebuild_vocab" in inspect.signature(ingest_document).parameters

    source = inspect.getsource(ingest_ncert.main)
    assert "rebuild_vocab=False" in source
    assert "rebuild_vocabulary" in source
