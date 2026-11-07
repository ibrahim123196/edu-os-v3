"""Running headers and footers.

NCERT pages carry "Reprint 2026-27" and "8 Physics" on every page. pypdf
returns them inside the page text, where they split sentences, get mistaken
for section headings, and pollute the spelling vocabulary. One real chunk came
back labelled with the section "Reprint 2026-27".
"""

from __future__ import annotations

from ai.ingestion.boilerplate import find_boilerplate, strip_boilerplate
from ai.ingestion.extract import Page


def ncert_like(count: int = 10) -> list[Page]:
    return [
        Page(
            n,
            f"Reprint 2026-27\n{n} Physics\n"
            f"Coulomb measured the force between two point charges on page {n}.\n"
            f"The force varies inversely with the square of the distance.\n{n}",
        )
        for n in range(6, 6 + count)
    ]


def test_repeated_footer_is_detected():
    found = find_boilerplate(ncert_like())
    assert any("reprint" in line for line in found)


def test_content_survives_stripping():
    cleaned, _ = strip_boilerplate(ncert_like())
    assert "Coulomb measured the force" in cleaned[0].text
    assert "Reprint" not in cleaned[0].text
    assert "Physics" not in cleaned[0].text


def test_bare_page_numbers_at_the_edges_are_removed():
    cleaned, _ = strip_boilerplate(ncert_like())
    assert not cleaned[0].text.strip().endswith("6")


def test_numbers_inside_the_page_body_are_kept():
    """A lone number mid-page may be an answer or a value, not a page number."""
    pages = [
        Page(
            n,
            f"Reprint 2026-27\nA sentence about charge on page {n}.\n"
            f"For case {n} the answer works out to\n42\n"
            f"and the derivation for case {n} continues below.\n"
            f"Further unique discussion of page {n} follows.\n{n}",
        )
        for n in range(1, 8)
    ]
    cleaned, _ = strip_boilerplate(pages)
    assert "42" in cleaned[0].text


def test_short_documents_are_left_alone():
    """Repetition means nothing across two pages."""
    pages = [Page(1, "Reprint 2026-27\nSome text."), Page(2, "Reprint 2026-27\nMore text.")]
    cleaned, found = strip_boilerplate(pages)
    assert found == set()
    assert "Reprint" in cleaned[0].text


def test_templated_body_text_is_not_mistaken_for_furniture():
    """"Example 1.1", "Example 1.2" normalise alike but are real content, and
    they sit in the body rather than the margins."""
    pages = [
        Page(
            n,
            f"Reprint 2026-27\nIntroductory sentence for this page.\n"
            f"Example 1.{n} If 10 electrons move out of a body.\n"
            f"Solution In one second the charge given out is calculated.\n{n}",
        )
        for n in range(1, 10)
    ]
    cleaned, _ = strip_boilerplate(pages)
    assert "Example 1.1" in cleaned[0].text


def test_a_document_with_no_furniture_is_unchanged():
    pages = [Page(n, f"Wholly unique prose on page {n} with nothing repeated.") for n in range(1, 8)]
    cleaned, found = strip_boilerplate(pages)
    assert found == set()
    assert len(cleaned) == len(pages)
