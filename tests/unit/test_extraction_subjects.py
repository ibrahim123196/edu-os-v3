"""Extraction rules found by running Chemistry, Maths and Biology chapters.

Each test builds a small page with the layout that broke, measured from the
real NCERT file it came from.
"""

from __future__ import annotations

import pytest

pymupdf = pytest.importorskip("pymupdf")

from ai.ingestion import heavy  # noqa: E402
from ai.ingestion.math_layout import page_text_with_math  # noqa: E402
from ai.ingestion.symbol_font import decode, is_symbol_font  # noqa: E402


def test_symbol_font_codes_become_the_characters_drawn():
    """NCERT sets Δ, ∝ and reaction arrows in the Symbol font: extracted
    as codes, "ΔTb ∝ m" read "DTb µ m"."""
    assert decode("D") == "Δ" and decode("µ") == "∝" and decode("®") == "→"
    assert decode("") == "Δ"                      # private-use copy
    assert decode("p e m") == "π ε μ"
    assert decode("2.5") == "2.5"                       # digits untouched
    assert is_symbol_font("Symbol") and is_symbol_font("ABCDEF+SymbolMT")
    assert not is_symbol_font("Bookman-Light") and not is_symbol_font(None)


def test_a_subscript_at_the_end_of_a_long_span_stays_on_its_line():
    """Relations and Functions, p.3: the "2" of "congruent to T2" ended a
    span 300 points wide and moved to the line below."""
    doc = pymupdf.open()
    page = doc.new_page()
    head = "R = {(T, T) : T is congruent to T"
    page.insert_text((60, 150), head, fontsize=11)
    end = 60 + pymupdf.get_text_length(head, fontsize=11)
    page.insert_text((end + 0.5, 153), "2", fontsize=6.4)
    page.insert_text((end + 4.5, 150), "}. Show that R is an equivalence.", fontsize=11)
    page.insert_text((60, 170), "Solution R is reflexive.", fontsize=11)
    lines = page_text_with_math(page).splitlines()
    first = next(l for l in lines if l.startswith("R = "))
    assert "congruent to T_2}" in first
    assert all("_2" not in l for l in lines if l.startswith("Solution"))


def test_table_columns_are_kept_apart():
    """d- and f-Block Elements, Table 4.4: the columns are placed apart, with
    no spaces stored, and a row arrived as "Ti4696561309-1866-1.63"."""
    doc = pymupdf.open()
    page = doc.new_page()
    for x, cell in zip((80, 150, 210, 270, 330, 400), ("Ti", "469", "656", "1309", "-1866", "-1.63")):
        page.insert_text((x, 200), cell, fontsize=10)
    assert "Ti 469 656 1309 -1866 -1.63" in page_text_with_math(page)


def test_a_header_page_number_is_not_a_superscript():
    """"if R is^3": the small page number in the header became the
    superscript of the page's first line."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((500, 80), "3", fontsize=8)
    page.insert_text((60, 104), "A relation R in a set A is an equivalence relation if R is", fontsize=11)
    page.insert_text((60, 118), "reflexive, symmetric and transitive.", fontsize=11)
    assert "is^3" not in page_text_with_math(page)


def test_a_page_of_traced_artwork_is_too_heavy_to_read_drawings(monkeypatch):
    """Human Reproduction, p.14: a 25 MB drawing stream; listing its
    drawings ran out of memory and killed the ingestion."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(10, 10, 20, 20))
    assert not heavy.too_heavy(page)
    monkeypatch.setattr(heavy, "MAX_DRAWING_BYTES", 1)
    assert heavy.too_heavy(page)
    # The page's text still comes out, from the text alone.
    page.insert_text((60, 100), "The text of the page.", fontsize=11)
    assert "The text of the page." in page_text_with_math(page)
