"""Rebuilding formulas from page geometry.

The failure this addresses: Coulomb's law sits in the PDF as a stacked
fraction, and plain extraction returned "F q= 2 0 24 pi e l" -- every
character present, every relationship lost. Retrieval could find the page and
still show the student nonsense.
"""

from __future__ import annotations

import pytest

pymupdf = pytest.importorskip("pymupdf")

from ai.ingestion.math_layout import page_text_with_math  # noqa: E402


def coulomb_page():
    """A page shaped like the NCERT original: an introducing sentence, a
    stacked fraction with scripts, an equation number in the margin, plus an
    underlined heading that must not be mistaken for a fraction."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "the magnitude of the force (F) is given by", fontsize=11)
    page.insert_text((100, 140), "F = k", fontsize=12)
    page.insert_text((140, 133), "|q", fontsize=11)
    page.insert_text((152, 136), "1", fontsize=7)
    page.insert_text((157, 133), "q", fontsize=11)
    page.insert_text((164, 136), "2", fontsize=7)
    page.insert_text((169, 133), "|", fontsize=11)
    page.draw_line(pymupdf.Point(138, 137), pymupdf.Point(185, 137), width=0.8)
    page.insert_text((152, 150), "r", fontsize=11)
    page.insert_text((158, 146), "2", fontsize=7)
    page.insert_text((420, 140), "(1.1)", fontsize=11)
    page.insert_text((72, 200), "This heading is underlined", fontsize=11)
    page.draw_line(pymupdf.Point(72, 203), pymupdf.Point(200, 203), width=0.8)
    page.insert_text((72, 222), "Ordinary prose continues below it.", fontsize=11)
    page.insert_text((72, 280), "If 10", fontsize=11)
    page.insert_text((96, 276), "9", fontsize=7)
    page.insert_text((102, 280), "electrons move out of a body.", fontsize=11)
    return page


def test_stacked_fraction_is_rebuilt_inline():
    text = page_text_with_math(coulomb_page())
    assert ")/(" in text, "fraction structure not recovered"
    assert "(|q_1q_2|)/(r^2)" in text


def test_numerator_and_denominator_are_not_separated():
    """Plain extraction put them on different lines, so the division was
    invisible to anything reading the text."""
    text = page_text_with_math(coulomb_page())
    line = next(l for l in text.splitlines() if ")/(" in l)
    assert "F = k" in line


def test_superscripts_are_marked():
    text = page_text_with_math(coulomb_page())
    # 10^9 read as "10 9" is a value a thousand million times too small.
    assert "10^9" in text
    assert "r^2" in text


def test_subscripts_are_marked():
    text = page_text_with_math(coulomb_page())
    assert "_1" in text and "_2" in text


def test_underline_is_not_treated_as_a_fraction():
    """An underlined heading has text above and below a horizontal rule, the
    same shape as a fraction. Width and proximity separate them."""
    text = page_text_with_math(coulomb_page())
    assert "(This heading is underlined)/" not in text
    assert "This heading is underlined" in text


def test_prose_is_unchanged():
    text = page_text_with_math(coulomb_page())
    assert "the magnitude of the force (F) is given by" in text


def test_page_with_no_drawings_is_handled():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Just ordinary prose with no equations at all.", fontsize=11)
    assert "ordinary prose" in page_text_with_math(page)


def test_empty_page_returns_empty_string():
    doc = pymupdf.open()
    assert page_text_with_math(doc.new_page()) == ""


# --- measured against the real NCERT chapter -----------------------------

def ncert_coulomb_page():
    """Page 7 of NCERT Physics Part I, Chapter 1, rebuilt to scale.

    Every coordinate here was measured from the real file. The gap ratios in
    particular matter: the numerator clears the bar by 0.25 of its height and
    the denominator by 0.29, which is why a tolerance tuned on an invented
    page rejected the denominator and produced half a fraction.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((78, 144), "the magnitude of the force (F) between them is given by", fontsize=10.5)
    page.insert_text((79.9, 170.9), "F", fontsize=10.5)
    page.insert_text((91.2, 171.6), "=", fontsize=10.5)
    page.insert_text((99.6, 170.9), "k", fontsize=10.5)
    page.insert_text((116.4, 163.2), "q", fontsize=10.5)
    page.insert_text((121.9, 166.3), "1", fontsize=6.0)
    page.insert_text((133.4, 163.2), "q", fontsize=10.5)
    page.insert_text((139.7, 166.1), "2", fontsize=6.0)
    page.draw_line(pymupdf.Point(114.5, 165.8), pymupdf.Point(146.9, 165.8), width=0.5)
    page.insert_text((125.0, 179.3), "r", fontsize=10.5)
    page.insert_text((130.6, 174.6), "2", fontsize=6.5)
    page.insert_text((338.2, 172.0), "(1.1)", fontsize=10.5)
    return page


def test_real_coulomb_law_is_rebuilt():
    text = page_text_with_math(ncert_coulomb_page())
    assert "(q_1q_2)/(r^2)" in text


def test_denominator_is_not_dropped_at_real_spacing():
    """The denominator sits further from the bar than the numerator. A gap
    tolerance that admits only the numerator yields a fraction missing its
    bottom half, which is worse than not rebuilding it at all."""
    text = page_text_with_math(ncert_coulomb_page())
    line = next(l for l in text.splitlines() if ")/(" in l)
    assert "r^2" in line


def small_caps_heading_page():
    """NCERT sets section headings in small capitals: a full-size initial
    followed by smaller capitals on the same baseline."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "1.14", fontsize=11)
    page.insert_text((100, 100), "A", fontsize=11)
    page.insert_text((108, 100), "PPLICATIONS", fontsize=8)
    page.insert_text((165, 100), "OF", fontsize=8)
    page.insert_text((180, 100), "G", fontsize=11)
    page.insert_text((188, 100), "AUSS", fontsize=8)
    page.insert_text((215, 100), "L", fontsize=11)
    page.insert_text((222, 100), "AW", fontsize=8)
    return page


def test_small_caps_headings_are_not_marked_as_scripts():
    """Marked as scripts this reads "A PPLICATIONS _OF G _AUSS L _AW", and
    headings are exactly what a student searches for."""
    text = page_text_with_math(small_caps_heading_page())
    assert "_OF" not in text
    assert "_AW" not in text
    assert "OF" in text and "AW" in text


def test_genuine_scripts_still_marked_on_ordinary_lines():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "If 10", fontsize=11)
    page.insert_text((96, 96), "9", fontsize=7)
    page.insert_text((102, 100), "electrons move out.", fontsize=11)
    assert "10^9" in page_text_with_math(page)


def test_operators_are_never_marked_as_scripts():
    """A display equation is often set smaller than the prose around it, so
    size alone marked every character in it. Ampere's law came back as
    "B ^. _d ^l _= _µ _0 I"."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "the integral is equal to", fontsize=11)
    page.insert_text((200, 100), "B", fontsize=8)
    page.insert_text((208, 100), ".", fontsize=8)
    page.insert_text((214, 100), "dl", fontsize=8)
    page.insert_text((228, 100), "=", fontsize=8)
    text = page_text_with_math(page)
    assert "_=" not in text and "^=" not in text
    assert "_." not in text and "^." not in text


def test_a_script_far_from_any_full_size_text_is_left_alone():
    """Beyond arm's reach the nearest full-size span is unrelated prose, and
    its size says nothing about whether the small span is raised."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "A sentence of ordinary prose here", fontsize=11)
    page.insert_text((460, 100), "xyz", fontsize=7)
    assert "^xyz" not in page_text_with_math(page)


def test_overprinted_bold_text_is_not_duplicated():
    """Textbooks fake bold by drawing the same text several times a fraction
    of a point apart. Concatenating the copies turned "2.1 INTRODUCTION"
    into a 90-character run that failed every heading check, which cost four
    pages of a chapter their curriculum tags."""
    doc = pymupdf.open()
    page = doc.new_page()
    for offset in (0.0, 0.2, 0.4, 0.6, 0.8):
        page.insert_text((72 + offset, 100), "2.1  INTRODUCTION", fontsize=11)
    text = page_text_with_math(page).strip()
    assert text == "2.1  INTRODUCTION"


def test_genuinely_repeated_text_further_apart_is_kept():
    """Two identical values in a table are not an overprint."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "5.0", fontsize=11)
    page.insert_text((200, 100), "5.0", fontsize=11)
    assert page_text_with_math(page).count("5.0") == 2


def test_paragraph_breaks_survive_extraction():
    """The chunker splits on blank lines, and section headings are detected
    from them. Joining every line with a single newline left a whole page as
    one paragraph and section detection fell to zero."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "First paragraph line one.", fontsize=11)
    page.insert_text((72, 114), "Still the first paragraph.", fontsize=11)
    page.insert_text((72, 160), "A new paragraph after a gap.", fontsize=11)
    assert "\n\n" in page_text_with_math(page)
