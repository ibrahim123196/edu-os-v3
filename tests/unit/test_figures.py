"""Cutting a textbook figure out of a PDF page.

The real NCERT chapters are not in the repository, so these build a small
page laid out the same way: a drawing, its caption in colour under it, body
text around. The cropper was tuned by eye on the real chapters (78 of 105
figures in chapters 1-4 cut; the rest refused, never cut wrong as far as
was seen); these keep its rules from drifting.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")

from ai.ingestion.figures import labels_in, render  # noqa: E402

BLUE = (0, 0.6, 0.85)
BODY = ("The two plates face each other across a thin gap, and the charge on "
        "one is equal and opposite to the charge on the other plate here.")


def make_pdf(path: Path, label: str = "2.24", caption: bool = True, pages_before: int = 0,
             equation: bool = False, second_caption: str | None = None) -> Path:
    doc = pymupdf.open()
    for _ in range(pages_before):
        doc.new_page(width=595, height=842).insert_text((80, 200), BODY[:60])
    page = doc.new_page(width=595, height=842)
    page.insert_text((80, 140), BODY[:70], fontsize=10)
    # The figure: two plates and a field line between them.
    page.draw_rect(pymupdf.Rect(200, 200, 210, 320), color=(0, 0, 0), fill=BLUE)
    page.draw_rect(pymupdf.Rect(330, 200, 340, 320), color=(0, 0, 0), fill=BLUE)
    page.draw_line((212, 260), (328, 260), color=(0, 0, 0))
    page.insert_text((190, 335), "+Q", fontsize=9)
    page.insert_text((330, 335), "-Q", fontsize=9)
    if equation:
        page.draw_rect(pymupdf.Rect(380, 239, 420, 241), fill=(0, 0, 0))   # a fraction bar
        page.insert_text((405, 245), "(2.31)", fontsize=10)
    if caption:
        page.insert_text((200, 360), f"FIGURE {label} Two plates.", fontsize=9, color=BLUE)
    else:
        page.insert_text((200, 360), f"Figure {label} is self-explanatory.", fontsize=9)
    if second_caption:                      # a second figure, close beside the first
        page.draw_rect(pymupdf.Rect(370, 200, 380, 320), color=(0, 0, 0), fill=BLUE)
        page.insert_text((360, 360), f"FIGURE {second_caption} More.", fontsize=9, color=BLUE)
    for i in range(12):
        page.insert_text((80, 420 + 14 * i), BODY[:90], fontsize=10)
    doc.save(path)
    return path


def test_figure_numbers_in_a_passage():
    assert labels_in("See FIGURE 2.24 and Fig. 2.3, then Figure 2.24 again.") == ["2.24", "2.3"]
    assert labels_in("Equation (2.24) has no figure.") == []
    assert labels_in("") == []


def test_the_figure_above_its_caption_is_cut(tmp_path):
    png = render(make_pdf(tmp_path / "c.pdf"), "2.24")
    assert png and png.startswith(b"\x89PNG")
    from PIL import Image
    im = Image.open(io.BytesIO(png))
    # Wider than the drawing alone and not the whole page: drawing, labels
    # and caption, nothing of the body text.
    assert 150 * 150 / 72 < im.width < 400 * 150 / 72
    assert im.height < 300 * 150 / 72


def test_the_page_hint_is_only_a_hint(tmp_path):
    pdf = make_pdf(tmp_path / "c.pdf", pages_before=3)
    assert render(pdf, "2.24", page_hint=1)
    assert render(pdf, "2.24", page_hint=4)
    assert render(pdf, "2.24", page_hint=99)            # out of range: searched anyway


def test_a_figure_that_is_not_there_gives_none(tmp_path):
    assert render(make_pdf(tmp_path / "c.pdf"), "2.25") is None


def test_body_text_naming_a_figure_is_not_its_caption(tmp_path):
    """'Figure 1.20 is self-explanatory' once passed for a caption."""
    assert render(make_pdf(tmp_path / "c.pdf", caption=False), "2.24") is None


def test_an_equation_taken_in_refuses_the_figure(tmp_path):
    assert render(make_pdf(tmp_path / "c.pdf", equation=True), "2.24") is None


def test_two_figures_grown_into_one_are_refused(tmp_path):
    """Side by side, either number would show both."""
    assert render(make_pdf(tmp_path / "c.pdf", second_caption="2.25"), "2.24") is None


def test_a_file_that_is_not_a_pdf_gives_none(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    assert render(bad, "2.24") is None
    assert render(tmp_path / "missing.pdf", "2.24") is None


def test_a_symbol_drawn_inside_a_sentence_is_not_part_of_the_figure(tmp_path):
    """A vector arrow drawn between two pieces of a line of body text, level
    with the figure, once pulled the paragraph into the picture and the
    figure was refused (Figure 4.9)."""
    path = tmp_path / "c.pdf"
    make_pdf(path)
    doc = pymupdf.open(path)
    page = doc[0]
    left = "The loop is placed in the"
    end = 40 + pymupdf.get_text_length(left, fontsize=10)
    for i in range(6):                      # body text beside the figure
        y = 220 + 14 * i
        page.insert_text((40, y), left, fontsize=10)
        page.draw_rect(pymupdf.Rect(end + 2, y - 8, end + 10, y), fill=(0, 0, 0))   # the symbol
        page.insert_text((end + 12, y), "plane", fontsize=10)
    doc.saveIncr()
    png = render(path, "2.24")
    assert png
    from PIL import Image
    assert Image.open(io.BytesIO(png)).width < 170 * 150 / 72       # not reaching the text
