"""Strip running headers, footers and page furniture.

Textbooks repeat the same line on every page: a reprint notice, the subject
name, the page number. pypdf returns these inside the page text, where they do
real damage:

* they land in the middle of chunks, so a retrieved passage reads
  "...follow the same inverse-square law. Reprint 2026-27 8 Physics 9 x 109 N."
* they look like headings to the heading detector, so chunks end up labelled
  "Reprint 2026-27" instead of "Coulomb's Law"
* they enter the spelling vocabulary, where "reprint" is a common word that
  no student will ever ask about

They are easy to find without hard-coding anything: boilerplate is text that
repeats across many pages, while real content does not. A line appearing on
more than a third of the pages is furniture.

Page numbers need separate handling, since they differ on every page and so
never repeat. They are matched by shape instead: a short line that is mostly
digits, at the very start or end of a page.
"""

from __future__ import annotations

import re
from collections import Counter

from ai.ingestion.extract import Page

# Fraction of pages a line must appear on before it counts as furniture.
REPEAT_THRESHOLD = 0.34

# Below this many pages, repetition means nothing.
MIN_PAGES = 4

# Furniture is short. A repeated long line is more likely a real refrain.
MAX_BOILERPLATE_WORDS = 6

# How many lines at the top and bottom of a page count as the margins.
# Running headers and footers live there by definition, and restricting the
# search to those bands stops templated body text ("Example 1.1 ...", which
# normalises to the same shape on every page) being mistaken for furniture.
EDGE_LINES = 3

_PAGE_NUMBER = re.compile(r"^\W*\d{1,4}\W*$")
_HAS_LETTER = re.compile(r"[a-z]")
_DIGIT_HEAVY = re.compile(r"^[\W\d]{1,24}$")


def _normalise(line: str) -> str:
    """Compare lines ignoring the digits that vary from page to page."""
    collapsed = re.sub(r"\d+", "#", line)
    return re.sub(r"\s+", " ", collapsed).strip().lower()


def find_boilerplate(pages: list[Page]) -> set[str]:
    """Normalised lines that repeat across enough pages to be furniture."""
    if len(pages) < MIN_PAGES:
        return set()

    seen_on_pages: Counter[str] = Counter()
    for page in pages:
        lines = [line for line in page.text.splitlines() if line.strip()]
        # On a sparse page the top and bottom bands can swallow everything,
        # which would let ordinary body text qualify as furniture. Margins
        # never cover more than half a page.
        edge = min(EDGE_LINES, max(1, len(lines) // 2))
        margins = lines[:edge] + lines[-edge:]
        unique_lines = {
            _normalise(line)
            for line in margins
            if len(line.split()) <= MAX_BOILERPLATE_WORDS
        }
        seen_on_pages.update(unique_lines)

    cutoff = max(2, int(len(pages) * REPEAT_THRESHOLD))
    return {
        line
        for line, count in seen_on_pages.items()
        # Must contain letters. Normalising digits to "#" collapses every
        # bare number to the same string, so without this the page-number
        # line qualifies as furniture and then strips every standalone
        # number in the book -- including answers and constants.
        if count >= cutoff and _HAS_LETTER.search(line)
    }


def strip_boilerplate(pages: list[Page]) -> tuple[list[Page], set[str]]:
    """Remove repeated furniture and bare page numbers from every page."""
    boilerplate = find_boilerplate(pages)

    cleaned: list[Page] = []
    for page in pages:
        lines = page.text.splitlines()
        kept: list[str] = []

        for position, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue

            if _normalise(stripped) in boilerplate:
                continue

            # A bare page number, but only at the edges of the page -- a lone
            # number in the middle of a page is probably an answer or a value.
            at_edge = position < 2 or position >= len(lines) - 2
            if at_edge and (_PAGE_NUMBER.match(stripped) or _DIGIT_HEAVY.match(stripped)):
                continue

            kept.append(line)

        text = "\n".join(kept).strip()
        if text:
            cleaned.append(Page(number=page.number, text=text))

    return cleaned, boilerplate
