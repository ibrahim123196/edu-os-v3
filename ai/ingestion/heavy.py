"""Pages whose drawings are too many to read one by one.

Most textbook pages hold a few hundred drawing commands: rules, arrows, the
lines of a diagram. A page whose illustration was traced into vectors can
hold millions -- one page of NCERT Biology's Human Reproduction chapter has a
25 MB (compressed) drawing stream. Listing those drawings in Python took
more than 8 GB of memory and the whole ingestion was killed.

On such a page the drawings are artwork, not fraction bars or a diagram's
lines worth measuring, so the code that reads drawings skips it: formulas
come from the text alone, and its figure is not cut.
"""

from __future__ import annotations

# Compressed bytes of a page's drawing commands. The largest ordinary page
# seen across nine NCERT chapters is about 2 MB; the traced one, 25 MB.
MAX_DRAWING_BYTES = 8_000_000


def drawing_bytes(page) -> int:
    """Size of the page's content streams as stored, without unpacking them."""
    doc = page.parent
    total = 0
    for xref in page.get_contents():
        kind, value = doc.xref_get_key(xref, "Length")
        if kind == "int":
            total += int(value)
    return total


def too_heavy(page) -> bool:
    try:
        return drawing_bytes(page) > MAX_DRAWING_BYTES
    except Exception:
        return False
