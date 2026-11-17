"""Cutting a textbook figure out of its PDF page, as a picture.

A lesson's "picture it" step shows the textbook's own figure rather than one
drawn by the model: the real one is accurate, and a drawing made up from
the text gets details wrong.

NCERT figures are mostly vector drawings, not embedded pictures, so they
cannot be extracted as files; the page region is rendered instead. The
region is found from the caption: "FIGURE 2.24 ..." sits under its figure,
so the figure is what lies above the caption, in the caption's column, up
to the paragraph before it. A figure that cannot be found that way gives
None, and the lesson keeps its words alone.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

LABEL = re.compile(r"\bFIG(?:URE|\.)?\s*(\d+\.\d+)\b", re.IGNORECASE)

# Raised whenever the cutting rules change, so figures kept on disk -- and
# figures remembered as impossible to cut -- are cut again by the new rules.
VERSION = 2

# A text block with at least this many characters is a paragraph, not a
# label inside a figure ("Conductor 1", "+Q").
PARAGRAPH_CHARS = 80
# The tallest a figure may be, in points (a page is about 780).
MAX_FIGURE_HEIGHT = 420
DPI = 150


def labels_in(text: str) -> list[str]:
    """Figure numbers mentioned in a passage: "FIGURE 2.24" -> "2.24"."""
    return list(dict.fromkeys(LABEL.findall(text or "")))


def render(pdf: Path, label: str, page_hint: int | None = None) -> bytes | None:
    """PNG of the figure with this number, or None if it cannot be found
    cleanly. page_hint (1-based) is searched first."""
    try:
        import pymupdf
    except ImportError:
        log.warning("PyMuPDF is not available; figures cannot be shown")
        return None
    try:
        doc = pymupdf.open(pdf)
    except Exception:
        log.warning("Could not open %s for figure %s", pdf, label, exc_info=True)
        return None
    with doc:
        order = list(range(len(doc)))
        if page_hint and 1 <= page_hint <= len(doc):
            # The cited page and its neighbours first: a passage can run
            # across a page break.
            near = [page_hint - 1, page_hint, page_hint - 2]
            order = [i for i in near if 0 <= i < len(doc)] + [i for i in order if i not in near]
        for i in order:
            png = _from_page(doc[i], label)
            if png:
                return png
    return None


def _from_page(page, label: str) -> bytes | None:
    import pymupdf

    found = _caption(page, label)
    if found is None:
        return None
    caption, colour = found

    pieces, labels = _pieces(page, caption, colour)
    # Start from the drawings just above the caption, then take whatever
    # drawing touches what has been taken -- lines, arrows, the (a) and (b)
    # of a figure -- until nothing more does. Labels are added afterwards in
    # one step: growing through text walked down the lines of an equation.
    seed = pymupdf.Rect(caption.x0 - 150, caption.y0 - SEED_REACH, caption.x1 + 150, caption.y0 + 2)
    box = None
    taken = set()
    for i, r in enumerate(pieces):
        if r.intersects(seed):
            box = r if box is None else box | r
            taken.add(i)
    if box is None:
        return None
    grew = True
    while grew:
        grew = False
        reach = box + (-GAP, -GAP, GAP, GAP)
        # A figure's (a) and (b) can stand further apart side by side than
        # its parts ever do: a drawing level with the figure, wholly within
        # its height, is taken from further off.
        beside = box + (-SIDE_GAP, 0, SIDE_GAP, 0)
        for i, r in enumerate(pieces):
            if i in taken or r.y0 >= caption.y1:
                continue
            level = r.y0 >= box.y0 - GAP and r.y1 <= box.y1 and r.intersects(beside)
            if r.intersects(reach) or level:
                box |= r
                taken.add(i)
                grew = True

    reach = box + (-GAP, -GAP, GAP, GAP)
    for r in labels:
        if r.intersects(reach) and r.y0 < caption.y1:
            box |= r

    box |= caption
    box = _trim_paragraph_above(page, box, caption)
    page_area = page.rect.width * page.rect.height
    if box is None or box.height < 40 or box.width < 60:
        return None                       # a stray mark, not a figure
    if box.height > MAX_FIGURE_HEIGHT or box.width * box.height > 0.45 * page_area:
        return None                       # took in too much to be one figure
    if _covers_paragraph(page, box, caption):
        return None                       # a picture with body text in it is wrong
    if _has_equation(page, box):
        return None                       # an equation beside it was taken in
    if _other_caption(page, box, label):
        return None                       # figures side by side, taken as one
    if _half_of_pair(page, box, caption):
        return None                       # its (b) without its (a)
    box = (box + (-6, -6, 6, 6)) & page.rect
    box.y0 = max(box.y0, HEADER_END)     # never the running header
    pix = page.get_pixmap(clip=box, dpi=DPI)
    _blank(pix, box, _not_figure(page, caption))
    return pix.tobytes("png")


def _not_figure(page, caption):
    """What may fall inside the picture's rectangle without being part of
    the figure: each line of body text beside it, and the page number box.
    Painted white rather than cut away, so the figure itself keeps every
    part. Lines, not whole paragraphs: a paragraph's outline can reach over
    a figure that its text wraps around."""
    import pymupdf

    out = list(_page_number_boxes(page))
    for b in page.get_text("dict")["blocks"]:
        lines = b.get("lines", [])
        text = "".join(sp["text"] for ln in lines for sp in ln["spans"]).strip()
        if pymupdf.Rect(b["bbox"]).intersects(caption):
            continue
        for ln in lines:
            line = "".join(sp["text"] for sp in ln["spans"]).strip()
            # A paragraph's lines, and any line too long to be a label: body
            # text wrapped round a figure is often stored a line at a time.
            if len(text) >= PARAGRAPH_CHARS or len(line) > MAX_LABEL_CHARS:
                out.append(pymupdf.Rect(ln["bbox"]) + (-1, -1, 1, 1))
    return out


def _blank(pix, clip, rects) -> None:
    """Paint these page rectangles white in a picture of `clip`."""
    import pymupdf

    scale = DPI / 72
    for r in rects:
        part = r & clip
        if part.is_empty:
            continue
        # A clipped pixmap keeps its place on the page: its pixel
        # coordinates start at (pix.x, pix.y), not at 0.
        ir = pymupdf.IRect(
            pix.x + int((part.x0 - clip.x0) * scale), pix.y + int((part.y0 - clip.y0) * scale),
            pix.x + int((part.x1 - clip.x0) * scale) + 1, pix.y + int((part.y1 - clip.y0) * scale) + 1,
        ) & pix.irect
        if not ir.is_empty:
            pix.set_rect(ir, (255, 255, 255) if pix.n < 4 else (255, 255, 255, 255))


# Captions are printed in their own colour; the figure starts within this
# many points above one, and its parts are no further apart than GAP.
SEED_REACH = 60
GAP = 20
SIDE_GAP = 60
# Page furniture, measured on the NCERT pages: the running header -- title
# box, rules, print marks -- ends at about y=96 points; below the text area
# there are only print marks. Header pieces were taken into figures printed
# near the top of a page, and narrow ones let the growing climb into the
# header. A figure's own narrow parts may still reach up into the band.
HEADER_END = 96
FOOTER_START = 45      # from the bottom edge
# The longest a label inside a figure is ("Conductor 1", "Silk thread",
# "(a)"). Longer short lines are body text in a narrow column, and taking
# them in pulled whole paragraphs and equations into a picture.
MAX_LABEL_CHARS = 30
# The widest symbol drawn into a sentence between two pieces of its text.
INLINE_GAP = 20
# A figure in two parts is never shorter than this, above its caption.
MIN_PAIR_HEIGHT = 80
# "(2.20)" beside an equation, level with a figure, is not one of its labels.
EQUATION_NUMBER = re.compile(r"^\(\d+\.\d+\)$")


def _caption(page, label: str):
    """The caption's full extent: its "FIGURE 2.24" span and every line
    below it in the same colour. "FIGURE" in capitals only: body text says
    "Figure 1.20 is self-explanatory", and once passed for a caption."""
    import pymupdf

    wanted = re.compile(rf"^\s*FIGURE\s*{re.escape(label)}\b")
    lines = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            spans = [sp for sp in line["spans"] if sp["text"].strip()]
            if spans:
                lines.append((pymupdf.Rect(line["bbox"]), spans[0]["color"],
                              "".join(sp["text"] for sp in spans)))
    start = next((ln for ln in lines if wanted.match(ln[2])), None)
    if start is None:
        return None
    # Returns the caption's colour too: other captions on the page share it,
    # and must not be taken into this figure.
    box, colour = pymupdf.Rect(start[0]), start[1]
    for rect, c, _ in sorted(lines, key=lambda ln: ln[0].y0):
        # The caption's later lines: same colour, just below, overlapping it.
        if (c == colour and rect.y0 >= box.y0 and rect.y0 <= box.y1 + 4
                and rect.x0 < box.x1 + 20 and rect.x1 > box.x0 - 20):
            box |= rect
    return box, colour


def _pieces(page, caption, colour):
    """The drawings and pictures a figure is made of, and the short text
    labels that may sit on it."""
    import pymupdf

    w, h = page.rect.width, page.rect.height

    furniture = _page_number_boxes(page)

    def keep(r):
        # Not the watermark across the page, not the header's banner and
        # rules, not the page number, not the caption itself.
        header = r.y1 <= HEADER_END + 1 or (r.y0 < HEADER_END and r.width > 150) or r.y0 < 0
        footer = r.y0 > h - FOOTER_START
        return (r.is_valid and not r.is_empty and r.width < 0.6 * w and r.height < 0.5 * h
                and not header and not footer
                and not any(r.intersects(f) for f in furniture)
                and not r.intersects(caption))

    # A symbol drawn inside a line of body text (a vector arrow, a hat) is
    # part of the sentence, not of a figure: taken in, it pulled the
    # paragraph beside a figure into the picture.
    body = _paragraph_lines(page, caption)
    drawn = [d["rect"] for d in page.get_drawings()]
    drawn = [r for r in drawn if not _in_sentence(r, body)]
    drawn += [page.get_image_bbox(img) for img in page.get_images(full=True)]
    labels = []
    for b in page.get_text("dict")["blocks"]:
        lines = b.get("lines", [])
        text = "".join(sp["text"] for ln in lines for sp in ln["spans"]).strip()
        colours = {sp["color"] for ln in lines for sp in ln["spans"] if sp["text"].strip()}
        # A short label, and not another figure's caption: the end of the
        # caption above once headed the next figure's picture.
        if (text and len(text) <= MAX_LABEL_CHARS and colour not in colours
                and not EQUATION_NUMBER.match(text)):
            labels.append(pymupdf.Rect(b["bbox"]))
    return [r for r in drawn if keep(r)], [r for r in labels if keep(r)]


def _in_sentence(r, body) -> bool:
    """A drawing inside a line of body text, or in the gap between two
    pieces of one line ("placed in the [y-z] plane")."""
    level = [ln for ln in body if r.y0 >= ln.y0 - 2 and r.y1 <= ln.y1 + 2]
    if any(ln.x0 - 1 <= r.x0 and r.x1 <= ln.x1 + 1 for ln in level):
        return True
    left = any(-2 <= r.x0 - ln.x1 < INLINE_GAP for ln in level)
    right = any(-2 <= ln.x0 - r.x1 < INLINE_GAP for ln in level)
    return left and right


def _page_number_boxes(page):
    """The page number and its box: digits alone, near the side of the page,
    in the lower half. It sat beside captions and came along with them."""
    import pymupdf

    w, h = page.rect.width, page.rect.height
    out = []
    for b in page.get_text("blocks"):
        r = pymupdf.Rect(b[:4])
        if b[4].strip().isdigit() and len(b[4].strip()) <= 3 and r.y0 > h / 2 \
                and (r.x0 > 0.75 * w or r.x1 < 0.25 * w):
            out.append(r + (-26, -10, 26, 10))
    return out


def _paragraph_lines(page, caption):
    """Each line of body text on the page, outside the caption. Lines, not
    blocks: a paragraph wrapped round a figure has a block outline reaching
    over the figure, though none of its lines does."""
    import pymupdf

    out = []
    for b in page.get_text("dict")["blocks"]:
        lines = b.get("lines", [])
        text = "".join(sp["text"] for ln in lines for sp in ln["spans"]).strip()
        if len(text) < PARAGRAPH_CHARS or pymupdf.Rect(b["bbox"]).intersects(caption):
            continue
        out += [pymupdf.Rect(ln["bbox"]) for ln in lines]
    return out


def _trim_paragraph_above(page, box, caption):
    """Cut off body text the figure's top edge ran into: lines of a
    paragraph in the top quarter of the figure end where the figure starts."""
    import pymupdf

    for r in _paragraph_lines(page, caption):
        if not r.intersects(box):
            continue
        if (r.y0 + r.y1) / 2 < box.y0 + 0.25 * box.height and r.y1 + 2 < caption.y0:
            box = pymupdf.Rect(box.x0, r.y1 + 2, box.x1, box.y1)
    return box if box.is_valid and not box.is_empty else None


def _other_caption(page, box, label) -> bool:
    """Another figure's caption inside the picture: two figures printed side
    by side grew into one picture, which would show under either number."""
    import pymupdf

    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        for m in re.finditer(r"FIGURE\s*(\d+\.\d+)", text):
            if m.group(1) != label and pymupdf.Rect(x0, y0, x1, y1).intersects(box):
                return True
    return False


def _half_of_pair(page, box, caption) -> bool:
    """A "(b)" in the picture with no "(a)": only part of the figure was
    found, and half a figure misleads more than none."""
    import pymupdf

    inside = {w[4] for w in page.get_text("words")
              if pymupdf.Rect(w[:4]).intersects(box) and not pymupdf.Rect(w[:4]).intersects(caption)}
    if "(b)" in inside and "(a)" not in inside:
        return True
    # Some figures' labels are drawn, not text, so the words above cannot
    # tell. A caption naming an (a) and a (b) over a sliver of drawing is the
    # same mistake: only the (b) nearest the caption was found.
    said = {w[4] for w in page.get_text("words") if pymupdf.Rect(w[:4]).intersects(caption)}
    return "(a)" in said and "(b)" in said and caption.y0 - box.y0 < MIN_PAIR_HEIGHT


def _has_equation(page, box) -> bool:
    """An equation number such as "(2.31)" inside the picture: the equation
    it numbers came along too, and its parts look like a figure's."""
    import pymupdf

    for b in page.get_text("blocks"):
        if EQUATION_NUMBER.match(b[4].strip()) and pymupdf.Rect(b[:4]).intersects(box):
            return True
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        if EQUATION_NUMBER.match(word) and pymupdf.Rect(x0, y0, x1, y1).intersects(box):
            return True
    return False


def _covers_paragraph(page, box, caption) -> bool:
    """Any real share of body text inside the picture: running down its
    side, or through it. Small slivers at an edge are allowed."""
    covered = 0.0
    for r in _paragraph_lines(page, caption):
        overlap = r & box
        if not overlap.is_empty:
            covered += overlap.width * overlap.height
    return covered > 0.03 * box.width * box.height
