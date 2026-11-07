"""Rebuild mathematical notation from the geometry of the page.

A textbook writes Coulomb's law as a stacked fraction: the numerator above a
drawn rule, the denominator below it, superscripts raised and shrunk. Every
text extractor flattens that into one line in reading order, so

    F = k |q1 q2| / r^2

arrives as "F = k |q1 q2| r2" or worse, with the pieces interleaved. The
characters all survive; only the spatial relationship that carried the meaning
is lost. A student reading the result learns nothing, and no amount of
retrieval tuning can repair it after the fact.

The relationship is recoverable, because PDFs record where every glyph sits.

* A fraction is a short horizontal rule with text directly above and directly
  below it, within the rule's horizontal span. Requiring text on *both* sides
  is what separates a fraction bar from an underline, a table border or a
  page rule.
* A superscript or subscript is a span set in a noticeably smaller font than
  its neighbours, with its baseline raised or dropped.

This handles single-level fractions and one level of script, which covers
essentially all of school physics and chemistry. Nested fractions, integrals
and matrices are not reconstructed, and are left as the extractor returned
them rather than guessed at.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# A fraction bar is short. Page rules and table borders run much wider.
MIN_BAR_WIDTH = 6.0
MAX_BAR_WIDTH = 260.0

# Measured on a real NCERT page: the numerator clears the bar by 0.25 of its
# height and the denominator by 0.29, while the paragraph after an underlined
# heading sits 0.48 away. The window is narrower than it looks, which is why
# centring below does most of the work of telling them apart.
MAX_GAP_RATIO = 0.45

# Each half of a fraction is roughly as wide as the bar. An underlined
# heading is followed by prose far wider than the rule above it.
MAX_HALF_WIDTH_RATIO = 1.6

# A fraction is centred on its bar; body text is not. On a real page the
# numerator's centre sat 0.8pt from a 32pt bar's centre, while a paragraph
# under an underlined heading sat 48pt off a 128pt rule. This separates the
# two far more cleanly than vertical distance does.
MAX_CENTRE_OFFSET_RATIO = 0.25

# A script is set smaller than the text it attaches to.
SCRIPT_SIZE_RATIO = 0.86

# How far a baseline must shift before it counts as raised or dropped.
SCRIPT_OFFSET_RATIO = 0.16

# Baselines on the same visual line agree to within a point or two.
BASELINE_TOLERANCE = 3.0

# How far apart two pieces of the same visual line may sit before they are
# treated as separate. An equation is routinely split across layout blocks --
# on a real page "F k q1q2" and "r = (1.1)" came back as two -- so the pieces
# must be allowed to rejoin. A column gutter is far wider than this, so the
# columns stay apart.
MAX_INLINE_GAP = 40.0

# Longest run that can plausibly be a script. Guards against small capitals,
# where a shrunken "XAMPLE" otherwise reads as a subscript of "E".
MAX_SCRIPT_CHARS = 4

# Operators and punctuation are never scripts. In a display equation the
# whole expression is often set smaller than the prose around it, so size
# alone marked every character in it: Ampere's law came back as
# "B ^. _d ^l _= _µ _0 I".
NEVER_SCRIPT = set("=.,;:\u00b7\u2219\u00d7\u00f7\u00b1\u2213\u2192\u2190\u2194\u222b\u222e\u2211\u220f\u221a\u2248\u2260\u2264\u2265\u2261\u221d/\\|()[]{}'\u2032\u2033")

# How far a script may sit from the character it attaches to. Beyond this
# the nearest full-size span is unrelated prose on the same line, and its
# size says nothing about whether the small span is raised or dropped.
MAX_ANCHOR_DISTANCE = 40.0


@dataclass
class Span:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    # Which layout block the glyph came from. Textbook pages are set in two
    # columns, and the left and right columns share baselines -- grouping on
    # baseline alone spliced them into single lines and scrambled the prose
    # into nonsense. Blocks already respect column structure.
    block: int = 0
    # True for a rebuilt fraction. Only lines carrying one may reach across
    # layout blocks; prose must not, or paragraphs interleave and the page
    # reads as nonsense.
    is_math: bool = False
    consumed: bool = False

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)


@dataclass
class Bar:
    x0: float
    x1: float
    y: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass
class Line:
    """A visual line of the page, rebuilt from spans."""

    spans: list[Span] = field(default_factory=list)

    @property
    def cy(self) -> float:
        return sum(s.cy for s in self.spans) / len(self.spans)

    @property
    def baseline(self) -> float:
        """Shared bottom edge of the full-size text on this line."""
        return max(s.y1 for s in self.spans)


def collect_spans(page) -> list[Span]:
    spans: list[Span] = []
    for index, block in enumerate(page.get_text("dict", sort=True).get("blocks", [])):
        for line in block.get("lines", []):
            for raw in line.get("spans", []):
                text = raw.get("text", "")
                if not text.strip():
                    continue
                x0, y0, x1, y1 = raw["bbox"]
                spans.append(
                    Span(
                        text=text, x0=x0, y0=y0, x1=x1, y1=y1,
                        size=raw.get("size", 0.0), block=index,
                    )
                )
    return spans


def collect_bars(page) -> list[Bar]:
    """Horizontal rules that are short enough to be fraction bars."""
    bars: list[Bar] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        log.debug("No drawing information available on this page")
        return bars

    for drawing in drawings:
        for item in drawing.get("items", []):
            kind = item[0]
            if kind == "l":
                start, end = item[1], item[2]
                x0, y0, x1, y1 = start.x, start.y, end.x, end.y
            elif kind == "re":
                rect = item[1]
                x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                # A filled rectangle acting as a rule is very thin.
                if (y1 - y0) > 2.5:
                    continue
            else:
                continue

            if abs(y1 - y0) > 1.5:
                continue
            width = abs(x1 - x0)
            if not (MIN_BAR_WIDTH <= width <= MAX_BAR_WIDTH):
                continue
            bars.append(Bar(x0=min(x0, x1), x1=max(x0, x1), y=(y0 + y1) / 2))

    return bars


def _overlaps_horizontally(span: Span, bar: Bar) -> bool:
    """The span must sit within the bar's horizontal span, allowing a little
    overhang for italic glyphs and bracket bars."""
    margin = 3.0
    return span.x1 > bar.x0 - margin and span.x0 < bar.x1 + margin


def _halves(spans: list[Span], bar: Bar) -> tuple[list[Span], list[Span]]:
    above: list[Span] = []
    below: list[Span] = []

    for span in spans:
        if span.consumed or not _overlaps_horizontally(span, bar):
            continue
        limit = span.height * MAX_GAP_RATIO
        # A subscript hanging off the numerator can cross the bar slightly,
        # so a strictly non-negative gap drops it into a crack between the
        # two halves and it ends up stranded outside the fraction entirely.
        # Which side it belongs to is decided by its centre.
        slack = -span.height * 0.5
        if span.cy < bar.y and slack <= bar.y - span.y1 <= limit:
            above.append(span)
        elif span.cy > bar.y and slack <= span.y0 - bar.y <= limit:
            below.append(span)

    return above, below


def _half_fits_bar(spans: list[Span], bar: Bar) -> bool:
    """A fraction's half is about as wide as the rule, and centred on it."""
    if not spans:
        return False

    left = min(s.x0 for s in spans)
    right = max(s.x1 for s in spans)
    reference = max(bar.width, MIN_BAR_WIDTH)

    if right - left > reference * MAX_HALF_WIDTH_RATIO:
        return False

    centre_offset = abs((left + right) / 2 - (bar.x0 + bar.x1) / 2)
    return centre_offset <= reference * MAX_CENTRE_OFFSET_RATIO



def join_spans(spans: list[Span]) -> str:
    """Concatenate spans in reading order, exactly as the PDF spaces them.

    Each span already carries its own spacing: a small-capitals heading
    arrives as "1.2  E", "LECTRIC", " C", "HARGE", with the word break
    sitting inside the third span. Stripping each one and joining with a
    space destroyed that -- "ELECTRIC CHARGE" became "I NTRODUCTION"-style
    wreckage, and section headings stopped being searchable. Geometry cannot
    recover it either, because the gap between those spans is genuinely zero.
    """
    ordered = sorted(spans, key=lambda s: s.x0)
    return "".join(s.text for s in ordered).strip()


def _join(spans: list[Span]) -> str:
    return join_spans(spans)


def reconstruct_fractions(spans: list[Span], bars: list[Bar]) -> list[Span]:
    """Replace each stacked fraction with a single inline span."""
    rebuilt: list[Span] = []

    for bar in sorted(bars, key=lambda b: (b.y, b.x0)):
        above, below = _halves(spans, bar)
        # Both halves required, and both must fit the bar. An underline has a
        # heading above and a full paragraph below, which fails on width even
        # when something happens to sit close enough underneath.
        if not above or not below:
            continue
        if not _half_fits_bar(above, bar) or not _half_fits_bar(below, bar):
            continue

        numerator = _join(above)
        denominator = _join(below)
        if not numerator or not denominator:
            continue

        for span in above + below:
            span.consumed = True

        # A display fraction is centred on the maths axis rather than
        # sitting on the text baseline, so the rebuilt span is given the
        # baseline the surrounding equation uses -- roughly half a body
        # height below the bar. Without this it lands on a line of its own
        # and "F = k" ends up separated from the fraction it introduces.
        body_height = max(s.height for s in above + below)
        rebuilt.append(
            Span(
                text=f" ({numerator})/({denominator}) ",
                x0=bar.x0,
                y0=bar.y - body_height,
                x1=bar.x1,
                y1=bar.y + body_height * 0.45,
                size=max(s.size for s in above + below),
                block=above[0].block,
                is_math=True,
            )
        )

    return rebuilt


def _group_into_lines(spans: list[Span]) -> list[Line]:
    """Group spans into visual lines by shared baseline.

    Two earlier attempts failed on the same page. Centre distance orphaned
    superscripts, because a raised, shrunken glyph sits further from the body
    centre than line spacing allows. Vertical overlap chained instead: each
    span added widened the line's extent, so the numerator reached the
    denominator through the subscripts between them, and a whole stacked
    fraction collapsed into one line along with the equation number in the
    right margin.

    Baselines do not chain. Full-size text on one visual line shares a
    baseline within a point or two, and the row below has a clearly different
    one. Scripts are then attached to the nearest full-size span, which is
    the character they actually belong to.
    """
    if not spans:
        return []

    sizes = sorted(s.size for s in spans if s.size > 0)
    median_size = sizes[len(sizes) // 2] if sizes else 0.0
    # Small *and* short. Footers, captions and marginal labels are set small
    # too, and treating them as scripts attached them to whichever line
    # happened to be nearest -- which is how "Reprint 2026-27" ended up
    # spliced into the middle of a sentence about Coulomb's experiments.
    def is_script(span: Span) -> bool:
        return (
            span.size < median_size * SCRIPT_SIZE_RATIO
            and len(span.text.strip()) <= MAX_SCRIPT_CHARS
        )

    body = [s for s in spans if not is_script(s)]
    scripts = [s for s in spans if is_script(s)]

    if not body:
        body, scripts = spans, []

    lines: list[Line] = []
    for span in sorted(body, key=lambda s: (s.y1, s.x0)):
        for line in lines:
            if abs(line.baseline - span.y1) > BASELINE_TOLERANCE:
                continue
            same_block = any(existing.block == span.block for existing in line.spans)
            if same_block:
                line.spans.append(span)
                break

            # Reaching across blocks is reserved for equations. An equation is
            # routinely split -- on a real page "F k q1q2" and "r = (1.1)"
            # arrived as separate blocks -- but letting prose do the same
            # interleaved two paragraphs and spliced a page footer into the
            # middle of a sentence.
            involves_math = span.is_math or any(e.is_math for e in line.spans)
            if not involves_math:
                continue

            gap = min(
                max(existing.x0 - span.x1, span.x0 - existing.x1, 0.0)
                for existing in line.spans
            )
            if gap <= MAX_INLINE_GAP:
                line.spans.append(span)
                break
        else:
            lines.append(Line(spans=[span]))

    # Attach each script to the line holding its nearest full-size span.
    for script in scripts:
        nearest_line = None
        nearest_distance = float("inf")
        for line in lines:
            for candidate in line.spans:
                dx = candidate.cx - script.cx
                dy = candidate.cy - script.cy
                distance = (dx * dx + dy * dy) ** 0.5
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_line = line
        if nearest_line is not None:
            nearest_line.spans.append(script)
        else:
            lines.append(Line(spans=[script]))

    return lines


def mark_scripts(line: Line) -> None:
    """Annotate raised and dropped spans with ^ and _ .

    Without this, 10⁹ arrives as "10 9" and r² as "r2" -- values that look
    ordinary and read wrong. A student revising from "9 × 109 N" has a number
    a thousand million times too small.

    Each small span is judged against its nearest full-size neighbour rather
    than against the line as a whole. Lines run the width of the page, so an
    equation number sitting in the right margin joins the line and drags the
    baseline down with it, at which point every subscript on the line looks
    raised.
    """
    sizes = [s.size for s in line.spans if s.size > 0]
    if len(sizes) < 2:
        return

    body_size = max(sizes)
    body_spans = [s for s in line.spans if s.size >= body_size * SCRIPT_SIZE_RATIO]
    if not body_spans:
        return

    # Section headings are set in small capitals, so every word after the
    # first letter is smaller text sitting on the same baseline. Marked as
    # scripts, "APPLICATIONS OF GAUSS'S LAW" becomes
    # "A PPLICATIONS _OF G _AUSS _S L _AW" -- and headings are exactly what a
    # student searches for. Several uppercase candidates on one line means
    # small capitals, not mathematics.
    uppercase_candidates = [
        s
        for s in line.spans
        if s.size < body_size * SCRIPT_SIZE_RATIO
        and s.text.strip().isalpha()
        and s.text.strip().isupper()
    ]
    small_caps_line = len(uppercase_candidates) >= 2

    for span in line.spans:
        if span.size <= 0 or span.size >= body_size * SCRIPT_SIZE_RATIO:
            continue
        if span.text.startswith(("^", "_", "(")):
            continue

        # Small capitals are set smaller than the initial letter, so
        # "EXAMPLE" arrives as a large E followed by a shrunken "XAMPLE" and
        # gets marked "E _XAMPLE". A real script is a digit or a symbol or
        # two, never a word.
        if len(span.text.strip()) > MAX_SCRIPT_CHARS:
            continue
        if small_caps_line and span.text.strip().isalpha() and span.text.strip().isupper():
            continue
        if all(c in NEVER_SCRIPT for c in span.text.strip()):
            continue

        anchor = min(body_spans, key=lambda b: abs(b.cx - span.cx))
        if abs(anchor.cx - span.cx) > MAX_ANCHOR_DISTANCE:
            continue
        threshold = anchor.height * SCRIPT_OFFSET_RATIO

        core = span.text.strip()
        lead = span.text[: len(span.text) - len(span.text.lstrip())]
        trail = span.text[len(span.text.rstrip()) :]

        if anchor.y1 - span.y1 > threshold:
            span.text = f"{lead}^{core}{trail}"
        elif span.y1 - anchor.y1 > -threshold and span.y0 - anchor.y0 > threshold:
            span.text = f"{lead}_{core}{trail}"


def page_text_with_math(page) -> str:
    """Extract a page, rebuilding fractions and scripts from its geometry."""
    spans = collect_spans(page)
    if not spans:
        return ""

    # Scripts first: a superscript inside a numerator must already be marked
    # before the fraction is collapsed into a single span, or it is lost.
    for line in _group_into_lines(spans):
        mark_scripts(line)

    bars = collect_bars(page)
    fractions = reconstruct_fractions(spans, bars) if bars else []

    remaining = [s for s in spans if not s.consumed] + fractions
    lines = _group_into_lines(remaining)

    out: list[str] = []
    # Block order first: it follows the columns, so the left column is read
    # all the way down before the right one begins.
    for line in sorted(lines, key=lambda l: (min(s.block for s in l.spans), l.cy)):
        text = join_spans(line.spans)
        if text:
            out.append(text)

    return "\n".join(out)
