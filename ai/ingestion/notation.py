"""Detect formulas, equations and chemical reactions in a passage.

Why this is needed: retrieval built on word overlap is blind to notation.
"F = kq₁q₂/r²" contains almost no English words, so a passage holding the exact
formula a student asked for scores far below a paragraph that merely talks
about it. Embeddings do not rescue this either -- mathematical expressions
tokenize poorly and land nowhere near the prose question in vector space.

The fix is to notice at ingestion time which passages actually contain
notation, store that as a number on the chunk, and let retrieval use it when
the question asks for a formula.

Scoring favours *containing* notation over *being made of* notation. A worked
example that is nine lines of explanation and one line of algebra is usually
the best answer to "what is the formula for X", so one real equation counts
for more than a high density of loose symbols.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# An equation: something on both sides of an equals sign. Requires an
# alphanumeric or closing bracket before, and a value-like token after, so
# prose such as "x = the unknown" does not qualify.
_EQUATION = re.compile(
    r"[A-Za-z0-9\)\]\u2082-\u2089]\s*[=≈≅∝]\s*([-+]?[A-Za-z0-9\(\[√∑∫][A-Za-z0-9]*)"
)

# Prose uses equals signs too: "let x = the unknown quantity". A real equation
# has a value on the right, not an article.
_NOT_A_VALUE = {
    "the", "a", "an", "this", "that", "it", "its", "their", "his", "her",
    "our", "your", "all", "any", "some", "no", "not", "what", "when", "where",
    "how", "we", "you", "they", "he", "she", "is", "are", "was", "were",
}

# Reaction or transformation arrows.
_REACTION = re.compile(r"(?:→|⟶|⇌|⇄|-->|->|=>)")

# Chemical formulas: H2O, CO2, C6H12O6, Ca(OH)2. Needs a digit following a
# capitalised element symbol.
_CHEMICAL = re.compile(r"\b(?:[A-Z][a-z]?\d+){1,}[A-Z]?[a-z]?\b|\([A-Z][a-z]?[A-Z]?[a-z]?\)\d")

# LaTeX survives in some PDF extractions.
_LATEX = re.compile(r"\\(?:frac|sqrt|sum|int|cdot|times|alpha|beta|gamma|delta|pi|theta)\b|[_^]\{")

# Division written inline: v = d/t, mol/L, m/s^2
_INLINE_FRACTION = re.compile(r"\b[A-Za-z0-9]{1,6}\s*/\s*[A-Za-z0-9]{1,6}\b")

_MATH_SYMBOLS = set("∝≈≠≤≥≡∫∑∏√±∓×÷°∞∂∇⟨⟩ħ")
_GREEK = set("ΔΘΛΞΠΣΦΨΩαβγδεζηθικλμνξπρστυφχψω")
_SUPERSCRIPT = set("⁰¹²³⁴⁵⁶⁷⁸⁹⁻ⁿ")
_SUBSCRIPT = set("₀₁₂₃₄₅₆₇₈₉")

# ASCII stand-ins that survive PDF extraction: r^2, x_1, 10^9
_ASCII_POWER = re.compile(r"[A-Za-z0-9]\^-?\d|[A-Za-z]_\d")

# Tiers by how conclusively a signal indicates real notation.
STRONG = {"equation", "reaction", "latex"}
MEDIUM = {"chemical", "exponent"}

# Chunks at or above this are treated as carrying notation worth retrieving.
NOTATION_THRESHOLD = 0.3


@dataclass(frozen=True)
class NotationProfile:
    score: float
    signals: tuple[str, ...]

    @property
    def has_notation(self) -> bool:
        return self.score > 0.0

    @property
    def has_formula(self) -> bool:
        return bool(STRONG & set(self.signals))


def _line_signals(line: str) -> set[str]:
    found: set[str] = set()
    if any(
        m.group(1).lower() not in _NOT_A_VALUE for m in _EQUATION.finditer(line)
    ):
        found.add("equation")
    if _REACTION.search(line):
        found.add("reaction")
    if _LATEX.search(line):
        found.add("latex")
    if _CHEMICAL.search(line):
        found.add("chemical")
    if _ASCII_POWER.search(line) or any(c in _SUPERSCRIPT or c in _SUBSCRIPT for c in line):
        found.add("exponent")
    if any(c in _MATH_SYMBOLS for c in line):
        found.add("symbol")
    if any(c in _GREEK for c in line):
        found.add("greek")
    if _INLINE_FRACTION.search(line):
        found.add("fraction")
    return found


def profile(text: str) -> NotationProfile:
    """Score how much mathematical or chemical notation a passage holds."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return NotationProfile(0.0, ())

    signals: set[str] = set()
    lines_with_notation = 0
    has_strong = False

    for line in lines:
        found = _line_signals(line)
        if found:
            lines_with_notation += 1
            signals |= found
            if found & STRONG:
                has_strong = True

    if not signals:
        return NotationProfile(0.0, ())

    density = lines_with_notation / len(lines)

    # Signals differ in how much they actually mean. An equals sign between
    # two terms is nearly conclusive; a lone degree symbol in a sentence about
    # the weather is not. Scoring them the same made ordinary prose look like
    # a formula sheet, so they are tiered.
    if has_strong:
        # A single real equation outweighs everything else: a worked example
        # that is mostly prose is still the right answer to "what is the
        # formula for X".
        score = 0.5 + 0.5 * density
    elif signals & MEDIUM:
        # Chemical formulas and exponents without an equation -- a table of
        # constants, a list of compounds. Real notation, less decisive.
        score = 0.15 + 0.35 * density
    else:
        # Symbols, Greek letters, inline slashes. Often incidental.
        score = 0.25 * density

    return NotationProfile(round(min(score, 1.0), 4), tuple(sorted(signals)))


def notation_score(text: str) -> float:
    return profile(text).score
