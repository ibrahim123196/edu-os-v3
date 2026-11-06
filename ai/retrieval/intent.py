"""What kind of answer is the question asking for?

"Formula for Coulomb force" and "explain Coulomb force" want different
passages from the same chapter. The first wants the line with the equation on
it; the second wants the paragraph underneath. Retrieval cannot tell them
apart from the shared words alone -- "coulomb" and "force" appear in both.

Two things follow from spotting the difference.

First, words like "formula" and "equation" describe the *shape* of the wanted
answer, not its subject. Searching for them is actively harmful: they match
prose that discusses formulas rather than passages that contain one. So they
are stripped from the search text and kept only as a signal.

Second, when the signal is present, passages carrying real notation get pulled
into the candidate pool and promoted during ranking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Asking for the mathematical or chemical form of something.
_NOTATION_WORDS = (
    r"formula|formulae|formulas|equation|equations|expression|expressions|"
    r"derive|derivation|derivations|reaction|reactions|balanced|"
    r"mathematical(?:ly)?|express(?:ed)?|relation|relationship|"
    r"symbolically|in terms of|general form|standard form"
)
_NOTATION_REQUEST = re.compile(rf"\b(?:{_NOTATION_WORDS})\b", re.IGNORECASE)

# Asking to compute something. These also want the formula, because that is
# what the student needs in order to do the calculation.
_CALCULATION = re.compile(
    r"\b(?:calculate|compute|find the value|how much|what is the value|"
    r"solve for|numerical|evaluate)\b",
    re.IGNORECASE,
)

# Removed from the search text once detected, since they describe the answer's
# form rather than its subject.
_STRIP = re.compile(
    rf"\b(?:{_NOTATION_WORDS}|write|state|give|what is the|the)\s+(?=\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryIntent:
    wants_notation: bool
    signals: tuple[str, ...]

    @property
    def describes(self) -> str:
        return ", ".join(self.signals) if self.signals else "general explanation"


def detect(question: str) -> QueryIntent:
    signals: list[str] = []

    notation_hits = {m.group(0).lower() for m in _NOTATION_REQUEST.finditer(question)}
    if notation_hits:
        signals.extend(sorted(notation_hits))

    calculation_hits = {m.group(0).lower() for m in _CALCULATION.finditer(question)}
    if calculation_hits:
        signals.extend(sorted(calculation_hits))

    return QueryIntent(
        wants_notation=bool(notation_hits or calculation_hits),
        signals=tuple(signals),
    )


def strip_form_words(question: str) -> str:
    """Remove 'formula for', 'state the', etc, leaving the actual subject.

    Falls back to the original if stripping would leave nothing -- "give me
    the formula" has no subject of its own and still needs to search for
    something.
    """
    stripped = _STRIP.sub(" ", question)
    stripped = re.sub(r"\s+", " ", stripped).strip(" ?.,")
    return stripped if len(stripped.split()) >= 1 and stripped else question
