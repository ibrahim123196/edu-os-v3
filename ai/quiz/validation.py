"""Decide whether a generated question is fit to show a student.

A model asked for a multiple-choice question will produce something
question-shaped every time. Question-shaped is not the same as correct, and
the failures are specific and recognisable: two options that mean the same
thing, a correct answer that is conspicuously longer than the rest, an
explanation that restates the stem, a stem that gives the answer away.

None of that is caught by reading the output quickly, and a student has no
way to tell a bad question from a hard one -- they just conclude they do not
understand the topic. So questions are checked before they are stored as
approved, and a failure is recorded rather than silently regenerated, so the
same mistake is visible instead of recurring.

These are cheap structural checks, not a judgement of whether the physics is
right. That still needs a human or a second model; what this catches is the
large class of questions that are wrong in ways a rule can see.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MIN_STEM_WORDS = 4
MAX_STEM_WORDS = 90
MIN_OPTIONS = 3
MAX_OPTIONS = 6
MIN_EXPLANATION_WORDS = 8

# How much longer the correct option may be than the longest wrong one
# before its length alone gives the answer away.
MAX_CORRECT_LENGTH_RATIO = 1.6


@dataclass
class ValidationResult:
    ok: bool
    problems: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return "; ".join(self.problems)


# Subscripts, superscripts and vulgar fractions carry the meaning in a
# physics option. Folding them to ASCII keeps "r^2" and "r2" comparable
# while keeping "r^2" and "r^3" apart.
_DIGIT_FORMS = str.maketrans(
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089",
    "01234567890123456789",
)
_FRACTIONS = {"\u00bc": "1/4", "\u00bd": "1/2", "\u00be": "3/4", "\u2153": "1/3"}


def _normalise(text: str) -> str:
    """Compare options by meaning, ignoring only decoration.

    Stripping everything outside a-z0-9 looked harmless and destroyed the
    content of physics options: "q1q2/r^2" and "q1q2/r" both collapsed to
    "q q r", and a perfectly good question was rejected for having four
    identical answers. Only punctuation and spacing are decoration here.
    """
    folded = text.lower().translate(_DIGIT_FORMS)
    for symbol, replacement in _FRACTIONS.items():
        folded = folded.replace(symbol, replacement)
    # Signs and relational operators are content, not decoration. Dropping
    # them made "V = e - Ir" and "V = e + Ir" identical, and "+50 J" and
    # "-50 J" identical -- three sound questions were rejected for having
    # duplicate options when the sign was the whole answer.
    kept = [c if (c.isalnum() or c in "/^_+-=<>") else " " for c in folded]
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def validate_multiple_choice(
    stem: str,
    options: list[str],
    correct_index: int,
    explanation: str,
) -> ValidationResult:
    problems: list[str] = []

    stem = (stem or "").strip()
    explanation = (explanation or "").strip()
    options = [o.strip() for o in options]

    # --- shape ----------------------------------------------------------
    words = len(stem.split())
    if words < MIN_STEM_WORDS:
        problems.append("stem too short to be a question")
    elif words > MAX_STEM_WORDS:
        problems.append("stem too long to read quickly")

    if not (MIN_OPTIONS <= len(options) <= MAX_OPTIONS):
        problems.append(f"{len(options)} options, expected {MIN_OPTIONS}-{MAX_OPTIONS}")

    if not 0 <= correct_index < len(options):
        problems.append("correct answer is not one of the options")
        return ValidationResult(False, problems)

    if any(not o for o in options):
        problems.append("an option is empty")

    if len(explanation.split()) < MIN_EXPLANATION_WORDS:
        problems.append("explanation too short to teach anything")

    # --- distinctness ---------------------------------------------------
    seen: dict[str, int] = {}
    for index, option in enumerate(options):
        key = _normalise(option)
        if key and key in seen:
            problems.append(
                f"options {seen[key] + 1} and {index + 1} say the same thing"
            )
        seen[key] = index

    # --- giveaways ------------------------------------------------------
    # A correct answer that dwarfs every distractor can be picked without
    # knowing any physics. Comparing against the average punished questions
    # with one short distractor among three reasonable ones, so the test is
    # against the longest wrong option: what matters is whether the correct
    # answer stands out, not whether some distractor was terse.
    correct = options[correct_index]
    wrong = [o for i, o in enumerate(options) if i != correct_index]
    if wrong:
        longest_wrong = max(len(o) for o in wrong)
        if longest_wrong > 0 and len(correct) > longest_wrong * MAX_CORRECT_LENGTH_RATIO:
            problems.append("correct option is conspicuously longer than the others")

    # The answer sitting in the stem makes the options decorative.
    normalised_stem = _normalise(stem)
    normalised_correct = _normalise(correct)
    if (
        normalised_correct
        and len(normalised_correct) > 12
        and normalised_correct in normalised_stem
    ):
        problems.append("the stem contains the correct answer")

    # A question that references its own source is a comprehension exercise,
    # not a physics question, and it cannot be asked in an exam where the
    # passage is absent.
    _references_source = re.compile(
        r"\b(according to|as stated in|as described in|in|from) the "
        r"(passage|text|extract|above)\b",
        re.I,
    )
    if _references_source.search(stem):
        problems.append("stem refers to the passage instead of asking physics")

    # The explanation is read after answering, when the passage is not on
    # screen. "According to the passage..." points at something the student
    # cannot see, and reads as though the physics were true only here.
    if _references_source.search(explanation):
        problems.append("explanation refers to the passage rather than the physics")

    # "All of the above" cannot be reordered, and these are shuffled.
    for option in options:
        if re.search(r"\b(all|none) of (the )?(above|these)\b", option, re.I):
            problems.append("'all/none of the above' does not survive shuffling")
            break

    # An explanation that only repeats the stem teaches nothing.
    if explanation and _normalise(explanation) == normalised_stem:
        problems.append("explanation only restates the question")

    return ValidationResult(not problems, problems)