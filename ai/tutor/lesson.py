"""A lesson for "teach me ...": the roadmap's teaching sequence.

    intuition -> simple explanation -> visual -> example
    -> common misconception -> mini-question -> check understanding

The first five are written by the model in one request, each section
behind a tag, all from the same sources and under the same rules as any
answer: facts only from the sources, with [S#] markers. The mini-question
is not written at all -- it is a reviewed question from the bank, chosen
elsewhere -- and the check is a question the student answers in their own
words, given feedback afterwards against the same sources.

The text is checked for citations as one piece, then split into steps,
so every step's [1], [2] refer to one list of sources.

The misconception step names one mistake from the reviewed list, or is left
out: a lesson inventing its own "common mistake" would teach the student a
claim nobody has checked.

Pointing to the textbook's own figure, rather than drawing one, is how the
visual step stays accurate: a diagram drawn in characters gets details
wrong, and the real figure is on the page cited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ai.tutor.mixups import MixUp

STEPS = (
    ("intuition", "The idea"),
    ("explanation", "Simple explanation"),
    ("visual", "Picture it"),
    ("example", "Worked example"),
    ("misconception", "Common misconception"),
)

_SECTION = re.compile(r"^[ \t]*\[(INTUITION|EXPLANATION|VISUAL|EXAMPLE|MISCONCEPTION|CHECK)"
                      r"(?::[ \t]*([A-Za-z0-9/_-]+))?\][ \t]*$", re.MULTILINE)

MAX_CHECK_WORDS = 40


def instruction(topic: str, offered: list[MixUp], figures: list[str] | None = None) -> str:
    """What the model is asked for, after the sources. `figures` are the
    figure numbers the sources mention, listed so the visual step uses one
    rather than describing a drawing while the real one is on the page."""
    if offered:
        listed = "\n".join(f"- {m.slug}: {m.label}" for m in offered)
        misconception = (
            "[MISCONCEPTION: <id>]\n"
            "The most important of these common mistakes for this topic, in "
            "one sentence, then why it is wrong, from the sources:\n"
            f"{listed}\n"
            "Use the id of the one you chose in the tag. If none of them fits "
            "this topic, write [MISCONCEPTION: none] and nothing under it.\n\n"
        )
    else:
        misconception = ""
    return (
        f"Teach the student this topic as a short lesson: {topic}\n\n"
        "Write these sections in this order. Start each one with its tag on "
        "a line of its own. Use only the sources, with [S#] markers, as "
        "always; a section the sources cannot support should say so in one "
        "line rather than fill in.\n\n"
        "[INTUITION]\n"
        "The core idea in two or three plain sentences, before any formula.\n\n"
        "[EXPLANATION]\n"
        "The simple explanation: define each term before using it, build up "
        "one step at a time, give the key formula if there is one.\n\n"
        f"{_visual(figures or [])}"
        "[EXAMPLE]\n"
        "One worked example from the sources, step by step. If the sources "
        "have none, a short concrete case using only what they say.\n\n"
        f"{misconception}"
        "[CHECK]\n"
        "One question asking the student to explain the core idea in their "
        "own words, in a sentence or two. The question only, no answer."
    )


def _visual(figures: list[str]) -> str:
    if not figures:
        return (
            "[VISUAL]\n"
            "The sources mention no figure: describe in words a simple picture "
            "the student could draw.\n\n"
        )
    listed = ", ".join(f"Figure {f}" for f in figures)
    return (
        "[VISUAL]\n"
        f"The sources mention these textbook figures: {listed}. Choose the one "
        "that best shows this topic, name it once as \"Figure N.N\", and say "
        "what to notice in it, part by part, from what the sources say about "
        "it. The figure is shown to the student right above your words, so "
        "write as if they are looking at it: do not tell them to find it, open "
        "the book or draw it. Only if none of them fits this topic, describe a "
        "simple picture the student could draw instead.\n\n"
    )


@dataclass
class Lesson:
    steps: list[tuple[str, str, str]] = field(default_factory=list)   # key, title, text
    check: str | None = None
    misconception: MixUp | None = None


def split(text: str, offered: list[MixUp]) -> Lesson:
    """The checked lesson text, cut into its steps."""
    found = list(_SECTION.finditer(text))
    sections: dict[str, tuple[str | None, str]] = {}
    for i, m in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(text)
        body = text[m.end():end].strip()
        sections.setdefault(m.group(1).lower(), (m.group(2), body))

    lesson = Lesson()
    by_slug = {m.slug: m for m in offered}
    for key, title in STEPS:
        if key not in sections:
            continue
        tag_value, body = sections[key]
        if key == "misconception":
            # Only a mistake from the reviewed list, and only with something
            # said about it.
            chosen = by_slug.get(tag_value or "")
            if chosen is None or not body:
                continue
            lesson.misconception = chosen
        if body:
            lesson.steps.append((key, title, body))

    check = sections.get("check", (None, ""))[1]
    check = check.splitlines()[0].strip() if check else ""
    if check and len(check.split()) <= MAX_CHECK_WORDS:
        lesson.check = check
    return lesson


def as_text(lesson: Lesson) -> str:
    """The lesson as one text, for the conversation history."""
    parts = [f"{title}:\n{text}" for _, title, text in lesson.steps]
    if lesson.check:
        parts.append(f"Check your understanding: {lesson.check}")
    return "\n\n".join(parts)


FEEDBACK = (
    "The student was asked: {check}\n"
    "They answered: {reply}\n\n"
    "Using only the sources above, give feedback in two to four sentences: "
    "what they got right, and what is missing or wrong, with [S#] markers. "
    "Be encouraging and specific. If their answer is off the topic or empty, "
    "say so kindly and give the one key point they should remember."
)
