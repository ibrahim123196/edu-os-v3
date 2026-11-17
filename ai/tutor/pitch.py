"""Pitching an answer to the student's level on the topic.

The evidence is the quiz record: a mastery estimate per concept (with
whether it rests on enough answers), and the named mistakes the student
repeats. Only reliable evidence moves the pitch. With too little, the
answer is written exactly as it would be for anyone -- guessing a level
from nothing would talk down to a student who knows the topic, or past one
who does not.

Only how things are explained changes. The facts still come from the
sources alone, and the page says when and why an answer was pitched, so it
is never a hidden judgement about the student.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai.mastery.estimate import MasteryEstimate

# At or above this, reliably, the basics can be skipped. Well clear of the
# weak line: between the two the answer stays as it is.
STRONG_THRESHOLD = 0.8

SIMPLER = (
    "This student has been finding this topic hard ({record}). Explain it "
    "from the basics: plain words, define each term before you use it, build "
    "up one step at a time, and include one small worked example if the "
    "sources have one. State facts only from the sources, as always."
)
DEEPER = (
    "This student already does well on this topic ({record}). Be concise: "
    "skip the basics and go a step further with what the sources offer -- "
    "the conditions a result needs, special cases, and how it connects to "
    "related ideas. State facts only from the sources, as always."
)
REPEATED = (
    "In earlier quizzes -- not in this message -- this student has "
    "repeatedly chosen answers showing {mistakes}. If that is relevant to "
    "the question, address it directly. This is not a mistake in their "
    "message: do not label it as one."
)


@dataclass(frozen=True)
class Pitch:
    level: str | None          # "simpler", "deeper", or None (as for anyone)
    note: str | None           # what the page shows, saying why
    instruction: str | None    # what the model is told
    # Slugs of the repeated quiz mistakes the note already names, so the
    # same mistake is not shown a second time as a mix-up in the message.
    repeated: tuple[str, ...] = ()


AS_FOR_ANYONE = Pitch(None, None, None)


def choose(
    estimates: dict[str, MasteryEstimate],
    repeated: list[str] | None = None,
) -> Pitch:
    """The pitch for a topic, from the student's estimates on its concepts
    (concept name -> estimate) and the labels of mistakes they repeat there.

    Weak on any concept the answer draws on means simpler: the hard part is
    what will lose them. Deeper needs every concept with a record to be
    strong -- one shaky concept is reason enough not to skip the basics.
    """
    reliable = {name: e for name, e in estimates.items() if e.is_reliable}
    level, note, parts = None, None, []

    weak = sorted((e.mastery, name) for name, e in reliable.items() if e.is_weak)
    if weak:
        _, name = weak[0]
        record = _record(name, reliable[name])
        level, note = "simpler", f"Pitched simpler: your quiz record on {record}."
        parts.append(SIMPLER.format(record=record))
    elif reliable and all(e.mastery >= STRONG_THRESHOLD for e in reliable.values()):
        name = max(reliable, key=lambda n: reliable[n].attempts)
        record = _record(name, reliable[name])
        level, note = "deeper", f"Pitched for more depth: your quiz record on {record}."
        parts.append(DEEPER.format(record=record))

    if repeated:
        listed = "; ".join(f'"{m}"' for m in repeated)
        parts.append(REPEATED.format(mistakes=listed))
        mistake_note = f"Watching for a mistake you repeat: {repeated[0]}."
        note = f"{note} {mistake_note}" if note else mistake_note

    if not parts:
        return AS_FOR_ANYONE
    return Pitch(level, note, "\n\n".join(parts))


def _record(name: str, est: MasteryEstimate) -> str:
    return f"{name} is {round(est.mastery * 100)}% from {est.attempts} questions"

