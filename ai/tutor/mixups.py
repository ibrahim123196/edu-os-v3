"""Spotting a known mistake in what the student types.

"If I double the voltage, the capacitance doubles, right?" is a question,
but it also shows a wrong idea -- one the quiz already has a name for. An
answer that explains capacitance correctly without saying "no, that is the
mistake" leaves the idea in place.

The named misconceptions (content/misconceptions.json, about three per
concept) are offered to the model alongside the sources, limited to the
concepts the retrieved passages are tagged with. The model corrects one if
the student's message states or assumes it, and labels it on a line of its
own, which is read and removed here.

Done inside the answer request, not as a separate check: no extra cost or
delay. A separate, single-purpose check would likely be more accurate;
every detection is recorded in the message's trace so accuracy can be
judged before deciding whether to switch. Detections do not count towards
mastery: a chat sentence is weaker evidence than a wrong quiz answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

# More than this and the list starts to crowd the sources out of the
# model's attention; the best passages' topics come first.
MAX_OFFERED = 12

_LABEL = re.compile(r"^[ \t]*\[MIX-UP:[ \t]*([A-Za-z0-9/_-]+)[ \t]*\][ \t]*$", re.MULTILINE)
# A divider the model sometimes puts just above its label line; left behind
# once the label is removed, it showed as a stray "---" under the answer.
_TRAILING_RULE = re.compile(r"(\n[ \t]*(-{3,}|\*{3,}|_{3,})[ \t]*)+\s*$")


@dataclass(frozen=True)
class MixUp:
    slug: str
    label: str


def offered_for(db: Session, chunk_ids: list) -> list[MixUp]:
    """The named mistakes for the concepts these passages are tagged with,
    best passage first."""
    from apps.api.curriculum import chunk_concepts
    from apps.api.quiz import Misconception

    if not chunk_ids:
        return []
    rank = {cid: i for i, cid in enumerate(chunk_ids)}
    rows = db.execute(
        select(chunk_concepts.c.chunk_id, Misconception.slug, Misconception.label)
        .join(Misconception, Misconception.concept_id == chunk_concepts.c.concept_id)
        .where(chunk_concepts.c.chunk_id.in_(list(rank)))
    ).all()
    seen: dict[str, MixUp] = {}
    for _, slug, label in sorted(rows, key=lambda r: (rank[r.chunk_id], r.slug)):
        seen.setdefault(slug, MixUp(slug, label))
    return list(seen.values())[:MAX_OFFERED]


def instruction(offered: list[MixUp]) -> str:
    """Added to the request when there is anything to offer."""
    lines = "\n".join(f"- {m.slug}: {m.label}" for m in offered)
    return (
        "Common mistakes students make on these topics:\n"
        f"{lines}\n\n"
        "If the student's message itself states or assumes one of these "
        "(for example \"it doubles, right?\"), start your answer by gently "
        "correcting it from the sources, with [S#] markers, then answer. "
        "Then, as the very last line, write [MIX-UP: <id>] with that "
        "mistake's id. If the message states none of them -- including when "
        "it only asks a neutral question about the topic -- do not write "
        "that line."
    )


def read_label(raw: str, offered: list[MixUp]) -> tuple[str, MixUp | None]:
    """The answer without its label line, and the mistake it named.

    Every label line is removed, so none reaches the student. A label naming
    a mistake that was not offered is ignored: the model is choosing from a
    list, and an id from nowhere is a guess.
    """
    found = _LABEL.findall(raw)
    text = _TRAILING_RULE.sub("", _LABEL.sub("", raw).strip()).strip()
    if not found:
        return text, None
    by_slug = {m.slug: m for m in offered}
    return text, by_slug.get(found[-1])
