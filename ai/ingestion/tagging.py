"""Link passages to curriculum concepts.

Nothing downstream works without this. A quiz cannot select questions by
concept if no passage is tagged with one; mastery cannot be tracked per
concept; "your weak area is electric field" has nothing to stand on.

The textbook does the work for us. Every section carries a number -- "1.5
Coulomb's Law" -- and the curriculum graph stores topics under the same
numbers, so matching one to the other needs no model and no guesswork.

The part that is easy to get wrong is that a heading applies to everything
after it, not just the passage it appears in. Tagging only the chunk that
contains the heading covered 23-46% of a chapter. Carrying the number
forward until the next heading covers 97%, which is what a reader would
assume: section 1.5 runs until 1.6 begins.

Confidence reflects that difference. A passage holding its own heading is
certain; one inherited from a heading further up is likely but not proven,
and a later stage may want to treat the two differently.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# "1.5", "3.14", "2.11.1"
SECTION_NUMBER = re.compile(r"^(\d+\.\d+(?:\.\d+)?)\b")

CONFIDENCE_OWN_HEADING = 1.0
CONFIDENCE_INHERITED = 0.8


@dataclass(frozen=True)
class ChunkTag:
    chunk_index: int
    topic_number: str
    confidence: float


def section_number(section: str | None) -> str | None:
    if not section:
        return None
    match = SECTION_NUMBER.match(section.strip())
    return match.group(1) if match else None


def resolve_topic(number: str, known: set[str]) -> str | None:
    """Match a heading number to a topic, falling back to its parent.

    NCERT numbers subsections ("3.5.1 Mobility") that the curriculum graph
    records only at section level. Rather than drop them, a subsection is
    attributed to its parent, which is where a student would look for it.
    """
    if number in known:
        return number
    parent = ".".join(number.split(".")[:2])
    return parent if parent in known else None


def tag_chunks(sections: list[str | None], known_topics: set[str]) -> list[ChunkTag]:
    """Assign a topic number to each chunk, in document order.

    `sections` is each chunk's detected heading, in the order the chunks
    appear. Order matters: that is what lets a heading carry forward.
    """
    tags: list[ChunkTag] = []
    current: str | None = None

    for index, section in enumerate(sections):
        number = section_number(section)
        own_heading = False

        if number:
            resolved = resolve_topic(number, known_topics)
            if resolved:
                current = resolved
                own_heading = True

        if current:
            tags.append(
                ChunkTag(
                    chunk_index=index,
                    topic_number=current,
                    confidence=(
                        CONFIDENCE_OWN_HEADING if own_heading else CONFIDENCE_INHERITED
                    ),
                )
            )

    return tags
