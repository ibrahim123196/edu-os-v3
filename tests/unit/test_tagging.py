"""Linking passages to curriculum concepts.

Nothing downstream works without this: a quiz cannot select by concept, and
mastery cannot be tracked per concept, if no passage carries one.
"""

from __future__ import annotations

from ai.ingestion.tagging import (
    CONFIDENCE_INHERITED,
    CONFIDENCE_OWN_HEADING,
    resolve_topic,
    section_number,
    tag_chunks,
)

TOPICS = {"1.1", "1.5", "1.6", "3.5"}


def test_section_numbers_are_read_from_headings():
    assert section_number("1.5  COULOMB'S LAW") == "1.5"
    assert section_number("3.5.1 Mobility") == "3.5.1"
    assert section_number("Figure 1.3 shows") is None
    assert section_number(None) is None


def test_a_subsection_falls_back_to_its_parent():
    """NCERT numbers subsections the curriculum graph records only at
    section level. Attributing them to the parent is where a student would
    look for them anyway."""
    assert resolve_topic("3.5.1", TOPICS) == "3.5"
    assert resolve_topic("1.5", TOPICS) == "1.5"
    assert resolve_topic("9.9", TOPICS) is None


def test_a_heading_carries_forward_to_later_passages():
    """A section runs until the next one begins. Tagging only the passage
    containing the heading covered 23-46% of a chapter; carrying it forward
    covers 97%."""
    tags = tag_chunks(["1.5 COULOMB'S LAW", None, None, "1.6 FORCES"], TOPICS)
    assert [t.topic_number for t in tags] == ["1.5", "1.5", "1.5", "1.6"]


def test_the_passage_holding_the_heading_is_more_certain():
    tags = tag_chunks(["1.5 COULOMB'S LAW", None], TOPICS)
    assert tags[0].confidence == CONFIDENCE_OWN_HEADING
    assert tags[1].confidence == CONFIDENCE_INHERITED


def test_passages_before_the_first_heading_are_left_untagged():
    """Front matter belongs to no section, and guessing would be worse than
    leaving it out."""
    tags = tag_chunks([None, None, "1.1 INTRODUCTION"], TOPICS)
    assert [t.chunk_index for t in tags] == [2]


def test_an_unknown_section_number_does_not_displace_the_current_one():
    """A misread heading should not silently retag everything after it."""
    tags = tag_chunks(["1.5 COULOMB'S LAW", "9.9 NONSENSE", None], TOPICS)
    assert {t.topic_number for t in tags} == {"1.5"}


def test_no_headings_produces_no_tags():
    assert tag_chunks([None, None], TOPICS) == []
