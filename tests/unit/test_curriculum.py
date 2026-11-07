"""The curriculum graph's seed data.

These check the data, not the database: a wrong prerequisite edge sends
Milestone 4 chasing the wrong weakness, and a cycle would hang any traversal.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SEED = Path(__file__).resolve().parents[2] / "scripts" / "seed_curriculum.py"


def _constants():
    tree = ast.parse(SEED.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], "id", "")
            if name in {"EXAMS", "CONCEPTS", "CHAPTERS"}:
                out[name] = ast.literal_eval(node.value)
    return out


DATA = _constants()
CONCEPTS = DATA["CONCEPTS"]
CHAPTERS = DATA["CHAPTERS"]
SLUGS = {slug for slug, _, _ in CONCEPTS}


def test_concept_slugs_are_unique():
    assert len(SLUGS) == len(CONCEPTS)


def test_every_prerequisite_exists():
    missing = [
        (slug, prereq)
        for slug, _, prereqs in CONCEPTS
        for prereq in prereqs
        if prereq not in SLUGS
    ]
    assert missing == []


def test_every_topic_concept_exists():
    missing = [
        (number, topic, concept)
        for number, (_, topics) in CHAPTERS.items()
        for topic, _, concepts in topics
        for concept in concepts
        if concept not in SLUGS
    ]
    assert missing == []


def test_no_concept_is_its_own_prerequisite():
    assert [s for s, _, p in CONCEPTS if s in p] == []


def test_prerequisite_graph_is_acyclic():
    """A cycle would hang any traversal and make mastery propagation circular."""
    prereqs = {slug: list(p) for slug, _, p in CONCEPTS}
    state: dict[str, int] = {}

    def visit(node: str, stack: list[str]) -> None:
        if state.get(node) == 1:
            raise AssertionError("cycle: " + " -> ".join(stack + [node]))
        if state.get(node) == 2:
            return
        state[node] = 1
        for parent in prereqs[node]:
            visit(parent, stack + [node])
        state[node] = 2

    for slug in prereqs:
        visit(slug, [])


def test_the_known_chain_holds():
    """Gauss's law cannot be understood before electric field, which cannot
    be understood before Coulomb's law. Milestone 4 reads exactly this."""
    prereqs = {slug: list(p) for slug, _, p in CONCEPTS}

    def ancestors(slug: str) -> set[str]:
        out: set[str] = set()
        for parent in prereqs[slug]:
            out.add(parent)
            out |= ancestors(parent)
        return out

    chain = ancestors("gauss-applications")
    assert {"gauss-law", "electric-flux", "electric-field", "coulombs-law"} <= chain


def test_every_topic_tags_at_least_one_concept():
    untagged = [
        (number, topic)
        for number, (_, topics) in CHAPTERS.items()
        for topic, _, concepts in topics
        if not concepts
    ]
    assert untagged == []


@pytest.mark.parametrize("number", [1, 2, 3, 4])
def test_the_four_chapters_are_present(number):
    assert number in CHAPTERS
    name, topics = CHAPTERS[number]
    assert name and len(topics) >= 10
