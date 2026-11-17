"""The topic groups as agreed, checked against the course map."""

from __future__ import annotations

import re
from pathlib import Path

from apps.api.topic_groups import BY_SLUG, CONCEPT_ALIASES, GROUPS

SEED = Path(__file__).resolve().parents[2] / "scripts" / "seed_curriculum.py"
CONCEPTS = re.findall(r'\("([a-z0-9-]+)", "[^"]+", \[', SEED.read_text(encoding="utf-8"))


def test_every_member_is_a_real_concept():
    """A typo in a slug would silently drop that topic from its group."""
    for g in GROUPS:
        assert set(g.members) <= set(CONCEPTS), (g.slug, set(g.members) - set(CONCEPTS))


def test_the_chapters_cover_every_concept_once():
    in_chapters = [m for g in GROUPS if g.chapter for m in g.members]
    assert sorted(in_chapters) == sorted(CONCEPTS)


def test_dielectrics_only_when_named():
    """Agreed: not part of Capacitors, though the textbook links them."""
    assert "dielectrics-polarisation" not in BY_SLUG["capacitors"].members
    assert not any("dielectrics-polarisation" in g.members for g in GROUPS if not g.chapter)


def test_flux_is_with_gauss_only():
    assert [g.slug for g in GROUPS if not g.chapter and "electric-flux" in g.members] == ["gauss"]


def test_energy_and_power_is_with_circuits():
    assert "electrical-energy-power" in BY_SLUG["circuits"].members


def test_capacitors_is_the_four_capacitor_topics():
    assert BY_SLUG["capacitors"].members == (
        "capacitance", "parallel-plate-capacitor", "capacitor-combinations", "energy-in-capacitor")


def test_slugs_are_unique():
    assert len(BY_SLUG) == len(GROUPS)


def test_everyday_names_point_at_real_concepts():
    assert set(CONCEPT_ALIASES) <= set(CONCEPTS)
