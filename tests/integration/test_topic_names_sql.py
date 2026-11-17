"""What real topic names in a message are taken to mean.

Against the real course map, seeded into the local database (seeding is
safe to repeat). Searching the textbook is not exercised here: every
phrase below must be understood by name alone. Local database only (see
conftest).
"""

from __future__ import annotations

import pytest

from ai.tutor.modes import detect
from apps.api.db import init_db, session_scope
from apps.api.services.tutor_service import find_topic

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def course():
    from scripts.seed_curriculum import seed

    init_db()
    seed()


@pytest.fixture(autouse=True)
def names_only(monkeypatch):
    """The search fallback off: a phrase that only works by searching
    fails here instead of passing for the wrong reason."""
    from apps.api.services import tutor_service

    monkeypatch.setattr(tutor_service, "_concept_by_search", lambda db, topic: None)


CASES = [
    # the subject means the whole group
    ("quiz me on capacitance", "Capacitors", 4),
    ("quiz me on capacitors", "Capacitors", 4),
    ("quiz me on gauss law", "Gauss's law", 3),
    ("revise electrostatic potential", "Electrostatic potential", 4),
    ("quiz me on the electric dipole", "Dipoles", 2),
    ("quiz me on current electricity", "Current electricity", 10),
    # one part means that part
    ("quiz me on parallel plate capacitor", "Parallel plate capacitor", 1),
    ("quiz me on capacitors in series and parallel", "Capacitors in series and parallel", 1),
    ("quiz me on applications of gauss's law", "Applications of Gauss's law", 1),
    ("quiz me on electric field lines", "Electric field lines", 1),
    # agreed: dielectrics only when named, and then on its own
    ("quiz me on dielectrics", "Dielectrics and polarisation", 1),
    # everyday names for single topics
    ("quiz me on drift velocity", "Drift velocity and mobility", 1),
    ("quiz me on emf", "EMF and internal resistance", 1),
    ("quiz me on galvanometer", "Moving coil galvanometer", 1),
]


@pytest.mark.parametrize("message,name,size", CASES)
def test_topic_names(message, name, size):
    with session_scope() as db:
        match = find_topic(db, detect(message).topic)
    assert (match.name, len(match.concepts), match.how) == (name, size, "name")


def test_capacitors_leaves_dielectrics_out():
    with session_scope() as db:
        match = find_topic(db, "capacitance")
    assert "dielectrics-polarisation" not in match.slugs
