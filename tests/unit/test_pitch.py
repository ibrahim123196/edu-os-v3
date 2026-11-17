"""When an answer is pitched simpler or deeper, and when it is left alone."""

from __future__ import annotations

from ai.mastery.estimate import MasteryEstimate
from ai.tutor.pitch import AS_FOR_ANYONE, choose


def est(mastery, attempts=10, confidence=0.8):
    return MasteryEstimate(mastery=mastery, confidence=confidence, attempts=attempts,
                           correct=round(mastery * attempts))


SHAKY = dict(confidence=0.2, attempts=2)       # too few answers to act on


def test_no_record_means_an_answer_as_for_anyone():
    assert choose({}) == AS_FOR_ANYONE


def test_too_little_evidence_changes_nothing():
    """Two answers is a guess, not a level."""
    assert choose({"Capacitance": est(0.1, **SHAKY)}) == AS_FOR_ANYONE
    assert choose({"Capacitance": est(1.0, **SHAKY)}) == AS_FOR_ANYONE


def test_a_weak_record_pitches_simpler_and_says_why():
    p = choose({"Capacitance": est(0.35, attempts=8)})
    assert p.level == "simpler"
    assert p.note == "Pitched simpler: your quiz record on Capacitance is 35% from 8 questions."
    assert "from the basics" in p.instruction and "only from the sources" in p.instruction


def test_a_strong_record_pitches_deeper():
    p = choose({"Capacitance": est(0.9, attempts=12)})
    assert p.level == "deeper"
    assert p.note == "Pitched for more depth: your quiz record on Capacitance is 90% from 12 questions."
    assert "skip the basics" in p.instruction and "only from the sources" in p.instruction


def test_in_between_is_left_alone():
    assert choose({"Capacitance": est(0.7)}) == AS_FOR_ANYONE


def test_one_weak_concept_is_enough_for_simpler():
    """The hard part is what will lose them."""
    p = choose({"Capacitance": est(0.95), "Energy stored": est(0.3, attempts=6)})
    assert p.level == "simpler" and "Energy stored is 30%" in p.note


def test_deeper_needs_every_concept_with_a_record_to_be_strong():
    assert choose({"Capacitance": est(0.95), "Energy stored": est(0.7)}) == AS_FOR_ANYONE


def test_a_shaky_record_does_not_block_deeper():
    """Only reliable estimates count, either way."""
    p = choose({"Capacitance": est(0.95), "Energy stored": est(0.1, **SHAKY)})
    assert p.level == "deeper"


def test_a_repeated_mistake_is_addressed_and_said():
    p = choose({}, repeated=["Thinks capacitance depends on the voltage applied"])
    assert p.level is None
    assert "repeatedly" in p.instruction and "depends on the voltage" in p.instruction
    assert p.note == "Watching for a mistake you repeat: Thinks capacitance depends on the voltage applied."


def test_both_together():
    p = choose({"Capacitance": est(0.3)}, repeated=["Thinks C depends on V"])
    assert p.level == "simpler"
    assert p.note.startswith("Pitched simpler:") and p.note.endswith("repeat: Thinks C depends on V.")
