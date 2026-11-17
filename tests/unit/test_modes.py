"""Which learning mode a message asks for, and what it is about."""

from __future__ import annotations

import pytest

from ai.tutor.modes import Mode, detect, points_back

CASES = [
    # practice
    ("quiz me on gauss's law", Mode.practice, "gauss's law"),
    ("Can you give me some practice questions on capacitors please?", Mode.practice, "capacitors"),
    ("5 questions on drift velocity", Mode.practice, "drift velocity"),
    ("Give me 10 MCQs on Coulomb’s law", Mode.practice, "coulomb's law"),
    ("practise kirchhoff's rules", Mode.practice, "kirchhoff's rules"),
    ("capacitance, quiz me", Mode.practice, "capacitance"),
    ("give me some practice questions", Mode.practice, ""),
    # test: before practice, so "test me with questions" is a test
    ("test me", Mode.test, ""),
    ("give me a mock test", Mode.test, ""),
    ("can you test me on the whole chapter", Mode.test, ""),
    # a test is the whole syllabus: naming a topic makes it practice
    ("test me on electric flux", Mode.practice, "electric flux"),
    # review
    ("revise capacitors in series", Mode.review, "capacitors in series"),
    ("let's revise the electric field of a dipole", Mode.review, "electric field of a dipole"),
    ("summary of ohm's law", Mode.review, "ohm's law"),
    # learn
    ("teach me electric potential", Mode.learn, "electric potential"),
    ("I want to learn about dielectrics", Mode.learn, "dielectrics"),
    # clarify: plain questions, including ones that mention a mode's word
    ("why is E zero inside a conductor", Mode.clarify, "why is E zero inside a conductor"),
    ("what is a mixed quiz", Mode.clarify, "what is a mixed quiz"),
    ("what does the review of chapter 1 say", Mode.clarify, "what does the review of chapter 1 say"),
    ("how do I revise for boards", Mode.clarify, "how do I revise for boards"),
    ("explain coulomb's law", Mode.clarify, "explain coulomb's law"),
]


@pytest.mark.parametrize("message,mode,topic", CASES)
def test_mode_and_topic(message, mode, topic):
    guess = detect(message)
    assert (guess.mode, guess.topic) == (mode, topic)


def test_the_topic_keeps_its_own_small_words():
    """Only request words at the ends go: 'in' inside the topic stays."""
    assert detect("quiz me on capacitors in series").topic == "capacitors in series"


def test_this_that_and_it_point_back_at_the_conversation():
    assert points_back(detect("quiz me on this").topic)
    assert points_back(detect("quiz me on it").topic)
    assert not points_back(detect("test me").topic)     # nothing named: all topics
    assert points_back(detect("revise that").topic)
    assert not points_back(detect("quiz me on gauss's law").topic)


def test_the_cue_is_recorded():
    assert detect("quiz me on gauss's law").cue == "quiz me"
    assert detect("why is that").cue is None
