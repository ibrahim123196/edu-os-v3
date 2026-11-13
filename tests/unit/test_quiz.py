"""The quiz engine.

A model asked for a multiple-choice question produces something
question-shaped every time. Question-shaped is not correct, and a student
cannot tell a bad question from a hard one -- they conclude they do not
understand the topic. These are the structural failures a rule can catch.
"""

from __future__ import annotations

import pytest

from ai.quiz.generation import GenerationRefused, _extract_json, generate_question
from ai.quiz.validation import validate_multiple_choice

GOOD_EXPLANATION = (
    "Charge is measured in coulombs; one coulomb is the charge carried by a "
    "current of one ampere flowing for one second."
)


def test_a_sound_question_passes():
    result = validate_multiple_choice(
        "What is the SI unit of electric charge?",
        ["Coulomb", "Ampere", "Volt", "Farad"],
        0,
        GOOD_EXPLANATION,
    )
    assert result.ok, result.reason


def test_options_that_mean_the_same_thing_are_rejected():
    result = validate_multiple_choice(
        "What is the SI unit of electric charge?",
        ["Coulomb", "coulomb!", "Volt", "Farad"],
        0,
        GOOD_EXPLANATION,
    )
    assert not result.ok
    assert "same thing" in result.reason


def test_a_conspicuously_long_correct_answer_is_rejected():
    """It can be picked without knowing any physics."""
    result = validate_multiple_choice(
        "Which statement about Coulomb's law is true?",
        [
            "Yes",
            "No",
            "Maybe",
            "The force between two point charges varies inversely as the square "
            "of the distance separating them and directly as the product of the "
            "magnitudes of the charges",
        ],
        3,
        GOOD_EXPLANATION,
    )
    assert not result.ok
    assert "longer" in result.reason


def test_an_answer_hidden_in_the_stem_is_rejected():
    result = validate_multiple_choice(
        "Coulomb's law says the force varies inversely as the square of the "
        "distance, so which option is right?",
        [
            "varies inversely as the square of the distance",
            "varies linearly with distance",
            "does not depend on distance",
            "varies with the cube of distance",
        ],
        0,
        GOOD_EXPLANATION,
    )
    assert not result.ok
    assert "stem contains" in result.reason


def test_all_of_the_above_is_rejected():
    """Options are shuffled, so it cannot survive."""
    result = validate_multiple_choice(
        "Which are true of electric charge?",
        ["It is quantised", "It is conserved", "It is additive", "All of the above"],
        3,
        GOOD_EXPLANATION,
    )
    assert not result.ok
    assert "shuffling" in result.reason


def test_an_explanation_that_restates_the_question_is_rejected():
    stem = "What is the SI unit of electric charge?"
    result = validate_multiple_choice(stem, ["Coulomb", "Volt", "Farad"], 0, stem)
    assert not result.ok


def test_a_correct_index_outside_the_options_is_rejected():
    result = validate_multiple_choice(
        "What is the SI unit of charge?", ["Coulomb", "Volt", "Farad"], 7,
        GOOD_EXPLANATION,
    )
    assert not result.ok
    assert "not one of the options" in result.reason


def test_too_few_options_is_rejected():
    result = validate_multiple_choice(
        "What is the SI unit of charge?", ["Coulomb", "Volt"], 0, GOOD_EXPLANATION
    )
    assert not result.ok


# --- parsing the model's reply -------------------------------------------

VALID = '{"stem":"q","options":[{"text":"a"}],"correct_index":0,"explanation":"e"}'


@pytest.mark.parametrize(
    "raw",
    [
        VALID,
        f"```json\n{VALID}\n```",
        f"```\n{VALID}\n```",
        f"Here is the question:\n\n{VALID}\n\nHope that helps.",
    ],
)
def test_json_is_recovered_however_the_model_wraps_it(raw):
    """A model asked for JSON usually complies and occasionally wraps it in
    prose or a fence. Falling over on a stray backtick is not useful."""
    assert _extract_json(raw)["stem"] == "q"


def test_a_refusal_is_distinguished_from_a_failure():
    """A passage that cannot support a fair question is a valid outcome, not
    an error -- the alternative is a model inventing one."""

    class Refuses:
        model = "test"

        def complete(self, **kwargs):
            return '{"stem": null}'

    with pytest.raises(GenerationRefused):
        generate_question(Refuses(), "a short caption", "coulombs law")


def test_an_unparseable_reply_is_a_refusal_not_a_crash():
    class Rambles:
        model = "test"

        def complete(self, **kwargs):
            return "I would rather not write a question about this."

    with pytest.raises(GenerationRefused):
        generate_question(Rambles(), "some passage", "coulombs law")


def test_generation_validates_before_returning():
    class Produces:
        model = "test"

        def complete(self, **kwargs):
            return (
                '{"stem":"Which are true of charge?","options":'
                '[{"text":"Quantised"},{"text":"Conserved"},{"text":"Additive"},'
                '{"text":"All of the above"}],"correct_index":3,'
                '"explanation":"Charge is quantised, conserved and additive."}'
            )

    result = generate_question(Produces(), "passage text", "electric charge")
    assert not result.validation.ok
    assert "shuffling" in result.validation.reason
