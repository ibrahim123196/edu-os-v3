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


# --- choosing what to ask about ------------------------------------------

def test_passages_are_ranked_by_notation_not_length():
    """Ordering by length picked the longest chunk tagged with the concept,
    which for Coulomb's law is the biography of Coulomb and his torsion
    balance. The questions that came back asked why a sensitive instrument
    was needed in 1785 -- structurally valid, useless for revision."""
    import inspect

    from apps.api.services.quiz_service import candidate_passages

    source = inspect.getsource(candidate_passages)
    notation = source.index("notation_score.desc()")
    length = source.index("word_count.desc()")
    assert notation < length, "length is still the primary sort key"


def test_the_generator_creates_its_tables():
    """The CLI writes straight through session_scope, which does not run
    init_db. Generation succeeded and then threw away both questions on
    'relation "questions" does not exist'."""
    import inspect

    from scripts import generate_questions

    assert "init_db()" in inspect.getsource(generate_questions.main)


def test_a_stem_that_references_the_passage_is_rejected():
    """"According to the passage..." is a comprehension exercise, not a
    physics question, and it cannot be asked in an exam where the passage
    is absent."""
    result = validate_multiple_choice(
        "According to the passage, why is the constant written as 1/(4pi e0)?",
        ["For later convenience", "To enlarge e0", "To avoid vectors", "For CGS units"],
        0,
        "The form is chosen because it simplifies later electrostatics equations.",
    )
    assert not result.ok
    assert "refers to the passage" in result.reason


def test_existing_questions_are_sent_to_the_generator():
    """Asked twice about Coulomb's law in ignorance of the first answer, the
    model produced two near-identical questions about why the constant is
    written as 1/(4*pi*epsilon_0)."""
    captured = {}

    class Records:
        model = "test"

        def complete(self, system, messages, **kwargs):
            captured["prompt"] = messages[0]["content"]
            return (
                '{"stem":"What happens to the force when the separation doubles?",'
                '"options":[{"text":"It falls to one quarter"},{"text":"It halves"},'
                '{"text":"It doubles"},{"text":"It is unchanged"}],'
                '"correct_index":0,"explanation":"The force varies inversely as the '
                'square of the distance, so doubling r divides the force by four."}'
            )

    from ai.quiz.generation import generate_question

    generate_question(
        Records(),
        content="passage",
        concept="coulombs law",
        existing_stems=["Why is the constant written as 1/(4 pi e0)?"],
    )
    assert "already exist" in captured["prompt"]
    assert "1/(4 pi e0)" in captured["prompt"]


def test_options_differing_only_in_notation_are_not_duplicates():
    """Stripping everything outside a-z0-9 destroyed the content of physics
    options: "q1q2/r^2" and "q1q2/r" both collapsed to "q q r", and a sound
    question was rejected for having four identical answers."""
    result = validate_multiple_choice(
        "Which expression gives the magnitude of the Coulomb force?",
        ["q\u2081q\u2082/r\u00b2", "q\u2081q\u2082/r", "q\u2081/r\u00b2", "q\u2082/r\u00b2"],
        0,
        "The force is proportional to the product of the charges and inversely "
        "proportional to the square of the separation.",
    )
    assert result.ok, result.reason


def test_fractions_in_options_are_distinguished():
    result = validate_multiple_choice(
        "What happens to the force when the separation is doubled?",
        ["\u00bc F", "\u00bd F", "2 F", "4 F"],
        0,
        "The force varies inversely as the square of the distance, so doubling "
        "the separation divides the force by four.",
    )
    assert result.ok, result.reason


def test_genuine_duplicates_are_still_caught():
    result = validate_multiple_choice(
        "What is the SI unit of charge?",
        ["Coulomb", "coulomb!", "Volt"],
        0,
        "The coulomb is the SI unit of electric charge.",
    )
    assert not result.ok
    assert "same thing" in result.reason


def test_an_explanation_referencing_the_passage_is_rejected():
    """The explanation is read after answering, with the passage no longer
    on screen. "According to the passage..." points at something the student
    cannot see, and reads as though the physics were true only there."""
    result = validate_multiple_choice(
        "If the separation doubles, by what factor does the force change?",
        ["1/4", "1/2", "4", "2"],
        0,
        "According to the passage, the force varies inversely as the square of "
        "the distance between the charges.",
    )
    assert not result.ok
    assert "explanation refers to the passage" in result.reason


def test_options_differing_only_by_sign_are_not_duplicates():
    """"V = e - Ir" and "V = e + Ir" both normalised to "v e ir", and three
    sound questions were rejected for duplicate options when the sign was
    the entire answer."""
    result = validate_multiple_choice(
        "A cell of emf e and internal resistance r drives a current I. "
        "Which expression gives the terminal potential difference?",
        ["V = e - Ir", "V = e + Ir", "V = Ir - e", "V = e / Ir"],
        0,
        "Some of the emf is dropped across the internal resistance, so the "
        "terminal voltage is lower than the emf by Ir.",
    )
    assert result.ok, result.reason


def test_signed_numeric_options_are_distinguished():
    result = validate_multiple_choice(
        "The external work done bringing a charge from infinity to P is 50 J. "
        "What is its electrostatic potential energy at P?",
        ["+50 J", "-50 J", "0 J", "25 J"],
        0,
        "Work done against the field by an external agent is stored as "
        "potential energy, so the energy equals the work done.",
    )
    assert result.ok, result.reason


def test_the_length_rule_compares_against_the_longest_distractor():
    """Comparing against the average punished questions that had one short
    distractor among three reasonable ones."""
    result = validate_multiple_choice(
        "Which statement describes the balance condition of a Wheatstone bridge?",
        [
            "The ratio of resistances in one arm equals that in the other",
            "No",
            "The galvanometer reading is at its maximum value instead",
            "The bridge supply voltage must be doubled for balance",
        ],
        0,
        "At balance no current flows through the galvanometer, which requires "
        "the two resistance ratios to be equal.",
    )
    assert result.ok, result.reason
