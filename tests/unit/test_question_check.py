"""The independent check on generated questions.

A review of the bank found answer keys that were wrong, options marked
wrong that were right, false premises, misfiled and repeated questions.
Each test pins one of those to a rejection.
"""

from __future__ import annotations

import json

from ai.quiz.checking import check_question, judge

EXISTING = ["How does the force change when the distance doubles?"]


def reply(verdicts, *, false_premise=None, tests_concept=True, repeats=None, note=""):
    return json.dumps({
        "options": [{"correct": v, "why": f"because {i}"} for i, v in enumerate(verdicts)],
        "false_premise": false_premise,
        "tests_concept": tests_concept,
        "concept_note": note,
        "repeats": repeats,
    })


def test_one_right_option_matching_the_key_passes():
    assert judge(reply([False, True, False, False]), 1, 4, EXISTING).ok


def test_a_wrong_key_is_rejected():
    """'Distance doubles, potential quarters' was keyed while the checker
    would have picked 'halves'."""
    result = judge(reply([True, False, False, False]), 2, 4, EXISTING)
    assert not result.ok
    assert "key says option 3, checker says 1" in result.reason


def test_two_defensible_options_are_rejected():
    """The commonest fault: the right reason, reworded, marked wrong."""
    result = judge(reply([True, True, False, False]), 0, 4, EXISTING)
    assert not result.ok and "2 options are defensible (1, 2)" in result.reason


def test_no_right_option_is_rejected():
    assert not judge(reply([False] * 4), 0, 4, EXISTING).ok


def test_a_false_premise_is_rejected_even_with_a_matching_key():
    """'Why does the current split unequally at the cube's corner?' -- by
    symmetry it splits equally, whatever the options say."""
    result = judge(reply([True, False, False, False], false_premise="it splits equally"), 0, 4, EXISTING)
    assert not result.ok and "false premise" in result.reason


def test_a_question_on_another_concept_is_rejected():
    result = judge(reply([True, False, False, False], tests_concept=False,
                         note="this is about field lines"), 0, 4, EXISTING)
    assert not result.ok and "field lines" in result.reason


def test_a_repeat_of_an_existing_question_is_rejected():
    result = judge(reply([True, False, False, False], repeats=1), 0, 4, EXISTING)
    assert not result.ok and "repeats" in result.reason


def test_a_repeat_number_that_points_nowhere_is_ignored():
    assert judge(reply([True, False, False, False], repeats=7), 0, 4, EXISTING).ok


def test_an_unreadable_reply_is_a_rejection_not_a_pass():
    """A check that silently passes when it fails is how faulty questions
    got in. The reply is kept in the reason, so the cause can be seen."""
    result = judge("I think option 1 is right.", 0, 4, EXISTING)
    assert not result.ok
    assert "I think option 1 is right." in result.reason


def test_a_cut_off_reply_shows_how_it_ends():
    """A reply cut off mid-answer is recognisable by its last words."""
    cut = reply([True, False, False, False])[:-40]
    result = judge("x" * 300 + cut, 0, 4, EXISTING)
    assert not result.ok and cut[-30:].split()[-1] in result.reason


def test_a_reply_judging_too_few_options_is_rejected():
    assert not judge(reply([True, False, False]), 0, 4, EXISTING).ok


def test_a_reply_wrapped_in_a_code_fence_is_read():
    assert judge("```json\n" + reply([True, False, False, False]) + "\n```", 0, 4, EXISTING).ok


class _Recording:
    model = "checker"

    def __init__(self):
        self.prompt = ""

    def complete(self, system, messages, max_tokens):
        self.prompt = system + messages[0]["content"]
        return reply([True, False, False, False])


def test_the_checker_never_sees_the_key_or_the_explanation():
    """Shown the key, a checker tends to agree with it."""
    llm = _Recording()
    check_question(
        llm, concept="Coulomb's law", passage="Some passage.",
        stem="What happens?", options=["KEYED", "b", "c", "d"], correct_index=0,
        existing_stems=EXISTING,
    )
    # check_question is never given the explanation, so it cannot leak;
    # the keyed option must look exactly like the others.
    assert "1. KEYED\n2. b\n3. c\n4. d" in llm.prompt
    assert "correct_index" not in llm.prompt and "(correct)" not in llm.prompt
    assert EXISTING[0] in llm.prompt
