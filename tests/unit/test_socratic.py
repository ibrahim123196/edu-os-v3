"""Work it out: the plan, the judging, and moving through the steps."""

from __future__ import annotations

from ai.tutor import socratic
from ai.tutor.socratic import advance, opening, read_verdict, split, start_state

PLAN = """[GOAL]
The field inside is zero because the free electrons move until their own field cancels it [S1].
[STEP]
Q: What do free electrons do in an electric field?
POINT: They move, opposite to the field [S1].
HINT: Think about the force on a charge in a field.
[STEP]
Q: Where do they end up?
POINT: On the surface [S1].
HINT: They cannot leave the conductor.
[STEP]
Q: Does their field help or oppose the original one?
POINT: It opposes it, until the two cancel inside [S2].
HINT: Which side is left positive?"""


def test_a_plan_becomes_its_goal_and_steps():
    plan = split(PLAN)
    assert plan.goal.startswith("The field inside is zero")
    assert [s.question for s in plan.steps][0] == "What do free electrons do in an electric field?"
    assert plan.steps[2].point == "It opposes it, until the two cancel inside [S2]."
    assert plan.steps[1].hint == "They cannot leave the conductor."


def test_what_is_not_a_plan_is_refused():
    """Never lead a student along half a plan."""
    assert split("The field inside a conductor is zero [S1].") is None           # an answer
    assert split(PLAN.split("[STEP]")[0]) is None                               # no steps
    assert split(PLAN.replace("[GOAL]\n", "").split("\n", 1)[1]) is None        # no goal
    one_step = PLAN.split("[STEP]")
    assert split("[STEP]".join(one_step[:2])) is None                           # too few
    assert split(PLAN.replace("POINT: On the surface [S1].\n", "")) is None     # no point
    many = PLAN + ("\n[STEP]\nQ: q?\nPOINT: p [S1].\nHINT: h" * 4)
    assert split(many) is None                                                   # too many


def test_the_judges_verdict_is_read_carefully():
    assert read_verdict("VERDICT: right\nFEEDBACK: Yes, they move [S1].") == ("right", "Yes, they move.")
    assert read_verdict("verdict: Partly\nfeedback: Close.\nextra line") == ("partly", "Close.")
    # Unreadable: not passed.
    assert read_verdict("I think so") == ("wrong", "")
    assert read_verdict("") == ("wrong", "")


def _state():
    return start_state(split(PLAN))


def test_it_opens_with_the_first_question():
    assert opening(_state()).endswith("What do free electrons do in an electric field?")


def test_right_moves_to_the_next_question():
    state, text, event = advance(_state(), "right", "Yes.")
    assert event == "right" and state["step"] == 1 and state["tries"] == 0
    assert text == "Yes.\n\nWhere do they end up?"


def test_wrong_asks_again_then_hints_then_shows_the_step():
    state = _state()
    state, text, event = advance(state, "wrong", "Not quite.")
    assert event == "again" and state["step"] == 0
    assert text.endswith("Try again: What do free electrons do in an electric field?")
    assert "Hint" not in text

    state, text, event = advance(state, "partly", "Closer.")
    assert event == "hint" and state["step"] == 0
    assert "Hint: Think about the force on a charge in a field." in text

    state, text, event = advance(state, "wrong", "")
    assert event == "shown" and state["step"] == 1 and state["tries"] == 0
    assert "Here's this step: They move, opposite to the field [S1]." in text
    assert text.endswith("Where do they end up?")                    # never stuck


def test_the_last_step_ends_with_the_answer():
    state = _state()
    for _ in range(3):
        state, text, _ = advance(state, "right", "Good.")
    assert state["done"]
    assert text.endswith("You've worked it out. The field inside is zero because the free "
                         "electrons move until their own field cancels it [S1].")


def test_the_state_is_not_changed_in_place():
    """It is saved with each message; the old turn must keep its own."""
    state = _state()
    advance(state, "wrong", "")
    assert state["tries"] == 0


def test_what_the_model_is_asked_for():
    asked = socratic.instruction("why is the field inside a conductor zero?")
    for tag in ("[GOAL]", "[STEP]", "Q:", "POINT:", "HINT:"):
        assert tag in asked
    assert "Do not answer it" in asked
