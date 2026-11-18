"""Cutting the model's lesson into steps, and what it is asked for."""

from __future__ import annotations

from ai.tutor.lesson import MAX_CHECK_WORDS, as_text, instruction, split
from ai.tutor.mixups import MixUp

OFFERED = [MixUp("cap/depends-on-v", "Thinks capacitance depends on the voltage")]

FULL = """[INTUITION]
A capacitor stores charge on two plates [S1].
[EXPLANATION]
Capacitance is C = Q/V [S1].
[VISUAL]
Look at Figure 2.24 on page 23: two conductors and the insulator between them [S2].
[EXAMPLE]
With A = 1 m2 and d = 1 mm, C = 8.85 x 10^-9 F [S3].
[MISCONCEPTION: cap/depends-on-v]
Many think C changes with V. It does not: it is fixed by the geometry [S1].
[CHECK]
In your own words, what decides how much capacitance a capacitor has?"""


def test_a_full_lesson_becomes_its_steps_in_order():
    lesson = split(FULL, OFFERED)
    assert [k for k, _, _ in lesson.steps] == [
        "intuition", "explanation", "visual", "example", "misconception"]
    assert lesson.steps[2][2].startswith("Look at Figure 2.24 on page 23")
    assert lesson.misconception == OFFERED[0]
    assert lesson.check == "In your own words, what decides how much capacitance a capacitor has?"


def test_a_misconception_not_on_the_list_is_left_out():
    """Only reviewed mistakes are taught as 'common'."""
    text = FULL.replace("[MISCONCEPTION: cap/depends-on-v]", "[MISCONCEPTION: made/up]")
    lesson = split(text, OFFERED)
    assert "misconception" not in [k for k, _, _ in lesson.steps]
    assert lesson.misconception is None


def test_none_leaves_the_misconception_step_out():
    text = FULL.split("[MISCONCEPTION")[0] + "[MISCONCEPTION: none]\n[CHECK]\nExplain C?"
    lesson = split(text, OFFERED)
    assert lesson.misconception is None and lesson.check == "Explain C?"
    assert len(lesson.steps) == 4


def test_a_section_it_did_not_write_is_simply_absent():
    text = FULL.replace("[VISUAL]\nLook at Figure 2.24 on page 23: two conductors and "
                        "the insulator between them [S2].\n", "")
    assert "visual" not in [k for k, _, _ in split(text, OFFERED).steps]


def test_a_check_that_is_not_one_short_question_is_dropped():
    long = " ".join(["word"] * (MAX_CHECK_WORDS + 5))
    assert split(FULL.rsplit("[CHECK]", 1)[0] + f"[CHECK]\n{long}", OFFERED).check is None


def test_text_without_tags_is_no_lesson():
    """The caller then shows it as an ordinary answer."""
    assert split("Capacitance is C = Q/V [S1].", OFFERED).steps == []


def test_the_lesson_as_one_text_for_the_conversation():
    text = as_text(split(FULL, OFFERED))
    assert text.startswith("The idea:\nA capacitor stores charge")
    assert text.endswith("Check your understanding: In your own words, what decides "
                         "how much capacitance a capacitor has?")


def test_what_the_model_is_asked_for():
    asked = instruction("capacitance", OFFERED)
    for tag in ("[INTUITION]", "[EXPLANATION]", "[VISUAL]", "[EXAMPLE]", "[CHECK]"):
        assert tag in asked
    assert "- cap/depends-on-v: Thinks capacitance depends" in asked
    assert "[S#]" in asked
    assert "mention no figure" in asked                    # none listed: a drawing


def test_the_figures_in_the_sources_are_listed_for_the_visual_step():
    asked = instruction("electric field", [], ["1.12", "1.13"])
    assert "these textbook figures: Figure 1.12, Figure 1.13" in asked
    assert "do not tell them to find it" in asked


def test_without_reviewed_mistakes_there_is_no_misconception_section():
    assert "[MISCONCEPTION" not in instruction("capacitance", [])


def test_the_worked_example_is_worked_not_described():
    """18 Nov: step 4 said "the sources provide the setup for Example 1.3 ...
    you would calculate" -- no working, and a reference the student cannot
    use."""
    asked = instruction("Coulomb's law", OFFERED)
    assert "Work one example all the way through" in asked
    assert "solve it here from the formula and the values in the sources" in asked
    assert "never send the student to look it up" in asked


def test_the_lesson_talks_to_the_student_not_about_the_sources():
    assert 'never write "the sources"' in instruction("Coulomb's law", OFFERED)
