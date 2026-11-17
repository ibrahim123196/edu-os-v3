"""Reading the model's mix-up label, and what the request looks like."""

from __future__ import annotations

from types import SimpleNamespace as Chunk

from ai.prompts.grounded_answer import USER_TEMPLATE, build_messages
from ai.tutor.mixups import MixUp, instruction, read_label

OFFERED = [MixUp("capacitance/depends-on-q-or-v", "Thinks capacitance depends on the charge or voltage"),
           MixUp("capacitance/net-charge-q", "Thinks a charged capacitor's net charge is Q")]


def test_the_label_is_read_and_removed():
    raw = "No: C stays the same [S1].\n\nIt depends on geometry [S1].\n[MIX-UP: capacitance/depends-on-q-or-v]"
    text, spotted = read_label(raw, OFFERED)
    assert spotted == OFFERED[0]
    assert text == "No: C stays the same [S1].\n\nIt depends on geometry [S1]."


def test_no_label_means_nothing_spotted():
    text, spotted = read_label("C = Q/V [S1].", OFFERED)
    assert spotted is None and text == "C = Q/V [S1]."


def test_a_label_that_was_not_offered_is_ignored_but_still_removed():
    """The model chooses from a list; an id from nowhere is a guess."""
    text, spotted = read_label("Answer [S1].\n[MIX-UP: gauss-law/depends-on-shape]", OFFERED)
    assert spotted is None and "MIX-UP" not in text


def test_every_label_line_is_removed_and_the_last_counts():
    raw = "[MIX-UP: capacitance/net-charge-q]\nAnswer [S1].\n[MIX-UP: capacitance/depends-on-q-or-v]"
    text, spotted = read_label(raw, OFFERED)
    assert text == "Answer [S1]." and spotted == OFFERED[0]


def test_tolerates_spacing():
    _, spotted = read_label("A [S1].\n  [MIX-UP:   capacitance/net-charge-q ]  ", OFFERED)
    assert spotted == OFFERED[1]


def test_the_instruction_lists_each_mistake_with_its_id():
    text = instruction(OFFERED)
    assert "- capacitance/depends-on-q-or-v: Thinks capacitance depends" in text
    assert "[MIX-UP: <id>]" in text and "do not write" in text


def _chunk():
    return Chunk(page_label="Page 23", document_title="NCERT", publisher="NCERT",
                 source_kind="platform", section=None, content="C = Q/V.")


def test_without_anything_to_offer_the_request_is_unchanged():
    """A question about a topic with no named mistakes: exactly as before."""
    plain = build_messages("what is C", [_chunk()])
    assert plain == build_messages("what is C", [_chunk()], extra=None)
    assert plain[0]["content"].endswith(USER_TEMPLATE.split("{question}")[1])


def test_the_mistakes_go_after_everything_else():
    msgs = build_messages("C doubles if V doubles, right?", [_chunk()], extra=instruction(OFFERED))
    content = msgs[0]["content"]
    assert content.index("C = Q/V.") < content.index("Common mistakes")
    assert content.rstrip().endswith("do not write that line.")


def test_a_divider_left_above_the_label_is_removed():
    """Seen: "---" showed under an answer once its label line was taken out."""
    text, _ = read_label(f"Answer [S1].\n\n---\n[MIX-UP: {OFFERED[0].slug}]", OFFERED)
    assert text == "Answer [S1]."


def test_a_divider_inside_an_answer_stays():
    text, _ = read_label("Part one [S1].\n\n---\n\nPart two [S1].", OFFERED)
    assert "---" in text
