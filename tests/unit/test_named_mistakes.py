"""Named mistakes: the list file, and what the generator keeps from a reply.

Every question used to describe its wrong options in its own words, so the
same mistake on two questions never counted as a repeat. These pin the
shared list and the guard against ids the model makes up.
"""

from __future__ import annotations

import json
from pathlib import Path

from ai.quiz.generation import _mistake_ids, generate_question
from scripts.seed_curriculum import CONCEPTS

LIST = json.loads(
    (Path(__file__).parents[2] / "content" / "misconceptions.json").read_text(encoding="utf-8")
)
OFFERED = [("coulombs-law/inverse-r", "Uses 1/r"), ("coulombs-law/ignores-medium", "Forgets K")]


# ---------------------------------------------------------------- the file

def test_every_concept_has_named_mistakes():
    """A concept missing from the file gets questions whose wrong answers
    can never be traced to a mistake."""
    assert set(LIST["concepts"]) == {slug for slug, _, _ in CONCEPTS}


def test_ids_are_unique_and_carry_their_concept():
    """The concept prefix keeps 'inverse-r' under Coulomb's law apart from
    'inverse-r' under the electric field -- the same slip, a different topic."""
    ids = []
    for concept, items in LIST["concepts"].items():
        for item in items:
            assert item["id"].startswith(f"{concept}/")
            assert item["label"].strip()
            ids.append(item["id"])
    assert len(ids) == len(set(ids))


# ------------------------------------------------------ keeping real ids

def test_ids_from_the_list_are_kept():
    raw = [{"mistake": None}, {"mistake": "coulombs-law/inverse-r"}]
    assert _mistake_ids(raw, 0, OFFERED) == [None, "coulombs-law/inverse-r"]


def test_an_invented_id_is_dropped():
    """Kept, it would record a mistake the student did not make."""
    raw = [{"mistake": None}, {"mistake": "coulombs-law/forgets-square"}]
    assert _mistake_ids(raw, 0, OFFERED) == [None, None]


def test_other_means_no_named_mistake():
    raw = [{"mistake": None}, {"mistake": "other"}]
    assert _mistake_ids(raw, 0, OFFERED) == [None, None]


def test_the_correct_option_never_names_a_mistake():
    raw = [{"mistake": "coulombs-law/inverse-r"}, {"mistake": "coulombs-law/ignores-medium"}]
    assert _mistake_ids(raw, 0, OFFERED) == [None, "coulombs-law/ignores-medium"]


def test_a_missing_or_odd_value_is_no_mistake():
    raw = [{}, {"mistake": 3}, {}]
    assert _mistake_ids(raw, 2, OFFERED) == [None, None, None]


# ------------------------------------------------------------ the prompt

class _RecordingLLM:
    model = "fake"

    def __init__(self, reply: str):
        self.reply, self.prompt = reply, None

    def complete(self, system, messages, max_tokens):
        self.prompt = messages[0]["content"]
        return self.reply


REPLY = json.dumps(
    {
        "stem": "How does the force change when the distance doubles?",
        "options": [
            {"text": "It becomes a quarter", "misconception": None, "mistake": None},
            {"text": "It halves", "misconception": "inverse r", "mistake": "coulombs-law/inverse-r"},
            {"text": "It doubles", "misconception": "backwards", "mistake": "other"},
            {"text": "It stays the same", "misconception": "ignores r", "mistake": "made-up/id"},
        ],
        "correct_index": 0,
        "explanation": "The force goes as 1/r², so doubling r quarters it.",
    }
)


def test_the_prompt_lists_the_concepts_mistakes():
    llm = _RecordingLLM(REPLY)
    generate_question(llm, "passage", "Coulomb's law", mistakes=OFFERED)
    for slug, label in OFFERED:
        assert f"- {slug}: {label}" in llm.prompt


def test_without_a_list_the_prompt_has_no_mistake_section():
    llm = _RecordingLLM(REPLY)
    generate_question(llm, "passage", "Coulomb's law")
    assert "Common mistakes" not in llm.prompt


def test_a_generated_question_keeps_only_listed_ids():
    generated = generate_question(_RecordingLLM(REPLY), "passage", "Coulomb's law", mistakes=OFFERED)
    assert generated.mistake_ids == [None, "coulombs-law/inverse-r", None, None]
