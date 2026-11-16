"""Conversation memory: earlier turns, follow-up rewriting, the prompt."""

from __future__ import annotations

from types import SimpleNamespace as Msg

from ai.prompts.grounded_answer import INSUFFICIENT_EVIDENCE, USER_TEMPLATE, build_messages
from ai.tutor.conversation import (
    MAX_HISTORY_CHARS,
    Turn,
    recent_history,
    standalone_question,
)

HISTORY = [Turn("user", "why is the electric field zero inside a conductor"),
           Turn("assistant", "Free charges move until the field inside vanishes.")]


class FakeLLM:
    model = "fake"

    def __init__(self, reply="", fail=False):
        self.reply, self.fail, self.calls = reply, fail, []

    def complete(self, system, messages, max_tokens=None, temperature=None):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("API down")
        return self.reply


# --- which earlier turns are used -------------------------------------------

def test_history_is_question_answer_pairs_oldest_first():
    msgs = [Msg(role="user", content="q1"), Msg(role="assistant", content="a1"),
            Msg(role="user", content="q2"), Msg(role="assistant", content="a2")]
    assert [(t.role, t.content) for t in recent_history(msgs)] == [
        ("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")]


def test_a_question_that_was_never_answered_is_left_out():
    """Two student turns in a row are rejected by the API: one failed
    request would break every later turn of the conversation."""
    msgs = [Msg(role="user", content="lost"), Msg(role="user", content="q"),
            Msg(role="assistant", content="a"), Msg(role="user", content="also lost")]
    history = recent_history(msgs)
    assert [t.content for t in history] == ["q", "a"]
    roles = [t.role for t in history]
    assert roles == ["user", "assistant"] * (len(roles) // 2)


def test_only_the_last_three_exchanges():
    msgs = []
    for i in range(5):
        msgs += [Msg(role="user", content=f"q{i}"), Msg(role="assistant", content=f"a{i}")]
    history = recent_history(msgs)
    assert len(history) == 6 and history[0].content == "q2" and history[0].role == "user"


def test_old_source_markers_are_removed():
    """[S1] then meant another passage; left in, the model reuses it."""
    msgs = [Msg(role="user", content="q"),
            Msg(role="assistant", content="E is zero inside [S1]. Charges sit on the surface [S2][S3].")]
    assert recent_history(msgs)[1].content == "E is zero inside. Charges sit on the surface."


def test_a_long_answer_is_cut():
    msgs = [Msg(role="user", content="q"), Msg(role="assistant", content="word " * 1000)]
    assert len(recent_history(msgs)[1].content) <= MAX_HISTORY_CHARS + 2


def test_a_new_conversation_has_no_history():
    assert recent_history([]) == []


# --- rewriting a follow-up for the search -----------------------------------

def test_a_first_question_costs_no_extra_call():
    llm = FakeLLM("anything")
    assert standalone_question(llm, [], "what is drift velocity") is None
    assert llm.calls == []


def test_a_follow_up_is_rewritten_with_the_topic():
    llm = FakeLLM("Explain more simply why the electric field is zero inside a conductor?")
    got = standalone_question(llm, HISTORY, "explain that more simply")
    assert got == "Explain more simply why the electric field is zero inside a conductor?"
    prompt = llm.calls[0][0]["content"]
    assert "Student: why is the electric field zero inside a conductor" in prompt
    assert "Latest message: explain that more simply" in prompt


def test_an_answer_instead_of_a_question_is_not_searched():
    llm = FakeLLM("The field is zero because charges rearrange.\n\nThis happens instantly.")
    assert standalone_question(llm, HISTORY, "why") is None


def test_a_refusal_is_not_searched():
    assert standalone_question(FakeLLM(INSUFFICIENT_EVIDENCE), HISTORY, "and then?") is None


def test_a_failed_call_falls_back_to_the_typed_words():
    assert standalone_question(FakeLLM(fail=True), HISTORY, "why is that") is None


def test_an_unchanged_question_is_reported_as_unchanged():
    llm = FakeLLM("What is drift velocity?")
    assert standalone_question(llm, HISTORY, "what is drift velocity") is None


def test_no_model_or_echo_means_no_rewrite():
    echo = FakeLLM("x y z")
    echo.model = "echo"
    assert standalone_question(echo, HISTORY, "why") is None
    assert standalone_question(None, HISTORY, "why") is None


# --- the request the answer is written from ---------------------------------

def _chunk():
    return Msg(page_label="Page 18", document_title="NCERT", publisher="NCERT",
               source_kind="platform", section=None, content="E = 0 inside a conductor.")


def test_a_first_question_gets_exactly_the_old_request():
    messages = build_messages("what is E inside", [_chunk()])
    assert len(messages) == 1
    assert messages[0]["content"].endswith(
        USER_TEMPLATE.split("{question}")[1])


def test_a_follow_up_sees_the_earlier_turns_and_alternates():
    messages = build_messages("explain that more simply", [_chunk()], history=HISTORY,
                              standalone="explain simply why E is zero inside a conductor")
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    last = messages[-1]["content"]
    assert "[S1]" in last and "E = 0 inside a conductor." in last
    assert "Student's question: explain that more simply" in last
    assert "In this conversation that means: explain simply why E is zero" in last
    assert "they are not sources" in last


def test_the_rewrite_is_told_not_to_narrow_the_topic():
    """'for example?' once became 'an example of how charges redistribute',
    and the model then denied an example it was holding."""
    llm = FakeLLM("Give an example of the electric field being zero inside a conductor?")
    standalone_question(llm, HISTORY, "for example?")
    prompt = llm.calls[0][0]["content"]
    assert "not a detail from the tutor's" in prompt
    assert "Do not make the question narrower" in prompt
