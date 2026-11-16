"""Conversation memory: what earlier turns contribute to the next answer.

Two separate jobs, because they fail differently.

1. Search. "Explain that more simply" contains nothing to search for: its
   only content word is "explain", which every passage in the textbook
   matches equally badly. Before searching, the follow-up is rewritten into
   the question it stands for ("explain the electric field inside a
   conductor more simply"), using the turns before it.

2. Answering. The model sees the last few exchanges, so "that" and "more
   simply" mean something. The fresh sources for this turn are still the
   only thing it may state facts from; history says what was asked and how
   it was explained, not what is true.

The rewrite is told to keep the student's topic, not narrow it to a detail
of the last answer. "For example?" after "why is the field zero inside a
conductor" was once rewritten as "an example of how charges redistribute",
and the model then said the textbook had no example -- while holding the
passage on electrostatic shielding, which is exactly one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Three exchanges. Enough for "and why is that?" chains; more costs tokens on
# every turn and drags in topics the student has already left.
MAX_HISTORY_MESSAGES = 6
# An earlier answer is context, not material to re-read. Cut long ones so a
# single essay cannot crowd this turn's sources out of the prompt.
MAX_HISTORY_CHARS = 1200

_MARKER = re.compile(r"\s*\[S\d+\]")


@dataclass
class Turn:
    role: str      # "user" | "assistant"
    content: str


def recent_history(messages: list, limit: int = MAX_HISTORY_MESSAGES) -> list[Turn]:
    """Completed exchanges from stored messages, oldest first.

    Only question-and-answer pairs. A question whose answer never arrived
    (the model was down, the request failed) would put two student turns
    side by side, which the Anthropic API rejects outright -- one failed
    request would break the rest of the conversation.
    """
    pairs: list[Turn] = []
    pending: Turn | None = None
    for m in messages:
        if m.role == "user":
            pending = Turn("user", m.content)      # a later question replaces an unanswered one
        elif m.role == "assistant" and pending is not None:
            pairs += [pending, Turn("assistant", _clean_answer(m.content))]
            pending = None
    return pairs[-limit:] if limit else []


def _clean_answer(text: str) -> str:
    """An earlier answer as the model should see it now.

    Source markers are removed. [S1] in an old answer meant that turn's
    first source; this turn's [S1] is a different passage. Left in, the
    model copies old markers and the citation points the student at the
    wrong page.
    """
    text = _MARKER.sub("", text).strip()
    if len(text) > MAX_HISTORY_CHARS:
        text = text[:MAX_HISTORY_CHARS].rsplit(" ", 1)[0] + " …"
    return text


STANDALONE_PROMPT = """Here is a conversation between a student and a \
physics tutor, followed by the student's latest message.

{history}

Latest message: {question}

Rewrite the latest message as one complete question that makes sense on its \
own, naming the topic it refers to. Keep the student's own request (simpler, \
an example, why, a formula). If it already makes sense on its own, return it \
unchanged. Do not answer it.

Name the topic the student asked about, not a detail from the tutor's \
explanation. After "why is the electric field zero inside a conductor?", \
"for example?" means "give an example of the electric field being zero \
inside a conductor", not an example of one step in how the tutor explained it. \
Do not make the question narrower than the student's.

Reply with the question only."""


def standalone_question(llm, history: list[Turn], question: str) -> str | None:
    """The follow-up as a self-contained question, or None to keep the original.

    None when there is nothing to resolve (first turn, no model) or when the
    rewrite cannot be trusted -- a bad rewrite searches for the wrong topic,
    and searching for the student's own words is the safer failure.
    """
    if not history or llm is None or getattr(llm, "model", "") == "echo":
        return None

    lines = []
    for t in history:
        who = "Student" if t.role == "user" else "Tutor"
        lines.append(f"{who}: {t.content}")
    try:
        candidate = llm.complete(
            system="You rewrite a student's follow-up into a standalone question. "
                   "Reply with the question only.",
            messages=[{"role": "user", "content": STANDALONE_PROMPT.format(
                history="\n\n".join(lines), question=question)}],
            max_tokens=120,
            temperature=0.0,
        ).strip().strip('"').strip()
    except Exception:
        log.warning("Follow-up rewrite failed; searching the message as typed", exc_info=True)
        return None

    if not _usable(candidate):
        log.info("Follow-up rewrite rejected: %r", candidate)
        return None
    if candidate.lower().rstrip("?. ") == question.lower().rstrip("?. "):
        return None
    return candidate


def _usable(candidate: str) -> bool:
    """A question, not an answer or a refusal."""
    from ai.prompts.grounded_answer import INSUFFICIENT_EVIDENCE

    words = len(candidate.split())
    if not 2 <= words <= 40:
        return False             # an answer, or nothing
    if "\n" in candidate:
        return False             # explanations come in paragraphs; questions do not
    return INSUFFICIENT_EVIDENCE.rstrip(".").lower() not in candidate.lower()
