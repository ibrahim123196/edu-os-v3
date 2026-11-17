"""What the student wants from a message: one of five learning modes.

    learn     "teach me capacitance"            -> an answer (a lesson, later)
    review    "revise Gauss's law"              -> key points + their progress
    practice  "quiz me on capacitors"           -> a quiz on that topic
    test      "test me" / "mock test"           -> a longer quiz
    clarify   anything else                     -> an answer, as before

Phrase rules, not a model call. A mode decides whether the textbook is
searched at all, so it has to be right before anything else runs, and a
model call here would add a second or two and a cost to every message --
including the plain questions that are most of them. Rules are also
testable line by line. The guess is shown to the student ("Understood as:
practice questions on ..."), so a wrong one is visible and fixed by
rephrasing.

Clarify is the default: a message that matches nothing is a question, and
answering it is what the chat always did.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Mode(str, Enum):
    learn = "learn"
    review = "review"
    practice = "practice"
    test = "test"
    clarify = "clarify"


@dataclass
class ModeGuess:
    mode: Mode
    topic: str        # what the message is about, with the request words removed
    cue: str | None   # the words that decided it, for the trace


# Checked in this order: the first match wins. Test before practice, so
# "test me with questions on X" is a test; practice before review, so
# "practice questions to revise X" is practice -- answering a request for
# questions with a summary would ignore the one thing it asked for.
_RULES: list[tuple[Mode, re.Pattern]] = [
    (Mode.test, re.compile(
        r"\b(test (me|myself|my ?self)|mock (test|exam)|full test|exam me|"
        r"mixed (quiz|test|questions))\b")),
    (Mode.practice, re.compile(
        r"\b(quiz me|give me (a )?quiz|(a |start a |take a )quiz on|practi[cs]e|"
        r"(some |a few |more |\d+ )?(questions|mcqs?|problems|numericals) "
        r"(on|about|for|from|in)|ask me (some )?questions)\b")),
    (Mode.review, re.compile(
        r"\b(revise|revising|revision|review|recap|summari[sz]e|summary of|"
        r"key points|quick notes|go over)\b")),
    (Mode.learn, re.compile(
        r"\b(teach me|i want to learn|help me learn|learn about|"
        r"explain from (the )?(start|scratch|basics|beginning)|"
        r"start from (the )?(basics|beginning)|new to|introduce me to)\b")),
]

# Words that are part of the request, not the topic. Trimmed from the two
# ends only, so "can you give me some practice questions on gauss's law
# please" leaves "gauss's law" -- and "capacitors in series" keeps its "in".
_FILLER_WORDS = {
    "can", "could", "would", "will", "you", "please", "i", "want", "to", "i'd",
    "like", "give", "me", "let", "help", "some", "few", "more", "a", "an", "the",
    "question", "questions", "mcq", "mcqs", "problem", "problems", "numericals",
    "quiz", "test", "topic", "chapter", "concept", "on", "about", "for", "from",
    "in", "of", "myself", "with", "and", "now", "again",
    # "the whole chapter", "everything": a request for breadth, not a topic.
    "let's", "lets", "us", "we", "go", "do", "quickly", "quick",
    "whole", "entire", "everything", "all", "full", "syllabus", "portion",
}
_NUMBER = re.compile(r"^\d+$")

# A message opening like a question is asking about something, even when it
# names a mode: "what is a mixed quiz?", "what does the review say?".
# Requests open with the request ("quiz me", "revise", "can you test me").
_QUESTION_OPENING = re.compile(
    r"^(what|why|how|which|who|when|where|whose|is|are|was|were|does|do|did|"
    r"should|isn't|aren't|doesn't)\b")
_PUNCT = re.compile(r"[?!.,:;]+")
_WS = re.compile(r"\s+")

# A topic that only points back at the conversation. The caller resolves it
# from the earlier turns instead of searching for the word "this".
POINTERS = {"this", "that", "it", "these", "those", "this topic", "that topic",
            "same", "the same", "this one", "that one"}


def detect(message: str) -> ModeGuess:
    text = _normalise(message)
    if _QUESTION_OPENING.match(text):
        return ModeGuess(Mode.clarify, message.strip(), None)
    for mode, rule in _RULES:
        match = rule.search(text)
        if match:
            return ModeGuess(mode, _topic(text, match), match.group(0))
    return ModeGuess(Mode.clarify, message.strip(), None)


def points_back(topic: str) -> bool:
    """True when the topic is 'this', 'that' or 'it': it needs the conversation.

    Not when it is empty. "test me" after a question about capacitors means
    a test across everything; reading it as "test me on that" once turned it
    into a test on the last topic discussed.
    """
    return topic.strip().lower() in POINTERS


def _normalise(message: str) -> str:
    # Curly apostrophes from phones: "gauss’s" must match "gauss's".
    return _WS.sub(" ", message.replace("’", "'").replace("‘", "'")).strip().lower()


def _topic(text: str, match: re.Match) -> str:
    """The message without the request: the words around the cue, with
    request words trimmed from each end."""
    pieces = []
    for side in (text[: match.start()], text[match.end():]):
        words = _PUNCT.sub(" ", side).split()
        while words and _is_filler(words[0]):
            words.pop(0)
        while words and _is_filler(words[-1]):
            words.pop()
        if words:
            pieces.append(" ".join(words))
    # The topic usually follows the cue ("quiz me on X"); when it comes
    # first ("capacitors, quiz me") it is the only piece.
    return pieces[-1] if pieces else ""


def _is_filler(word: str) -> bool:
    return word in _FILLER_WORDS or bool(_NUMBER.match(word))
