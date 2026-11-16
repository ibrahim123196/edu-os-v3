"""Generate questions from a passage the student's own material contains.

Questions are generated from a specific passage, never from the model's own
knowledge of physics. That keeps them answerable from what the student has
actually read, keeps the cited explanation honest, and means a question that
turns out to be wrong can be traced to the text that produced it.

Distractors matter more than the stem. A wrong option that nobody would pick
teaches nothing; a wrong option that corresponds to a specific mistake --
forgetting to square the distance, confusing potential with potential energy
-- turns a wrong answer into a diagnosis. So the model is asked to name the
misconception behind each distractor, and that reasoning is stored with it.

The model returns JSON. That is parsed defensively: a model asked for JSON
will usually comply and will occasionally wrap it in prose or a code fence,
and a generator that falls over on a stray backtick is not much use.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ai.generation.llm import LLMClient, LLMError
from ai.quiz.validation import ValidationResult, validate_multiple_choice

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You write exam-practice questions for Indian school \
students (CBSE, JEE, NEET) from a passage of their course material.

Rules:
- Ask only about what the passage states. Never use outside knowledge, even \
if you are confident it is correct.
- Ask about the physics, not about the writing. Never write "according to \
the passage", "the passage states", or ask why the author chose a notation \
or phrasing. An exam paper does not reference the textbook it came from, \
and a student who memorises what a paragraph says has learnt nothing.
- Ask what a student must know or be able to do: state a relationship, say \
what happens to one quantity when another changes, identify the correct \
form of an expression, apply it to a described situation.
- If the passage cannot support a question at the requested difficulty, say \
so rather than inventing one.
- Exactly one option is correct, and it must be the only defensible one. \
Prefer questions with a single clear answer: what happens to a quantity, \
which expression holds, a short calculation. Ask "why" only when the \
wrong options are plainly false statements -- never the right reason in \
other words.
- Every wrong option must be a mistake a real student would make, and you \
must name that mistake. An option nobody would pick teaches nothing.
- Keep all four options a similar length. A longer correct answer can be \
picked without knowing any physics.
- Never write "all of the above" or "none of the above"; options get \
shuffled.
- Do not put the answer in the question.
- Write formulas in plain readable notation with real symbols, never LaTeX \
and never wrapped in dollar signs. The passage stores subscripts as q_1 and \
superscripts as r^2 because of how it was extracted; convert those to real \
characters.
- The explanation must say why the correct answer is right, in two or three \
sentences, and state the physics directly. Do not write "according to the \
passage" or quote it: the student reads the explanation after answering, \
with the passage no longer on screen, and the physics is true regardless of \
where it was written down.

Reply with JSON only, no prose and no code fence:

{"stem": "...", "options": [{"text": "...", "misconception": null, "mistake": null},
{"text": "...", "misconception": "why a student might pick this", "mistake": "an id from the list, or other"}],
"correct_index": 0, "explanation": "..."}

Set "stem" to null if the passage cannot support a fair question."""

DIFFICULTY_GUIDE = {
    1: "recall a definition, unit or stated fact",
    # Was "show understanding of why something is so". It is the default
    # difficulty, so nearly every question became "Why does X?" -- and the
    # wrong options of a why-question are other reasons, which the model
    # kept writing as the right reason in other words, marked wrong.
    2: "show understanding: predict what happens when something changes, or "
       "pick the relationship that holds",
    3: "apply a relationship to a situation described in the passage",
    4: "compare two ideas in the passage, or reason about a limiting case",
    5: "combine two statements from the passage to reach a conclusion",
}

USER_TEMPLATE = """Passage (from {title}{location}):

{content}

---

Write one multiple-choice question with four options.
Difficulty {difficulty}: {difficulty_guide}.
The question must be about: {concept}{mistakes}{already_asked}"""

MISTAKES = """

Common mistakes students make with {concept}. Build each wrong option on \
one of these where you can, and give its id as "mistake". Use "other" only \
when none fits -- a wrong id is worse than "other", because it records a \
mistake the student did not make:
{items}"""

ALREADY_ASKED = """

These questions already exist for this concept. Ask about something else \
in the passage; do not rephrase them:
{stems}"""


@dataclass
class GeneratedQuestion:
    stem: str
    options: list[str]
    correct_index: int
    explanation: str
    misconceptions: list[str | None]
    # The named mistake behind each option, as an id from the list given to
    # the model; None for the correct option, for "other", and for any id
    # the model made up.
    mistake_ids: list[str | None]
    validation: ValidationResult


class GenerationRefused(Exception):
    """The passage could not support a fair question."""


def _extract_json(raw: str) -> dict:
    """Pull a JSON object out of a reply that may be wrapped in prose."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object in the reply")


def _mistake_ids(
    raw_options: list[dict], correct_index: int, mistakes: list[tuple[str, str]]
) -> list[str | None]:
    """Keep only ids that were actually offered.

    A model asked to pick from a list will sometimes invent a plausible id
    instead. Kept, that would point at nothing; worse, a near-miss could be
    mistaken for a real one. Anything not on the list becomes "no named
    mistake", which the doctor counts.
    """
    allowed = {slug for slug, _ in mistakes}
    ids: list[str | None] = []
    for position, option in enumerate(raw_options):
        given = option.get("mistake")
        if position == correct_index or not isinstance(given, str):
            ids.append(None)
        elif given in allowed:
            ids.append(given)
        else:
            if given != "other":
                log.info("Generator named a mistake not on the list: %r", given)
            ids.append(None)
    return ids


def generate_question(
    llm: LLMClient,
    content: str,
    concept: str,
    title: str = "course material",
    location: str = "",
    difficulty: int = 2,
    existing_stems: list[str] | None = None,
    mistakes: list[tuple[str, str]] | None = None,
) -> GeneratedQuestion:
    """Ask for one question about `concept`, grounded in `content`.

    `existing_stems` are the questions already written for this concept.
    Without them the second question tends to rephrase the first: asked
    twice about Coulomb's law, the model produced two near-identical
    questions about why the constant is written as 1/(4*pi*epsilon_0).
    """
    already_asked = ""
    if existing_stems:
        already_asked = ALREADY_ASKED.format(
            stems="\n".join(f"- {stem}" for stem in existing_stems[:8])
        )

    listed = ""
    if mistakes:
        listed = MISTAKES.format(
            concept=concept,
            items="\n".join(f"- {slug}: {label}" for slug, label in mistakes),
        )

    prompt = USER_TEMPLATE.format(
        title=title,
        location=f", {location}" if location else "",
        content=content,
        difficulty=difficulty,
        difficulty_guide=DIFFICULTY_GUIDE.get(difficulty, DIFFICULTY_GUIDE[2]),
        concept=concept,
        mistakes=listed,
        already_asked=already_asked,
    )

    try:
        raw = llm.complete(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
        )
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Question generation failed: {exc}") from exc

    try:
        data = _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise GenerationRefused(f"Unparseable reply: {raw[:120]}") from exc

    stem = (data.get("stem") or "").strip()
    if not stem:
        raise GenerationRefused("The passage cannot support a fair question")

    raw_options = data.get("options") or []
    options = [str(o.get("text", "")).strip() for o in raw_options]
    misconceptions = [o.get("misconception") for o in raw_options]
    correct_index = int(data.get("correct_index", -1))
    explanation = (data.get("explanation") or "").strip()
    mistake_ids = _mistake_ids(raw_options, correct_index, mistakes or [])

    return GeneratedQuestion(
        stem=stem,
        options=options,
        correct_index=correct_index,
        explanation=explanation,
        misconceptions=misconceptions,
        mistake_ids=mistake_ids,
        validation=validate_multiple_choice(stem, options, correct_index, explanation),
    )
