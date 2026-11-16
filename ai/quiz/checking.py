"""An independent check of a generated question, before it is used.

Validation looks at shape: four options, one correct, an explanation. It
never asked whether the physics is right, and a review of 128 approved
questions found 19 that were not -- an answer key saying the potential
quarters when the distance doubles, options marked wrong that were the
right reason in other words, a question built on a false premise. Others
were filed under the wrong concept or repeated an earlier question.

So a second model call answers the question without seeing the key. It
judges every option on its own, says which concept the question tests, and
says whether an existing question already asks the same thing. The
question is kept only if exactly one option is right, it is the keyed one,
and the concept and novelty checks pass.

A model checking a model still misses things: this catches slips, it does
not replace a person reading the question.

Word overlap was tried for spotting repeats first and measured against the
reviewed bank: rewordings of one question share fewer words than different
questions about the same worked example, so no threshold separated them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ai.generation.llm import LLMClient

SYSTEM_PROMPT = """You check physics exam questions for Indian school \
students (CBSE, JEE, NEET) before they are used. You are given a question \
and its options, but not the answer key. Be strict: a student who knows \
the physics must be able to pick exactly one option, and no other option \
may also be defensible.

Judge each option on the physics alone. An option that states a correct \
reason in different words from another option is also correct -- say so. \
If the question assumes something false, say so.

Reply with JSON only, no prose and no code fence:

{"options": [{"correct": true or false, "why": "one short sentence"}, ...],
 "false_premise": null or "what the question wrongly assumes",
 "tests_concept": true or false,
 "concept_note": "if false: what the question is really about",
 "repeats": null or the number of the existing question it repeats}"""

USER_TEMPLATE = """Concept the question is meant to test: {concept}

Context the question was written from (it may refer to an example in it):
{passage}

Question: {stem}

Options:
{options}

Existing questions on this concept (does the new one ask the same thing, \
with the same answer, in other words?):
{existing}"""


@dataclass
class CheckResult:
    ok: bool
    reason: str = ""


def check_question(
    llm: LLMClient,
    *,
    concept: str,
    passage: str,
    stem: str,
    options: list[str],
    correct_index: int,
    existing_stems: list[str],
) -> CheckResult:
    prompt = USER_TEMPLATE.format(
        concept=concept,
        passage=passage.strip(),
        stem=stem,
        options="\n".join(f"{i + 1}. {o}" for i, o in enumerate(options)),
        existing="\n".join(f"{i + 1}. {s}" for i, s in enumerate(existing_stems)) or "(none)",
    )
    # 900 tokens left three of 77 replies unreadable on the first run over
    # the bank, probably cut off mid-answer; an unreadable reply rejects a
    # question that may be fine, and it was paid for.
    raw = llm.complete(SYSTEM_PROMPT, [{"role": "user", "content": prompt}], max_tokens=1600)
    return judge(raw, correct_index, len(options), existing_stems)


def judge(raw: str, correct_index: int, option_count: int, existing_stems: list[str]) -> CheckResult:
    """Turn the checker's reply into keep or reject, with the reason.

    Anything that cannot be read is a rejection: a check that silently
    passes when it fails is how the faulty questions got in.
    """
    data = _parse(raw)
    if data is None:
        # What the model said, so the failure can be diagnosed instead of
        # guessed at -- the first run kept nothing and could not be.
        said = " ".join((raw or "").split())
        tail = f" ... {said[-80:]}" if len(said) > 160 else ""
        return CheckResult(False, f"check: reply could not be read: {said[:160]!r}{tail}")

    verdicts = data.get("options")
    if not isinstance(verdicts, list) or len(verdicts) != option_count:
        return CheckResult(False, "check: did not judge every option")

    right = [i for i, v in enumerate(verdicts) if isinstance(v, dict) and v.get("correct") is True]
    if data.get("false_premise"):
        return CheckResult(False, f"check: false premise: {str(data['false_premise'])[:160]}")
    if not right:
        return CheckResult(False, "check: no option is correct")
    if len(right) > 1:
        return CheckResult(False, f"check: {len(right)} options are defensible ({_numbers(right)})")
    if right[0] != correct_index:
        why = verdicts[right[0]].get("why", "")
        return CheckResult(
            False,
            f"check: key says option {correct_index + 1}, checker says {right[0] + 1}: {str(why)[:120]}",
        )
    if data.get("tests_concept") is not True:
        return CheckResult(False, f"check: wrong concept: {str(data.get('concept_note', ''))[:120]}")

    repeats = data.get("repeats")
    if isinstance(repeats, int) and 1 <= repeats <= len(existing_stems):
        return CheckResult(False, f"check: repeats: {existing_stems[repeats - 1][:120]}")

    return CheckResult(True)


def _numbers(indexes: list[int]) -> str:
    return ", ".join(str(i + 1) for i in indexes)


def _parse(raw: str) -> dict | None:
    text = raw.strip()
    fenced = re.search(r"\{.*\}", text, re.DOTALL)
    if fenced:
        text = fenced.group(0)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None
