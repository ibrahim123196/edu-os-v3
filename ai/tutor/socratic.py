"""Work it out: the tutor asks, the student reasons, step by step.

Socratic mode leads a student to an answer with a chain of small questions
instead of giving it. The chain is planned once, from the same passages
and under the same rules as any answer: each step is a question, the point
a right answer makes (with [S#] markers), and a hint that nudges without
giving the point away. The plan is checked for citations as one text, so
every step's markers refer to one list of sources, then split into steps.

Each reply is judged against the step's point, using the same passages:
right moves on; wrong asks again; the second wrong try brings the hint; the
third shows the step's point and moves on, so a student is never stuck.
The end is the answer the steps led to, with its sources.

The plan, the step reached and the tries are kept in the trace saved with
each tutor message, so a reply continues from exactly where the last one
left off, and nothing new is searched for mid-way: a fresh search on a
student's half-right answer could find passages that contradict the plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MIN_STEPS, MAX_STEPS = 2, 6
HINT_AFTER = 2          # wrong tries before the hint
SHOW_AFTER = 3          # wrong tries before the step is shown and passed

_SECTION = re.compile(r"^[ \t]*\[(GOAL|STEP)\][ \t]*$", re.MULTILINE)
_FIELD = re.compile(r"^[ \t]*(Q|POINT|HINT)[ \t]*:[ \t]*(.*)$", re.MULTILINE)
_MARKER = re.compile(r"\s*\[S\d+\]")


def instruction(question: str) -> str:
    return (
        f"The student wants to work this out for themselves: {question}\n\n"
        "Do not answer it. Plan how to lead them to the answer with a short "
        f"chain of questions ({MIN_STEPS} to {MAX_STEPS} steps), each small enough "
        "to answer in a sentence, each building on the one before, ending at "
        "the answer. Use only the sources.\n\n"
        "Write the plan in exactly this form, each tag on a line of its own:\n\n"
        "[GOAL]\n"
        "The answer the steps lead to, in two or three sentences, with [S#] "
        "markers.\n\n"
        "[STEP]\n"
        "Q: the question to ask the student\n"
        "POINT: what a right answer says, in one sentence, with [S#] markers\n"
        "HINT: a nudge towards the point that does not state it\n\n"
        "(one [STEP] block per step, in order)\n\n"
        "If the sources cannot support leading the student to this answer, "
        "write only the sentence that says the material does not cover it."
    )


@dataclass
class Step:
    question: str
    point: str
    hint: str


@dataclass
class Plan:
    goal: str = ""
    steps: list[Step] = field(default_factory=list)


def split(text: str) -> Plan | None:
    """The checked plan text as a plan, or None if it is not one: no goal,
    too few or too many steps, or a step missing its question or point."""
    found = list(_SECTION.finditer(text))
    plan = Plan()
    for i, m in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(text)
        body = text[m.end():end].strip()
        if m.group(1) == "GOAL":
            plan.goal = plan.goal or body
            continue
        fields = {k: v.strip() for k, v in _FIELD.findall(body)}
        if not fields.get("Q") or not fields.get("POINT"):
            return None
        plan.steps.append(Step(fields["Q"], fields["POINT"], fields.get("HINT", "")))
    if not plan.goal or not MIN_STEPS <= len(plan.steps) <= MAX_STEPS:
        return None
    return plan


def as_text(plan: Plan) -> str:
    """The plan as one text: what the tutor led to, for the conversation."""
    return plan.goal


JUDGE = (
    "A student is being led, step by step, to work something out.\n"
    "The question they were asked: {question}\n"
    "A right answer says: {point}\n"
    "They answered: {reply}\n\n"
    "Judge their answer against the sources above and the right answer. "
    "Wording does not matter, the physics or chemistry does.\n"
    "Reply in exactly two lines:\n"
    "VERDICT: right, partly or wrong\n"
    "FEEDBACK: one or two short, encouraging sentences to the student. If "
    "right, say what they got right. If partly or wrong, say what is right "
    "in their answer, if anything, and what to think about -- without "
    "giving the right answer away. No [S#] markers."
)

_VERDICT = re.compile(r"VERDICT\s*:\s*(right|partly|wrong)", re.I)
_FEEDBACK = re.compile(r"FEEDBACK\s*:\s*(.+)", re.I | re.S)


def read_verdict(raw: str) -> tuple[str, str]:
    """(verdict, feedback). Anything unreadable counts as not yet right:
    a step is never passed on a reply nobody judged."""
    raw = raw or ""
    verdict = _VERDICT.search(raw)
    feedback = _FEEDBACK.search(raw)
    text = _MARKER.sub("", feedback.group(1)).strip() if feedback else ""
    return (verdict.group(1).lower() if verdict else "wrong",
            text.splitlines()[0].strip() if text else "")


def start_state(plan: Plan) -> dict:
    return {
        "goal": plan.goal,
        "steps": [{"question": s.question, "point": s.point, "hint": s.hint} for s in plan.steps],
        "step": 0,
        "tries": 0,
        "done": False,
    }


def opening(state: dict) -> str:
    return f"Let's work it out together.\n\n{state['steps'][0]['question']}"


def advance(state: dict, verdict: str, feedback: str) -> tuple[dict, str, str]:
    """The next state, the tutor's reply, and what happened:
    "right", "again", "hint", "shown" (the step given after the last try)."""
    state = dict(state)
    steps = state["steps"]
    current = steps[state["step"]]
    parts = [feedback] if feedback else []

    if verdict == "right":
        event = "right"
        passed = True
    else:
        state["tries"] += 1
        if state["tries"] >= SHOW_AFTER:
            event, passed = "shown", True
            parts.append(f"Here's this step: {current['point']}")
        elif state["tries"] >= HINT_AFTER and current.get("hint"):
            event, passed = "hint", False
            parts.append(f"Hint: {current['hint']}")
            parts.append(f"Try again: {current['question']}")
        else:
            event, passed = "again", False
            parts.append(f"Try again: {current['question']}")

    if passed:
        state["step"] += 1
        state["tries"] = 0
        if state["step"] >= len(steps):
            state["done"] = True
            parts.append(f"You've worked it out. {state['goal']}")
        else:
            parts.append(steps[state["step"]]["question"])
    return state, "\n\n".join(parts), event
