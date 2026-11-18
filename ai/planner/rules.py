"""The study plan, from rules: which quizzes to do on each day until the exam.

No model writes the plan. Every session comes from a rule a student or a
teacher can read back ("Gauss's law -- weak, 42% over 9 questions"), it
costs nothing to rebuild, and it is rebuilt whenever the facts change: a
quiz lands, a day is missed, the hours change.

The plan is quizzes, topic by topic. Understanding a topic first is the
student's choice: every session has an "Explain it" beside its Start, which
opens the 7-step explanation, and nothing in the plan depends on it
(Ibrahim, 18 Nov: explanations when a student wants them, not as lessons).

Each concept in the exam's chapters that has questions is in one state:

    new      fewer than 3 answers                      -> a first quiz, in syllabus order
    weak     reliably under 60%                        -> a quiz every 2 days,
                                                          its weak prerequisite first
    shaky    neither weak nor strong                   -> a quiz every 3 days
    strong   reliably 80% or more                      -> a quiz every 7 days, to keep it

First quizzes are paced so every concept has had one before the final
stretch (the last quarter of the time, at most a week), which adds timed
tests. Each day is filled in priority order until its minutes run out: a
test on test days, weak concepts, first quizzes at the pace needed, other
quizzes that are due, then first quizzes ahead of pace.

Future days are a forecast: they assume each quiz is done, and that a quiz
moves a concept up one state. The real results replace the forecast as
they come in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

QUIZ_MINUTES = 15
TEST_MINUTES = 30
QUIZ_QUESTIONS = 5
TEST_QUESTIONS = 15

WEAK_BELOW = 0.6           # as ai.mastery.estimate.WEAK_THRESHOLD
STRONG_FROM = 0.8          # as ai.tutor.pitch: "pitched deeper"
# Answers before a concept counts as started.
MIN_ANSWERS = 3

# Days between quizzes on a concept, by its state.
GAP = {"weak": 2, "shaky": 3, "strong": 7}
STRONG_GAP_IN_STRETCH = 3
MAX_WEAK_PER_DAY = 2
# First quizzes in a day, unless the pace needs more.
MAX_NEW_PER_DAY = 3
# Other due quizzes in a day. A Test me over the whole syllabus makes many
# concepts due at once; they are spread over the following days.
MAX_SHAKY_PER_DAY = 3
MAX_STRETCH_DAYS = 7


@dataclass
class ConceptState:
    """What the planner needs to know about one concept."""
    id: object
    slug: str
    name: str
    order: tuple                 # curriculum order: chapter, then topic number
    questions: int               # approved questions a quiz could use
    attempts: int = 0
    mastery: float = 0.5
    reliable: bool = False
    last_activity: date | None = None
    prerequisite_id: object = None   # its weak prerequisite, when one is blamed

    @property
    def state(self) -> str:
        if self.attempts < MIN_ANSWERS:
            return "new"
        if self.reliable and self.mastery < WEAK_BELOW:
            return "weak"
        if self.reliable and self.mastery >= STRONG_FROM:
            return "strong"
        return "shaky"


@dataclass
class Session:
    kind: str                    # quiz | test
    concept_ids: list
    label: str
    minutes: int
    reason: str
    count: int | None = None     # questions


@dataclass
class Day:
    date: date
    minutes: int                 # the time available that day
    sessions: list[Session] = field(default_factory=list)
    stretch: bool = False        # in the final stretch before the exam

    @property
    def used(self) -> int:
        return sum(s.minutes for s in self.sessions)


@dataclass
class Plan:
    days: list[Day]
    warnings: list[str]
    unstarted_by_exam: list[str]
    no_questions: list[str] = field(default_factory=list)


def _pct(c: ConceptState) -> str:
    return f"{round(c.mastery * 100)}% over {c.attempts} question{'s' if c.attempts != 1 else ''}"


def build(concepts: list[ConceptState], start: date, exam: date,
          minutes_by_weekday: list[int], horizon: int | None = None) -> Plan:
    """Days from start (included) to the day before the exam.

    minutes_by_weekday: seven numbers, Monday first. horizon: how many days
    to return; the whole run to the exam is still planned, so the pace and
    the warnings account for every day."""
    if exam <= start:
        return Plan([], ["The exam date has to be after today."], [])
    if len(minutes_by_weekday) != 7 or not any(m > 0 for m in minutes_by_weekday):
        return Plan([], ["Add some study time on at least one day of the week."], [])

    no_questions = [c.name for c in sorted(concepts, key=lambda c: c.order) if not c.questions]
    plannable = [c for c in concepts if c.questions]
    warnings: list[str] = []
    if no_questions:
        warnings.append(
            f"{len(no_questions)} topic{'s have' if len(no_questions) != 1 else ' has'} no quiz "
            "questions yet, so the plan leaves them out.")
    if not plannable:
        return Plan([], warnings or ["None of the chosen chapters has topics to plan yet."], [],
                    no_questions)

    total = (exam - start).days
    stretch_days = min(MAX_STRETCH_DAYS, total // 4)
    stretch_from = exam - timedelta(days=stretch_days)
    days = [Day(start + timedelta(days=i), minutes_by_weekday[(start + timedelta(days=i)).weekday()])
            for i in range(total)]
    for d in days:
        d.stretch = d.date >= stretch_from

    state = {c.id: replace(c) for c in sorted(plannable, key=lambda c: c.order)}
    due: dict = {}
    for c in state.values():
        s = c.state
        if s != "new":
            when = (c.last_activity + timedelta(days=GAP[s])) if c.last_activity else start
            due[c.id] = max(when, start)

    learning_days = [d for d in days if not d.stretch and d.minutes >= QUIZ_MINUTES]
    test_count = min(TEST_QUESTIONS, sum(c.questions for c in state.values()))
    stretch_seen = 0

    for day in days:
        budget = day.minutes
        today_ids: set = set()

        def add(session: Session) -> bool:
            nonlocal budget
            if session.minutes > budget or any(i in today_ids for i in session.concept_ids):
                return False
            day.sessions.append(session)
            budget -= session.minutes
            today_ids.update(session.concept_ids)
            return True

        # 1. Timed tests in the final stretch, every other study day, not the eve.
        if day.stretch and day.minutes > 0:
            stretch_seen += 1
            eve = day.date == exam - timedelta(days=1)
            if stretch_seen % 2 == 1 and not eve and test_count:
                add(Session("test", list(state), f"Test: everything so far ({test_count} questions)",
                            TEST_MINUTES, "Exam practice under time, in the final stretch.",
                            test_count))

        def quiz(c: ConceptState, reason: str) -> bool:
            n = min(QUIZ_QUESTIONS, c.questions)
            if add(Session("quiz", [c.id], c.name, QUIZ_MINUTES, reason, n)):
                _after_quiz(c, day.date, due, day.stretch)
                return True
            return False

        is_due = [c for c in state.values() if c.id in due and due[c.id] <= day.date]

        # 2. Weak concepts, each with its weak prerequisite first.
        foundation_of = {c.prerequisite_id: c.name for c in state.values()
                         if c.prerequisite_id is not None and c.state == "weak"}

        def weak_reason(c: ConceptState) -> str:
            if c.id in foundation_of:
                return f"Weak foundation for {foundation_of[c.id]}: {c.name} is at {_pct(c)}."
            return f"Weak: {_pct(c)}."

        weak_done = 0
        for c in sorted([c for c in is_due if c.state == "weak"], key=lambda c: c.mastery):
            if weak_done >= MAX_WEAK_PER_DAY:
                break
            pre = state.get(c.prerequisite_id)
            if pre is not None and pre.id not in today_ids and pre.state != "new":
                quiz(pre, weak_reason(pre) if pre.state == "weak"
                     else f"Weak foundation for {c.name}: {pre.name} is at {_pct(pre)}.")
            if c.id not in today_ids and quiz(c, weak_reason(c)):
                weak_done += 1

        # 3. First quizzes, at the pace that starts everything before the stretch.
        new = [c for c in state.values() if c.state == "new"]
        remaining = [d for d in learning_days if d.date >= day.date]
        pace = math.ceil(len(new) / len(remaining)) if remaining else len(new)
        started = 0
        for c in new:
            if started >= pace:
                break
            reason = ("Not started yet, and the exam is close." if day.stretch
                      else "Next in the syllabus.")
            if quiz(c, reason):
                started += 1

        # 4. Other quizzes that are due, a few a day, weakest first.
        shaky_done = 0
        for c in sorted([c for c in is_due if c.state in ("shaky", "strong")
                         and c.id not in today_ids], key=lambda c: c.mastery):
            if shaky_done >= MAX_SHAKY_PER_DAY:
                break
            reason = (f"Keep it going: {_pct(c)}." if c.state == "shaky"
                      else f"Strong ({_pct(c)}); a quick quiz so it stays that way.")
            if quiz(c, reason):
                shaky_done += 1

        # 5. Time left: get ahead, up to a day's worth.
        if not day.stretch:
            for c in [c for c in state.values() if c.state == "new"]:
                if started >= max(pace, MAX_NEW_PER_DAY):
                    break
                if not quiz(c, "Ahead of schedule: you have time today."):
                    break
                started += 1

    unstarted = [c.name for c in state.values() if c.state == "new"]
    if unstarted:
        warnings.append(
            f"At these hours, {len(unstarted)} topic{'s' if len(unstarted) != 1 else ''} "
            "won't get a quiz before the exam. More time on some days would cover them.")
    shown = days[:horizon] if horizon else days
    return Plan(shown, warnings, unstarted, no_questions)


def _after_quiz(c: ConceptState, when: date, due: dict, stretch: bool) -> None:
    """The forecast after a quiz: one state up, due again after that
    state's gap. The real result replaces this when it lands."""
    s = c.state
    if s == "new":
        c.mastery, c.reliable = 0.65, False          # shaky
    elif s == "weak":
        c.mastery, c.reliable = 0.7, True            # shaky
    elif s == "shaky":
        c.mastery, c.reliable = 0.85, True           # strong
    c.attempts = max(c.attempts, MIN_ANSWERS) + QUIZ_QUESTIONS
    c.last_activity = when
    new_state = c.state
    gap = STRONG_GAP_IN_STRETCH if (stretch and new_state == "strong") else GAP[new_state]
    due[c.id] = when + timedelta(days=gap)
