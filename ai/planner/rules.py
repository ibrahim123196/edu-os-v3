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
stretch of the exam it is in (the last quarter of the time, at most a
week), which adds timed tests on that exam's topics. Several exams share
the day's minutes, nearest exam first. Each day is filled in priority order until its minutes run out: a
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
class Exam:
    """One exam the student is preparing for: its date and the concepts it
    covers. Several can run at once -- a Physics mid-term in December and
    Chemistry in January -- sharing the same daily minutes."""
    id: object
    name: str
    date: date
    concept_ids: frozenset
    # The day it was added. Its final stretch is measured from then, so the
    # stretch stays put as the exam comes closer; without it, from today.
    set_on: date | None = None


@dataclass
class Plan:
    days: list[Day]
    warnings: list[str]
    unstarted_by_exam: dict           # exam name -> topics without a quiz before it
    no_questions: list[str] = field(default_factory=list)


def _pct(c: ConceptState) -> str:
    return f"{round(c.mastery * 100)}% over {c.attempts} question{'s' if c.attempts != 1 else ''}"


def _when(d: date) -> str:
    return f"{d.day} {d.strftime('%b')}"


def build(concepts: list[ConceptState], start: date, exams: list[Exam],
          minutes_by_weekday: list[int], horizon: int | None = None) -> Plan:
    """Days from start (included) to the day before the last exam.

    Exams already past are left out. A concept in several exams is due for
    the nearest one. The day's minutes are shared, nearest exam first: its
    first quizzes are paced to finish before its final stretch, and the
    stretch adds timed tests on its own chapters.

    minutes_by_weekday: seven numbers, Monday first. horizon: how many days
    to return; the whole run is still planned, so the pace and the warnings
    account for every day."""
    exams = sorted([e for e in exams if e.date > start], key=lambda e: e.date)
    if not exams:
        return Plan([], ["Add an exam with a date after today."], {})
    if len(minutes_by_weekday) != 7 or not any(m > 0 for m in minutes_by_weekday):
        return Plan([], ["Add some study time on at least one day of the week."], {})

    covered = set().union(*(e.concept_ids for e in exams))
    concepts = [c for c in concepts if c.id in covered]
    no_questions = [c.name for c in sorted(concepts, key=lambda c: c.order) if not c.questions]
    plannable = [c for c in concepts if c.questions]
    warnings: list[str] = []
    if no_questions:
        one = len(no_questions) == 1
        warnings.append(
            f"{len(no_questions)} topic{' has' if one else 's have'} no quiz "
            f"questions yet, so the plan leaves {'it' if one else 'them'} out.")
    if not plannable:
        return Plan([], warnings or ["None of the chosen chapters has topics to plan yet."], {},
                    no_questions)

    last = exams[-1].date
    days = [Day(start + timedelta(days=i), minutes_by_weekday[(start + timedelta(days=i)).weekday()])
            for i in range((last - start).days)]

    # Each exam's final stretch: the last quarter of the time to it, at most a week.
    stretch_from = {}
    for e in exams:
        since = min(e.set_on or start, start)
        stretch_from[e.id] = e.date - timedelta(days=min(MAX_STRETCH_DAYS, (e.date - since).days // 4))

    def stretch_exams(d: date) -> list[Exam]:
        return [e for e in exams if stretch_from[e.id] <= d < e.date]

    for d in days:
        d.stretch = bool(stretch_exams(d.date))

    # The exam each concept is due for: the nearest one covering it.
    first_exam = {}
    for e in exams:
        for cid in e.concept_ids:
            first_exam.setdefault(cid, e)

    def order(c: ConceptState) -> tuple:
        return (first_exam[c.id].date, c.order)

    state = {c.id: replace(c) for c in sorted(plannable, key=order)}
    due: dict = {}
    for c in state.values():
        s = c.state
        if s != "new":
            when = (c.last_activity + timedelta(days=GAP[s])) if c.last_activity else start
            due[c.id] = max(when, start)

    def exam_of(c: ConceptState, today: date) -> Exam | None:
        """The nearest exam still ahead that covers it."""
        return next((e for e in exams if e.date > today and c.id in e.concept_ids), None)

    tests_seen = {e.id: 0 for e in exams}
    many = len(exams) > 1

    for day in days:
        budget = day.minutes
        today_ids: set = set()
        in_stretch = stretch_exams(day.date)

        def add(session: Session) -> bool:
            nonlocal budget
            if session.minutes > budget or (session.kind != "test"
                                            and any(i in today_ids for i in session.concept_ids)):
                return False
            day.sessions.append(session)
            budget -= session.minutes
            if session.kind != "test":
                today_ids.update(session.concept_ids)
            return True

        def for_exam(c: ConceptState) -> str:
            e = exam_of(c, day.date)
            return f" For {e.name} ({_when(e.date)})." if many and e else ""

        # 1. Timed tests on each exam in its final stretch, every other study
        #    day, not on its eve, over that exam's own topics.
        if day.minutes > 0:
            for e in in_stretch:
                tests_seen[e.id] += 1
                eve = day.date == e.date - timedelta(days=1)
                ids = [cid for cid in state if cid in e.concept_ids]
                n = min(TEST_QUESTIONS, sum(state[cid].questions for cid in ids))
                if tests_seen[e.id] % 2 == 1 and not eve and n:
                    add(Session("test", ids, f"Test: {e.name} ({n} questions)", TEST_MINUTES,
                                f"Exam practice under time: {e.name} is on {_when(e.date)}.", n))

        def quiz(c: ConceptState, reason: str) -> bool:
            n = min(QUIZ_QUESTIONS, c.questions)
            if add(Session("quiz", [c.id], c.name, QUIZ_MINUTES, reason + for_exam(c), n)):
                e = exam_of(c, day.date)
                _after_quiz(c, day.date, due, bool(e and e in in_stretch))
                return True
            return False

        live = [c for c in state.values() if exam_of(c, day.date) is not None]
        is_due = [c for c in live if c.id in due and due[c.id] <= day.date]

        # 2. Weak concepts, nearest exam first, each with its weak prerequisite first.
        foundation_of = {c.prerequisite_id: c.name for c in live
                         if c.prerequisite_id is not None and c.state == "weak"}

        def weak_reason(c: ConceptState) -> str:
            if c.id in foundation_of:
                return f"Weak foundation for {foundation_of[c.id]}: {c.name} is at {_pct(c)}."
            return f"Weak: {_pct(c)}."

        weak_done = 0
        for c in sorted([c for c in is_due if c.state == "weak"],
                        key=lambda c: (exam_of(c, day.date).date, c.mastery)):
            if weak_done >= MAX_WEAK_PER_DAY:
                break
            pre = state.get(c.prerequisite_id)
            if pre is not None and pre.id not in today_ids and pre.state != "new":
                quiz(pre, weak_reason(pre) if pre.state == "weak"
                     else f"Weak foundation for {c.name}: {pre.name} is at {_pct(pre)}.")
            if c.id not in today_ids and quiz(c, weak_reason(c)):
                weak_done += 1

        # 3. First quizzes, at the pace each exam needs to have everything
        #    started before its stretch; the most demanding exam sets it.
        new = [c for c in live if c.state == "new"]
        pace = 0
        for e in exams:
            if e.date <= day.date:
                continue
            needed = [c for c in new if first_exam[c.id].date <= e.date]
            days_left = [d for d in days if day.date <= d.date < stretch_from[e.id]
                         and d.minutes >= QUIZ_MINUTES]
            if needed:
                pace = max(pace, math.ceil(len(needed) / len(days_left)) if days_left else len(needed))
        started = 0
        for c in new:
            if started >= pace:
                break
            e = exam_of(c, day.date)
            reason = ("Not started yet, and the exam is close." if e in in_stretch
                      else "Next in the syllabus.")
            if quiz(c, reason):
                started += 1

        # 4. Other quizzes that are due, a few a day, nearest exam and weakest first.
        shaky_done = 0
        for c in sorted([c for c in is_due if c.state in ("shaky", "strong")
                         and c.id not in today_ids],
                        key=lambda c: (exam_of(c, day.date).date, c.mastery)):
            if shaky_done >= MAX_SHAKY_PER_DAY:
                break
            reason = (f"Keep it going: {_pct(c)}." if c.state == "shaky"
                      else f"Strong ({_pct(c)}); a quick quiz so it stays that way.")
            if quiz(c, reason):
                shaky_done += 1

        # 5. Time left: get ahead, up to a day's worth, on exams not yet in
        #    their stretch.
        for c in [c for c in live if c.state == "new" and exam_of(c, day.date) not in in_stretch]:
            if started >= max(pace, MAX_NEW_PER_DAY):
                break
            if not quiz(c, "Ahead of schedule: you have time today."):
                break
            started += 1

    # What never got a first quiz before the exam it was due for.
    unstarted: dict = {}
    first_quiz: dict = {}
    for d in days:
        for sess in d.sessions:
            if sess.kind == "quiz":
                first_quiz.setdefault(sess.concept_ids[0], d.date)
    for e in exams:
        missing = [state[cid].name for cid in state if cid in e.concept_ids
                   and (cid not in first_quiz or first_quiz[cid] >= e.date)
                   and plannable_new(cid, concepts)]
        if missing:
            unstarted[e.name] = missing
            warnings.append(
                f"{e.name}: at these hours, {len(missing)} topic{'s' if len(missing) != 1 else ''} "
                "won't get a quiz before it. More time on some days would cover them.")
    shown = days[:horizon] if horizon else days
    return Plan(shown, warnings, unstarted, no_questions)


def plannable_new(cid, concepts) -> bool:
    """Was it still to start when planning began (so a missing quiz matters)?"""
    return any(c.id == cid and c.state == "new" for c in concepts)


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
