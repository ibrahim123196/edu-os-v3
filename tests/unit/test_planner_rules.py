"""The study plan's rules, without a database."""

from __future__ import annotations

from datetime import date, timedelta

from ai.planner.rules import MAX_NEW_PER_DAY, MAX_SHAKY_PER_DAY, QUIZ_MINUTES, ConceptState, build

MON = date(2026, 11, 16)            # a Monday
WEEKDAYS = [60, 60, 60, 60, 60, 180, 0]


def concepts(n=10, questions=6):
    return [ConceptState(i, f"c{i}", f"Concept {i}", (1, i), questions) for i in range(n)]


def labels(day):
    return [s.label for s in day.sessions]


def test_the_plan_is_quizzes_only():
    """Explanations are the student's choice, beside each quiz; never a session."""
    plan = build(concepts(), MON, MON + timedelta(days=30), WEEKDAYS)
    assert {s.kind for d in plan.days for s in d.sessions} <= {"quiz", "test"}


def test_first_quizzes_go_in_syllabus_order():
    plan = build(concepts(), MON, MON + timedelta(days=30), WEEKDAYS)
    firsts = [s.label for d in plan.days for s in d.sessions if s.reason == "Next in the syllabus."
              or s.reason.startswith("Ahead of schedule")]
    assert firsts == [f"Concept {i}" for i in range(10)]


def test_every_topic_gets_a_quiz_before_the_final_stretch():
    exam = MON + timedelta(days=28)
    plan = build(concepts(20), MON, exam, WEEKDAYS)
    assert not plan.unstarted_by_exam
    for d in plan.days:
        if d.stretch:
            assert not any(s.reason in ("Next in the syllabus.",) for s in d.sessions)


def test_a_weak_concept_comes_with_its_weak_prerequisite_first():
    cs = concepts(3)
    cs[0].attempts, cs[0].mastery, cs[0].reliable = 8, 0.3, True     # the foundation
    cs[2].attempts, cs[2].mastery, cs[2].reliable = 8, 0.45, True
    cs[2].prerequisite_id = cs[0].id
    plan = build(cs, MON, MON + timedelta(days=20), WEEKDAYS)
    first = labels(plan.days[0])
    assert first.index("Concept 0") < first.index("Concept 2")
    reason = next(s.reason for s in plan.days[0].sessions if s.label == "Concept 0")
    assert reason.startswith("Weak foundation for Concept 2")


def test_strong_concepts_come_back_for_a_quick_quiz():
    cs = concepts(1)
    cs[0].attempts, cs[0].mastery, cs[0].reliable = 12, 0.9, True
    cs[0].last_activity = MON - timedelta(days=10)
    plan = build(cs, MON, MON + timedelta(days=30), WEEKDAYS)
    assert plan.days[0].sessions[0].reason.startswith("Strong (90% over 12 questions)")
    assert len([d for d in plan.days if "Concept 0" in labels(d)]) >= 3


def test_topics_without_questions_are_left_out_and_named():
    cs = concepts(3)
    cs[1].questions = 0
    plan = build(cs, MON, MON + timedelta(days=20), WEEKDAYS)
    assert "Concept 1" not in [s.label for d in plan.days for s in d.sessions]
    assert plan.no_questions == ["Concept 1"]
    assert "1 topic has no quiz questions yet" in plan.warnings[0]


def test_one_or_two_answers_mean_not_started():
    """A whole-syllabus Test me leaves one answer on many concepts; that is
    not "in progress" (18 Nov: eleven quizzes in one day)."""
    cs = concepts(4)
    for c in cs:
        c.attempts, c.mastery = 1, 0.4
    plan = build(cs, MON, MON + timedelta(days=30), WEEKDAYS)
    assert plan.days[0].sessions[0].reason == "Next in the syllabus."


def test_quizzes_that_fall_due_together_are_spread_out():
    cs = concepts(10)
    for c in cs:
        c.attempts, c.mastery = 5, 0.65
        c.last_activity = MON - timedelta(days=5)
    plan = build(cs, MON, MON + timedelta(days=30), [300] * 7)
    assert len(plan.days[0].sessions) == MAX_SHAKY_PER_DAY
    assert len(plan.days[1].sessions) == MAX_SHAKY_PER_DAY


def test_the_final_stretch_has_timed_tests_but_not_on_the_eve():
    exam = MON + timedelta(days=28)
    plan = build(concepts(8), MON, exam, WEEKDAYS)
    tests = [d.date for d in plan.days if "test" in [s.kind for s in d.sessions]]
    assert tests and all(d >= exam - timedelta(days=7) for d in tests)
    assert exam - timedelta(days=1) not in tests


def test_a_rest_day_gets_nothing():
    plan = build(concepts(), MON, MON + timedelta(days=14), WEEKDAYS)
    sundays = [d for d in plan.days if d.date.weekday() == 6]
    assert sundays and all(not d.sessions for d in sundays)


def test_no_day_goes_over_its_minutes():
    plan = build(concepts(30), MON, MON + timedelta(days=40), [45, 30, 90, 20, 60, 240, 15])
    assert all(d.used <= d.minutes for d in plan.days)


def test_first_quizzes_per_day_are_capped_when_not_behind():
    plan = build(concepts(30), MON, MON + timedelta(days=200), [0, 0, 0, 0, 0, 300, 0])
    for d in plan.days:
        firsts = [s for s in d.sessions if s.reason in ("Next in the syllabus.",)
                  or s.reason.startswith("Ahead of schedule")]
        assert len(firsts) <= MAX_NEW_PER_DAY


def test_too_little_time_says_what_will_not_be_covered():
    plan = build(concepts(40), MON, MON + timedelta(days=8), [QUIZ_MINUTES] * 7)
    assert plan.unstarted_by_exam
    assert any("won't get a quiz before the exam" in w for w in plan.warnings)


def test_a_concept_appears_at_most_once_a_day():
    plan = build(concepts(12), MON, MON + timedelta(days=30), [300] * 7)
    for d in plan.days:
        seen = [tuple(s.concept_ids) for s in d.sessions if s.kind != "test"]
        assert len(seen) == len(set(seen))


def test_settings_that_cannot_be_planned_are_explained():
    assert build(concepts(), MON, MON, WEEKDAYS).warnings == ["The exam date has to be after today."]
    assert "study time" in build(concepts(), MON, MON + timedelta(days=9), [0] * 7).warnings[0]
    assert "has topics to plan" in build([], MON, MON + timedelta(days=9), WEEKDAYS).warnings[0]


def test_the_horizon_only_trims_what_is_shown():
    whole = build(concepts(), MON, MON + timedelta(days=30), WEEKDAYS)
    shown = build(concepts(), MON, MON + timedelta(days=30), WEEKDAYS, horizon=5)
    assert [labels(d) for d in shown.days] == [labels(d) for d in whole.days[:5]]


def test_the_students_own_facts_are_not_changed():
    cs = concepts(2)
    build(cs, MON, MON + timedelta(days=10), WEEKDAYS)
    assert all(c.attempts == 0 for c in cs)
