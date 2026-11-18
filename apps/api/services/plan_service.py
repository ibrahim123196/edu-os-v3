"""The study plan: settings, today's sessions, the days ahead.

The rules live in ai/planner/rules.py; this gathers what they need from the
database -- the chapters' concepts in syllabus order, the student's mastery,
the quizzes done from the plan -- and keeps today's sessions once given.

Few round trips, whatever the plan holds: the syllabus comes from
curriculum_cache, and the quizzes behind every session shown are read in
three queries together, not three per session. Each trip to the database
across the internet costs a noticeable fraction of a second (18 Nov: the
Plan tab was the slow one).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ai.mastery.estimate import diagnose, estimate
from ai.planner import rules
from apps.api.planner import PlanExam, PlanSession, StudyPlan
from apps.api.quiz import Attempt, QuizSession, QuizSessionQuestion
from apps.api.services.curriculum_cache import graph
from apps.api.services.mastery_service import estimates_with_dates
from apps.api.services.tutor_service import concepts_with_questions

FORECAST_DAYS = 13
HISTORY_DAYS = 7
MAX_MINUTES_PER_DAY = 600
QUIZ_KINDS = ("quiz", "test", "practice")      # "practice": sessions stored before 18 Nov
MAX_EXAMS = 6


class PlanError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def today() -> date:
    """The student's today. One function, so tests can move the clock."""
    return date.today()


# --- what can be planned ------------------------------------------------------

def options(db: Session) -> list[dict]:
    """Chapters that have topics to plan, in syllabus order."""
    g = graph(db)
    return [{"id": str(ch.id), "subject": ch.subject, "number": ch.number, "part": ch.part,
             "name": ch.name, "concepts": g.concepts_per_chapter.get(ch.id, 0)}
            for ch in g.chapters if g.concepts_per_chapter.get(ch.id)]


def _plan(db: Session, student_id: uuid.UUID) -> StudyPlan | None:
    return db.scalars(select(StudyPlan).where(StudyPlan.student_id == student_id)).first()


def save_settings(db: Session, student_id: uuid.UUID, exams: list[dict],
                  minutes: list[int]) -> StudyPlan:
    """exams: [{"name", "date", "chapter_ids"}], one per exam, any order.
    The whole list is replaced, so removing an exam is leaving it out."""
    if len(minutes) != 7 or any(m < 0 or m > MAX_MINUTES_PER_DAY for m in minutes):
        raise PlanError(f"Give minutes for each of the 7 days, from 0 to {MAX_MINUTES_PER_DAY}.")
    if not any(minutes):
        raise PlanError("Add some study time on at least one day of the week.")
    if not exams:
        raise PlanError("Add at least one exam.")
    if len(exams) > MAX_EXAMS:
        raise PlanError(f"Plan up to {MAX_EXAMS} exams at a time.")
    known = {c["id"] for c in options(db)}
    cleaned = []
    for i, e in enumerate(exams, 1):
        name = (e.get("name") or "").strip()[:200] or f"Exam {i}"
        if e["date"] <= today():
            raise PlanError(f"{name}: the date has to be after today.")
        chosen = [str(c) for c in dict.fromkeys(e.get("chapter_ids") or [])]
        if not chosen:
            raise PlanError(f"{name}: choose at least one chapter.")
        if any(c not in known for c in chosen):
            raise PlanError(f"{name}: some chosen chapters have nothing to plan yet.", 404)
        cleaned.append((name, e["date"], chosen))
    names = [n.lower() for n, _, _ in cleaned]
    if len(set(names)) != len(names):
        raise PlanError("Give each exam a different name.")

    plan = _plan(db, student_id)
    if plan is None:
        plan = StudyPlan(student_id=student_id)
        db.add(plan)
    plan.minutes = [int(m) for m in minutes]
    plan.planned_day = None
    # An exam saved again under the same name keeps the day it was added.
    added = {e.name.lower(): e.set_on for e in plan.exams}
    plan.exams.clear()
    db.flush()
    for i, (name, when, chosen) in enumerate(sorted(cleaned, key=lambda x: x[1])):
        plan.exams.append(PlanExam(name=name, exam_date=when, chapter_ids=chosen, position=i,
                                   set_on=added.get(name.lower()) or today()))
    db.flush()
    # Today is planned again under the new settings; what was started stays.
    db.execute(delete(PlanSession).where(
        PlanSession.plan_id == plan.id, PlanSession.day >= today(),
        PlanSession.status == "todo", PlanSession.link_id.is_(None)))
    db.flush()
    return plan


def _exams(db: Session, plan: StudyPlan) -> list[rules.Exam]:
    """The plan's exams with the concepts each covers: those first taught in
    its chapters."""
    g = graph(db)
    out = []
    for e in plan.exams:
        chosen = {uuid.UUID(c) for c in e.chapter_ids}
        ids = frozenset(c.id for c in g.concepts if c.home is not None and c.home.id in chosen)
        out.append(rules.Exam(e.id, e.name, e.exam_date, ids, e.set_on))
    return out


# --- the facts the rules plan from -------------------------------------------

def concept_states(db: Session, student_id: uuid.UUID, plan: StudyPlan,
                   exams: list[rules.Exam] | None = None) -> list[rules.ConceptState]:
    g = graph(db)
    exams = _exams(db, plan) if exams is None else exams
    wanted = set().union(*(e.concept_ids for e in exams)) if exams else set()
    in_scope = [c for c in g.concepts if c.id in wanted]
    ids = [c.id for c in in_scope]
    stored, last_attempt = estimates_with_dates(db, student_id, ids) if ids else ({}, {})
    questions = concepts_with_questions(db)

    last_plan: dict = {}
    for day, concept_ids in db.execute(
            select(PlanSession.day, PlanSession.concept_ids)
            .where(PlanSession.plan_id == plan.id, PlanSession.status == "done")).all():
        for cid in concept_ids:
            cid = uuid.UUID(cid)
            last_plan[cid] = max(last_plan.get(cid, day), day)

    by_id = g.by_id()
    by_name = {c.name: c.id for c in in_scope}
    states = []
    for c in in_scope:
        est = stored.get(c.id, estimate([]))
        prereqs = {by_id[p].name: stored.get(p, estimate([])) for p in c.prerequisite_ids
                   if p in by_id}
        blamed = diagnose(c.name, est, prereqs).blamed_prerequisite
        seen = last_attempt.get(c.id)
        dates = [d for d in (seen.date() if seen else None, last_plan.get(c.id)) if d]
        order = (c.home.part or "", c.home.number, _topic_key(c.first_topic), c.name)
        states.append(rules.ConceptState(
            id=c.id, slug=c.slug, name=c.name, order=order, questions=questions.get(c.id, 0),
            attempts=est.attempts, mastery=est.mastery, reliable=est.is_reliable,
            last_activity=max(dates) if dates else None,
            prerequisite_id=by_name.get(blamed) if blamed else None,
        ))
    return states


def _topic_key(number: str | None) -> tuple:
    return tuple(int(p) if p.isdigit() else 0 for p in (number or "").split("."))


# --- the quizzes behind sessions ------------------------------------------------

def link(db: Session, student_id: uuid.UUID, session_id: uuid.UUID,
         quiz_session_id: uuid.UUID) -> PlanSession:
    """The quiz this session opened, or a reattempt of it. Finishing it
    ticks the session off; every attempt is kept, for the best score."""
    s = _own_session(db, student_id, session_id)
    if s.kind not in QUIZ_KINDS:
        raise PlanError("This session has no quiz.")
    if s.day > today():
        raise PlanError("That day hasn't come yet.")
    quiz = db.get(QuizSession, quiz_session_id)
    if quiz is None or quiz.student_id != student_id:
        raise PlanError("No such quiz.", 404)
    s.link_kind, s.link_id = "quiz", quiz.id
    if str(quiz.id) not in (s.quiz_ids or []):
        s.quiz_ids = [*(s.quiz_ids or []), str(quiz.id)]
    db.flush()
    refresh([s], db, _quiz_facts(db, [s]))
    return s


def _quiz_facts(db: Session, sessions: list[PlanSession]) -> dict:
    """For every quiz behind these sessions: whether it is over, right
    answers, answered, and questions asked. Three queries in all."""
    ids = {uuid.UUID(q) for s in sessions for q in (s.quiz_ids or [])}
    ids |= {s.link_id for s in sessions if s.link_kind == "quiz" and s.link_id}
    if not ids:
        return {}
    facts = {qid: {"finished": False, "ended": False, "right": 0, "answered": 0, "asked": 0}
             for qid in ids}
    for qid, finished_at, stopped in db.execute(
            select(QuizSession.id, QuizSession.finished_at, QuizSession.stopped)
            .where(QuizSession.id.in_(ids))).all():
        facts[qid]["finished"] = finished_at is not None
        facts[qid]["ended"] = stopped == "ended"
    for qid, right, answered in db.execute(
            select(Attempt.session_id, func.count().filter(Attempt.is_correct), func.count())
            .where(Attempt.session_id.in_(ids)).group_by(Attempt.session_id)).all():
        facts[qid]["right"], facts[qid]["answered"] = int(right or 0), int(answered or 0)
    for qid, asked in db.execute(
            select(QuizSessionQuestion.session_id, func.count())
            .where(QuizSessionQuestion.session_id.in_(ids))
            .group_by(QuizSessionQuestion.session_id)).all():
        facts[qid]["asked"] = int(asked)
    return facts


def _over(f: dict | None) -> bool:
    """Finished, or ended early with something answered."""
    return bool(f) and (f["finished"] or (f["ended"] and f["answered"] > 0))


def refresh(sessions: list[PlanSession], db: Session, facts: dict) -> None:
    """Tick off sessions whose quiz is over."""
    changed = False
    for s in sessions:
        if s.status == "todo" and s.link_kind == "quiz" and _over(facts.get(s.link_id)):
            s.status, s.done_how, s.done_at = "done", "auto", datetime.now(timezone.utc)
            changed = True
    if changed:
        db.flush()


def best_score(s: PlanSession, facts: dict) -> dict | None:
    """The best of this session's attempts, out of the questions asked: a
    reattempt that goes worse does not lower it (Ibrahim, 18 Nov). Only
    attempts that are over count, except while the first is under way."""
    tried = [facts[uuid.UUID(q)] for q in (s.quiz_ids or []) if uuid.UUID(q) in facts]
    if not tried and s.link_kind == "quiz" and s.link_id in facts:
        tried = [facts[s.link_id]]                  # linked before attempts were kept
    over = [f for f in tried if _over(f)]
    pool = over or [f for f in tried if f["answered"]]
    if not pool:
        return None
    best = max(pool, key=lambda f: (f["right"] / max(f["asked"] or f["answered"], 1), f["right"]))
    return {"right": best["right"], "out_of": best["asked"] or best["answered"],
            "attempts": len(over), "best": bool(over)}


def _own_session(db: Session, student_id: uuid.UUID, session_id: uuid.UUID) -> PlanSession:
    s = db.get(PlanSession, session_id)
    if s is None or s.plan.student_id != student_id:
        raise PlanError("No such session in your plan.", 404)
    return s


# --- the plan as the page shows it -------------------------------------------

def view(db: Session, student_id: uuid.UUID) -> dict:
    plan = _plan(db, student_id)
    if plan is None:
        return {"settings": None, "chapters": options(db)}
    now = today()
    exams = _exams(db, plan)
    states = concept_states(db, student_id, plan, exams)
    built = rules.build(states, now, exams, plan.minutes)

    # Today and the past week, in one read.
    window = db.scalars(select(PlanSession).where(
        PlanSession.plan_id == plan.id, PlanSession.day <= now,
        PlanSession.day >= now - timedelta(days=HISTORY_DAYS))
        .order_by(PlanSession.day, PlanSession.position)).all()
    todays = [s for s in window if s.day == now]
    # Sessions given out under the earlier rules (lessons, revision): no
    # longer part of the plan, so today is planned again without them.
    stale = [x for x in todays if x.kind not in QUIZ_KINDS and x.status == "todo"]
    if stale:
        for x in stale:
            db.delete(x)
        db.flush()
        plan.planned_day = None
    if plan.planned_day != now and built.days:
        _freeze(db, plan, built.days[0], [x for x in todays if x not in stale])
        plan.planned_day = now
        db.flush()
        window = db.scalars(select(PlanSession).where(
            PlanSession.plan_id == plan.id, PlanSession.day <= now,
            PlanSession.day >= now - timedelta(days=HISTORY_DAYS))
            .order_by(PlanSession.day, PlanSession.position)).all()
        todays = [s for s in window if s.day == now]
    history = [s for s in window if s.day < now]

    facts = _quiz_facts(db, window)
    refresh(window, db, facts)

    by_state: dict[str, int] = {}
    for c in states:
        by_state[c.state] = by_state.get(c.state, 0) + 1
    names = {c.id: c for c in states}

    return {
        "settings": {
            "exams": [{"id": str(e.id), "name": e.name, "date": e.exam_date.isoformat(),
                       "chapter_ids": e.chapter_ids, "days_left": (e.exam_date - now).days,
                       "topics": len(r.concept_ids), "past": e.exam_date <= now}
                      for e, r in zip(plan.exams, exams)],
            "minutes": plan.minutes,
        },
        "chapters": options(db),
        "today": now.isoformat(),
        "days_left": min([(e.date - now).days for e in exams if e.date > now], default=0),
        "topics": {"total": len(states), **by_state},
        "warnings": built.warnings,
        "unstarted": built.unstarted_by_exam,
        "no_questions": built.no_questions,
        "today_sessions": [_session_out(s, names, now, facts) for s in todays],
        "today_minutes": plan.minutes[now.weekday()],
        "upcoming": [
            {"date": d.date.isoformat(), "minutes": d.minutes, "stretch": d.stretch,
             "sessions": [_forecast_out(s) for s in d.sessions]}
            for d in built.days[1:1 + FORECAST_DAYS]
        ],
        "history": _history_out(history, names, now, facts),
    }


def _freeze(db: Session, plan: StudyPlan, day: rules.Day, kept: list[PlanSession]) -> None:
    """Today's sessions, stored so the day stays put while it is worked
    through. Concepts already on today's list are not given twice."""
    covered = {c for s in kept for c in s.concept_ids}
    position = max((s.position for s in kept), default=-1) + 1
    for s in day.sessions:
        ids = [str(c) for c in s.concept_ids]
        if s.kind != "test" and covered.intersection(ids):
            continue
        db.add(PlanSession(plan_id=plan.id, day=day.date, position=position, kind=s.kind,
                           concept_ids=ids, label=s.label, reason=s.reason,
                           minutes=s.minutes, question_count=s.count, quiz_ids=[]))
        position += 1
    db.flush()


def _open(kind: str, concepts: list, count: int | None) -> dict:
    """What the page does to start the session."""
    if kind in ("quiz", "practice") and concepts:
        return {"type": "quiz", "concept_slugs": [c.slug for c in concepts], "count": count or 5}
    if kind == "test":
        # One exam's chapters; a test stored before 19 Nov has every topic.
        return {"type": "test", "count": count or rules.TEST_QUESTIONS,
                "concept_slugs": [c.slug for c in concepts]}
    return {"type": "none"}


def _session_out(s: PlanSession, names: dict, now: date, facts: dict) -> dict:
    status = "missed" if (s.status == "todo" and s.day < now) else s.status
    concepts = [names[uuid.UUID(str(c))] for c in s.concept_ids if uuid.UUID(str(c)) in names]
    label = s.label
    if s.kind == "practice" and label.startswith("Practise: "):     # stored before 18 Nov
        label = label[len("Practise: "):].split(" (")[0]
    return {"id": str(s.id), "day": s.day.isoformat(), "kind": s.kind, "label": label,
            "reason": s.reason, "minutes": s.minutes, "count": s.question_count,
            "status": status, "started": s.link_id is not None,
            "score": best_score(s, facts),
            # The quiz "Reattempt" repeats: the same questions again.
            "quiz_id": str(s.link_id) if s.link_kind == "quiz" and s.link_id else None,
            "open": _open(s.kind, concepts, s.question_count),
            # The 7-step explanation, for a student who wants it. Optional.
            "explain": concepts[0].name if len(concepts) == 1 else None}


def _forecast_out(s: rules.Session) -> dict:
    return {"kind": s.kind, "label": s.label, "reason": s.reason, "minutes": s.minutes,
            "count": s.count}


def _history_out(sessions, names, now, facts) -> list[dict]:
    days: dict[str, list] = {}
    for s in sessions:
        days.setdefault(s.day.isoformat(), []).append(_session_out(s, names, now, facts))
    return [{"date": d, "sessions": ss} for d, ss in days.items()]
