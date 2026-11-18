"""The study plan through its API, on a database. Local only (see conftest).

A student of its own and a subject of its own, so nothing here touches
the real student's plan or the real syllabus.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, insert, select

from apps.api.curriculum import Chapter, Concept, Subject, Topic, topic_concepts
from apps.api.db import init_db, session_scope
from apps.api.main import app
from apps.api.models import Conversation, Message
from apps.api.quiz import Attempt, Question, QuestionKind, QuestionOption, QuestionStatus, QuizSession
from apps.api.routes import plan as plan_route
from apps.api.services import plan_service
from apps.api.student import Student

pytestmark = pytest.mark.integration

TAG = "t" + uuid.uuid4().hex[:5]
MON = date(2026, 11, 16)
WEEK = [60, 60, 60, 60, 60, 180, 0]


@pytest.fixture
def world(monkeypatch):
    init_db()
    clock = {"today": MON}
    with session_scope() as db:
        student = Student(id=uuid.uuid4(), display_name=f"plan {TAG}")
        other = Student(id=uuid.uuid4(), display_name=f"other {TAG}")
        subject = Subject(code=f"plan-{TAG}", name=f"Planning {TAG}")
        db.add_all([student, other, subject])
        db.flush()
        chapter = Chapter(subject_id=subject.id, number=1, name="Charges")
        db.add(chapter)
        db.flush()
        concepts = []
        for i, name in enumerate(["Charge", "Coulomb", "Field"]):
            topic = Topic(chapter_id=chapter.id, number=f"1.{i + 1}", name=name)
            concept = Concept(slug=f"plan-{TAG}-{i}", name=f"{name} {TAG}")
            db.add_all([topic, concept])
            db.flush()
            db.execute(insert(topic_concepts).values(topic_id=topic.id, concept_id=concept.id))
            for k in range(3):
                q = Question(concept_id=concept.id, kind=QuestionKind.multiple_choice,
                             stem=f"About {name} {TAG} {k}?", explanation="x",
                             status=QuestionStatus.approved)
                q.options = [QuestionOption(position=0, text="A", is_correct=True),
                             QuestionOption(position=1, text="B", is_correct=False)]
                db.add(q)
            concepts.append(concept)
        db.flush()
        ids = {"student": student.id, "other": other.id, "subject": subject.id,
               "chapter": chapter.id, "concepts": [c.id for c in concepts],
               "question": q.id,
               "questions": [x.id for x in db.scalars(
                   select(Question).where(Question.concept_id == concepts[0].id))]}

    me = {"id": ids["student"]}
    monkeypatch.setattr(plan_route, "active_student_id", lambda: me["id"])
    monkeypatch.setattr(plan_service, "today", lambda: clock["today"])
    with TestClient(app) as client:
        def setup(exams=None, minutes=WEEK, **over):
            """One exam by default; its fields can be overridden."""
            if exams is None:
                exams = [{"name": "Boards", "date": (MON + timedelta(days=30)).isoformat(),
                          "chapter_ids": [str(ids["chapter"])], **over}]
            return client.put("/plan/settings", json={"exams": exams, "minutes": minutes})
        yield client, setup, ids, clock, me

    with session_scope() as db:
        db.execute(delete(Question).where(Question.concept_id.in_(ids["concepts"])))
        db.execute(delete(Concept).where(Concept.id.in_(ids["concepts"])))
        db.execute(delete(Subject).where(Subject.id == ids["subject"]))
        db.execute(delete(Student).where(Student.id.in_([ids["student"], ids["other"]])))


def test_no_plan_yet_offers_the_chapters(world):
    client, _, ids, _, _ = world
    body = client.get("/plan").json()
    assert body["settings"] is None
    assert str(ids["chapter"]) in [c["id"] for c in body["chapters"]]


def test_setting_it_up_gives_quizzes_with_an_explain_beside_each(world):
    _, setup, _, _, _ = world
    r = setup()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["days_left"] == 30 and body["topics"] == {"total": 3, "new": 3}
    first = body["today_sessions"][0]
    assert first["kind"] == "quiz" and first["label"] == f"Charge {TAG}"
    assert first["open"] == {"type": "quiz", "concept_slugs": [f"plan-{TAG}-0"], "count": 3}
    assert first["explain"] == f"Charge {TAG}"                       # optional, beside Start
    assert first["score"] is None and first["status"] == "todo"
    assert len(body["upcoming"]) == 13
    kinds = {s["kind"] for d in body["upcoming"] for s in d["sessions"]}
    assert kinds <= {"quiz", "test"}


def test_today_stays_put_while_it_is_worked_through(world):
    client, setup, _, _, _ = world
    first = setup().json()["today_sessions"]
    again = client.get("/plan").json()["today_sessions"]
    assert [s["id"] for s in again] == [s["id"] for s in first]


def test_settings_that_cannot_be_planned_are_refused(world):
    _, setup, _, _, _ = world
    assert setup(date=MON.isoformat()).status_code == 400
    assert setup(minutes=[0] * 7).status_code == 400
    assert setup(minutes=[60] * 6).status_code == 422
    assert setup(minutes=[60, 60, 60, 60, 60, 60, 9999]).status_code == 400
    assert setup(chapter_ids=[str(uuid.uuid4())]).status_code == 404
    assert setup(chapter_ids=[]).status_code == 422


def test_several_exams_share_one_plan(world):
    client, setup, ids, _, _ = world
    ch = [str(ids["chapter"])]
    r = setup(exams=[
        {"name": "Finals", "date": (MON + timedelta(days=40)).isoformat(), "chapter_ids": ch},
        {"name": "Unit test", "date": (MON + timedelta(days=10)).isoformat(), "chapter_ids": ch},
    ])
    assert r.status_code == 200, r.text
    body = r.json()
    exams = body["settings"]["exams"]
    assert [e["name"] for e in exams] == ["Unit test", "Finals"]      # nearest first
    assert [e["days_left"] for e in exams] == [10, 40]
    assert body["days_left"] == 10
    assert len(body["upcoming"]) == 13
    # Saving again replaces the list: leaving one out removes it.
    r = setup(exams=[{"name": "Finals", "date": (MON + timedelta(days=40)).isoformat(),
                      "chapter_ids": ch}])
    assert [e["name"] for e in r.json()["settings"]["exams"]] == ["Finals"]


def test_exam_lists_that_cannot_be_planned_are_refused(world):
    _, setup, ids, _, _ = world
    one = {"name": "A", "date": (MON + timedelta(days=9)).isoformat(),
           "chapter_ids": [str(ids["chapter"])]}
    assert setup(exams=[]).status_code == 422
    assert setup(exams=[one, {**one, "name": "a"}]).status_code == 400     # same name
    assert setup(exams=[{**one, "name": f"E{i}"} for i in range(7)]).status_code == 422


def test_a_test_in_the_plan_covers_its_exam(world):
    client, setup, ids, clock, _ = world
    setup(exams=[{"name": "Boards", "date": (MON + timedelta(days=12)).isoformat(),
                  "chapter_ids": [str(ids["chapter"])]}], minutes=[300] * 7)
    ahead = client.get("/plan").json()["upcoming"]
    tests = [s for d in ahead for s in d["sessions"] if s["kind"] == "test"]
    assert tests and all(s["label"].startswith("Test: Boards") for s in tests)
    # On the day, it opens a test over that exam's topics only.
    clock["today"] = MON + timedelta(days=9)
    today = [s for s in client.get("/plan").json()["today_sessions"] if s["kind"] == "test"]
    assert today and today[0]["open"]["type"] == "test"
    assert sorted(today[0]["open"]["concept_slugs"]) == [f"plan-{TAG}-{i}" for i in range(3)]


def _today(client):
    return client.get("/plan").json()["today_sessions"]


def _quiz(student_id, answers=(), finished=False, stopped=None):
    """A quiz session with these answers (True = right)."""
    with session_scope() as db:
        quiz = QuizSession(student_id=student_id, kind="practice", stopped=stopped,
                           finished_at=datetime.now(timezone.utc) if finished else None)
        db.add(quiz)
        db.flush()
        qids = db.scalars(select(Question.id).where(Question.stem.like(f"%{TAG}%"))).all()
        for qid, right in zip(qids, answers):
            db.add(Attempt(student_id=student_id, question_id=qid, session_id=quiz.id,
                           is_correct=right))
        return quiz.id


def test_a_finished_quiz_ticks_its_session_with_its_score(world):
    client, setup, ids, _, me = world
    first = setup().json()["today_sessions"][0]
    quiz_id = _quiz(me["id"], answers=[True, False])
    r = client.post(f"/plan/sessions/{first['id']}/link", json={"quiz_session_id": str(quiz_id)})
    assert r.status_code == 200 and r.json()["status"] == "todo"        # started, not finished
    now = next(s for s in _today(client) if s["id"] == first["id"])
    assert now["started"]
    assert now["score"] == {"right": 1, "out_of": 2, "attempts": 0, "best": False}   # under way
    with session_scope() as db:
        db.get(QuizSession, quiz_id).finished_at = datetime.now(timezone.utc)
    after = next(s for s in _today(client) if s["id"] == first["id"])
    assert after["status"] == "done"


def test_a_quiz_ended_early_counts_only_with_something_answered(world):
    client, setup, ids, _, me = world
    first = setup().json()["today_sessions"][0]
    empty = _quiz(me["id"], stopped="ended")
    client.post(f"/plan/sessions/{first['id']}/link", json={"quiz_session_id": str(empty)})
    assert next(s for s in _today(client) if s["id"] == first["id"])["status"] == "todo"
    some = _quiz(me["id"], answers=[True], stopped="ended")
    client.post(f"/plan/sessions/{first['id']}/link", json={"quiz_session_id": str(some)})
    assert next(s for s in _today(client) if s["id"] == first["id"])["status"] == "done"


def test_a_reattempt_keeps_the_best_score(world):
    client, setup, _, _, me = world
    first = setup().json()["today_sessions"][0]
    one = _quiz(me["id"], answers=[False, False, True], finished=True)
    client.post(f"/plan/sessions/{first['id']}/link", json={"quiz_session_id": str(one)})
    done = next(s for s in _today(client) if s["id"] == first["id"])
    assert done["status"] == "done"
    assert done["score"] == {"right": 1, "out_of": 3, "attempts": 1, "best": True}
    assert done["quiz_id"] == str(one)                 # what "Reattempt" repeats
    again = _quiz(me["id"], answers=[True, True, True], finished=True)
    r = client.post(f"/plan/sessions/{first['id']}/link", json={"quiz_session_id": str(again)})
    assert r.status_code == 200
    redone = next(s for s in _today(client) if s["id"] == first["id"])
    assert redone["status"] == "done"
    assert redone["score"] == {"right": 3, "out_of": 3, "attempts": 2, "best": True}
    # A worse reattempt does not lower the best (Ibrahim, 18 Nov).
    worse = _quiz(me["id"], answers=[False, False, False], finished=True)
    client.post(f"/plan/sessions/{first['id']}/link", json={"quiz_session_id": str(worse)})
    kept = next(s for s in _today(client) if s["id"] == first["id"])
    assert kept["score"] == {"right": 3, "out_of": 3, "attempts": 3, "best": True}
    assert kept["quiz_id"] == str(worse)               # reattempt repeats the latest


def test_someone_elses_quiz_cannot_be_linked(world):
    client, setup, ids, _, _ = world
    first = setup().json()["today_sessions"][0]
    theirs = _quiz(ids["other"])
    r = client.post(f"/plan/sessions/{first['id']}/link", json={"quiz_session_id": str(theirs)})
    assert r.status_code == 404


def test_another_students_session_is_not_yours(world):
    client, setup, ids, _, me = world
    first = setup().json()["today_sessions"][0]
    mine = _quiz(me["id"])
    me["id"] = ids["other"]
    r = client.post(f"/plan/sessions/{first['id']}/link", json={"quiz_session_id": str(mine)})
    assert r.status_code == 404
    assert client.post(f"/plan/sessions/{uuid.uuid4()}/link",
                       json={"quiz_session_id": str(mine)}).status_code == 404


def test_a_missed_day_is_kept_as_missed_and_the_plan_moves_on(world):
    client, setup, _, clock, _ = world
    first = setup().json()["today_sessions"][0]
    clock["today"] = MON + timedelta(days=2)
    body = client.get("/plan").json()
    missed = [s for d in body["history"] for s in d["sessions"] if s["id"] == first["id"]]
    assert missed and missed[0]["status"] == "missed"
    assert body["today_sessions"][0]["label"] == f"Charge {TAG}"       # still first


def test_changing_the_hours_replans_today_but_keeps_what_was_started(world):
    client, setup, _, _, me = world
    first = setup().json()["today_sessions"]
    quiz_id = _quiz(me["id"], answers=[True], finished=True)
    client.post(f"/plan/sessions/{first[0]['id']}/link", json={"quiz_session_id": str(quiz_id)})
    body = setup(minutes=[200] * 7).json()
    ids = [s["id"] for s in body["today_sessions"]]
    assert first[0]["id"] in ids
    labels = [s["label"] for s in body["today_sessions"]]
    assert labels.count(f"Charge {TAG}") == 1                          # not given twice


def test_sessions_from_the_old_rules_are_replanned(world):
    """Lessons and revision were sessions before 18 Nov; any left unstarted
    on today are replaced by today's quizzes."""
    client, setup, _, _, _ = world
    setup()
    with session_scope() as db:
        from apps.api.planner import PlanSession
        s = db.scalars(select(PlanSession).where(PlanSession.label == f"Charge {TAG}")).first()
        s.kind, s.label = "lesson", f"Learn: Charge {TAG}"
    labels = [s["label"] for s in _today(client)]
    assert f"Learn: Charge {TAG}" not in labels and f"Charge {TAG}" in labels


def test_a_future_day_cannot_be_linked(world):
    client, setup, _, clock, me = world
    first = setup().json()["today_sessions"][0]
    clock["today"] = MON - timedelta(days=1)
    r = client.post(f"/plan/sessions/{first['id']}/link",
                    json={"quiz_session_id": str(_quiz(me["id"]))})
    assert r.status_code == 400


def test_the_plan_reads_the_database_a_fixed_number_of_times(world):
    """Each trip to Neon costs a fraction of a second: the Plan tab was the
    slow one (18 Nov). The count must not grow with the sessions shown."""
    from sqlalchemy import event

    from apps.api.db import engine

    client, setup, _, clock, me = world
    sessions = setup(minutes=[300] * 7).json()["today_sessions"]
    for s in sessions:
        quiz_id = _quiz(me["id"], answers=[True], finished=True)
        client.post(f"/plan/sessions/{s['id']}/link", json={"quiz_session_id": str(quiz_id)})
    client.get("/plan")                                  # warm: the syllabus cached
    count = {"n": 0}

    def tally(*_):
        count["n"] += 1
    event.listen(engine, "before_cursor_execute", tally)
    try:
        client.get("/plan")
    finally:
        event.remove(engine, "before_cursor_execute", tally)
    assert len(sessions) >= 3 and count["n"] <= 10
