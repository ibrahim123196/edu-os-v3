"""The calendar file, from a plan view made up here."""

from __future__ import annotations

from datetime import datetime, timezone

from apps.api.services import ics

VIEW = {
    "settings": {"exams": [{"id": "e1", "name": "Physics, mid-term; Class 12",
                            "date": "2026-12-15", "topics": 12}]},
    "today": "2026-11-24",
    "today_sessions": [
        {"kind": "quiz", "label": "Coulomb's law", "minutes": 15, "status": "done"},
        {"kind": "quiz", "label": "Electric field", "minutes": 15, "status": "todo"},
    ],
    "calendar": [
        {"date": "2026-11-25", "sessions": [
            {"kind": "quiz", "label": "Capacitance", "minutes": 20},
            {"kind": "quiz", "label": "Dielectrics", "minutes": 20}]},
        {"date": "2026-11-26", "sessions": []},
        {"date": "2026-12-10", "sessions": [
            {"kind": "test", "label": "Test: Physics (20 questions)", "minutes": 40}]},
    ],
}


def _build():
    return ics.build(VIEW, "plan-1", now=datetime(2026, 11, 24, tzinfo=timezone.utc))


def _events(body):
    return body.split("BEGIN:VEVENT")[1:]


def test_one_event_per_exam_and_study_day_and_none_for_empty_days():
    body = _build()
    events = _events(body)
    assert len(events) == 4    # exam, today, 25 Nov, 10 Dec; not 26 Nov
    assert "DTSTART;VALUE=DATE:20261126" not in body


def test_exams_and_days_are_all_day_and_never_busy():
    for event in _events(_build()):
        assert "DTSTART;VALUE=DATE:" in event and "DTEND;VALUE=DATE:" in event
        assert "TRANSP:TRANSPARENT" in event


def test_summaries_say_what_and_how_long():
    body = _build()
    assert "SUMMARY:EduOS: 2 quizzes (40 min)" in body
    assert "SUMMARY:EduOS: Test: Physics (20 questions) (40 min)" in body
    # Today: only what is left.
    assert "SUMMARY:EduOS: 1 quiz (15 min)" in body


def test_commas_and_semicolons_are_escaped():
    assert r"SUMMARY:Exam: Physics\, mid-term\; Class 12" in _build()


def test_lines_are_folded_and_end_in_crlf():
    view = dict(VIEW, calendar=[{"date": "2026-11-25", "sessions": [
        {"kind": "quiz", "label": "Électrostatique — ε₀ " * 8, "minutes": 20}]}])
    body = ics.build(view, "plan-1")
    lines = body.split("\r\n")
    assert all(len(line.encode()) <= 75 for line in lines)
    assert any(line.startswith(" ") for line in lines)        # a continuation
    unfolded = body.replace("\r\n ", "")
    assert "Électrostatique — ε₀ " * 3 in unfolded              # nothing split mid-character
    assert "\n" not in body.replace("\r\n", "")


def test_the_same_day_keeps_its_id_so_calendars_update_it():
    a, b = _build(), ics.build(VIEW, "plan-1")
    uid = "UID:study-2026-11-25-plan-1@eduos"
    assert uid in a and uid in b
