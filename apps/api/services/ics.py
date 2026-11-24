"""The study plan as a calendar file (.ics), for Google, Apple or Outlook
calendars.

Milestone 6, the way chosen on 24 Nov: EduOS never writes into a student's
calendar. It publishes a file their calendar app imports or subscribes to,
so nothing of theirs can be changed or deleted by a mistake here -- the
risk the roadmap names ("a deleted exam reminder is not recoverable").

What goes in: every exam, and every day with study sessions, as all-day
events marked "free", so a study day never shows as a clash with a lesson
or a match. The plan only knows minutes per day, not clock times, which is
why the events are all-day.

Each event's UID is fixed by its date and plan, so a calendar that fetches
the file again updates the day instead of adding a copy.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone


def _escape(text: str) -> str:
    return (text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
            .replace("\r\n", "\\n").replace("\n", "\\n"))


def _fold(line: str) -> list[str]:
    """Lines may be at most 75 bytes; longer ones continue after a space."""
    out, raw = [], line.encode("utf-8")
    while len(raw) > 75:
        cut = 75 if not out else 74
        # Never split inside a multi-byte character.
        while cut > 0 and (raw[cut] & 0xC0) == 0x80:
            cut -= 1
        out.append(raw[:cut].decode("utf-8"))
        raw = raw[cut:]
    out.append(raw.decode("utf-8"))
    return [out[0]] + [" " + part for part in out[1:]]


def _day(iso: str) -> str:
    return iso.replace("-", "")


def _event(uid: str, day: str, summary: str, description: str, stamp: str) -> list[str]:
    start = date.fromisoformat(day)
    return [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{stamp}",
        f"DTSTART;VALUE=DATE:{_day(day)}",
        f"DTEND;VALUE=DATE:{(start + timedelta(days=1)).strftime('%Y%m%d')}",
        f"SUMMARY:{_escape(summary)}",
        f"DESCRIPTION:{_escape(description)}",
        "TRANSP:TRANSPARENT",
        "END:VEVENT",
    ]


def _sessions_text(sessions: list[dict]) -> tuple[str, str]:
    minutes = sum(s.get("minutes") or 0 for s in sessions)
    tests = [s for s in sessions if s.get("kind") == "test"]
    quizzes = len(sessions) - len(tests)
    parts = []
    if tests:
        parts.append(tests[0]["label"] if len(tests) == 1 else f"{len(tests)} tests")
    if quizzes:
        parts.append(f"{quizzes} quiz{'zes' if quizzes != 1 else ''}")
    summary = f"EduOS: {', '.join(parts)} ({minutes} min)"
    lines = [f"- {'' if s.get('kind') == 'test' else 'Quiz: '}{s['label']} ({s['minutes']} min)"
             for s in sessions]
    return summary, "\n".join(lines + ["", "Open EduOS to start. The plan changes as your "
                                       "quiz results come in."])


def build(view: dict, plan_id: str, now: datetime | None = None) -> str:
    """The calendar file for a plan, from plan_service.view's output."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//EduOS//Study plan//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "X-WR-CALNAME:EduOS study plan",
        # How often a subscribed calendar should fetch again. Google decides
        # for itself (often every 12-24 hours); Apple and Outlook follow it.
        "REFRESH-INTERVAL;VALUE=DURATION:PT6H",
        "X-PUBLISHED-TTL:PT6H",
    ]
    settings = view.get("settings") or {}
    for exam in settings.get("exams", []):
        lines += _event(f"exam-{exam['id']}@eduos", exam["date"], f"Exam: {exam['name']}",
                        f"{exam.get('topics', 0)} topics in your EduOS plan.", stamp)

    days = []
    if view.get("today") and view.get("today_sessions"):
        days.append({"date": view["today"],
                     "sessions": [s for s in view["today_sessions"] if s.get("status") != "done"]})
    days += view.get("calendar", [])
    for d in days:
        if not d["sessions"]:
            continue
        summary, description = _sessions_text(d["sessions"])
        lines += _event(f"study-{d['date']}-{plan_id}@eduos", d["date"], summary, description,
                        stamp)
    lines.append("END:VCALENDAR")
    return "\r\n".join(folded for line in lines for folded in _fold(line)) + "\r\n"
