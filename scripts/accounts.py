"""Make and manage student accounts. There is no sign-up page: an adult
makes every account here and hands over the email and starting password.

    python -m scripts.accounts list
    python -m scripts.accounts claim local --email you@example.com --name "Ibrahim"
    python -m scripts.accounts claim test  --email test@eduos.local
    python -m scripts.accounts add   --email asha@example.com --name "Asha"
    python -m scripts.accounts reset --email asha@example.com
    python -m scripts.accounts delete --email asha@example.com

`claim` gives one of the two stand-in students a login: "local" holds every
answer, plan and upload from before accounts, so claiming it keeps all of
that; "test" is the account for click-through testing.

A new or reset password is made up and printed once; it is not stored
anywhere readable. Add --choose to type one instead (it is not shown).
Resetting a password also logs that student out on every device.

`delete` removes a student and everything that is theirs: answers, quiz
history, mastery, plan, chats, uploads (the files too) and logins. It asks
you to type the email again first, and there is no undo. The two stand-ins
("local" and "test") cannot be deleted: scripts and tests rely on them.
"""

from __future__ import annotations

import argparse
import getpass
import sys
import uuid

from sqlalchemy import select

from pathlib import Path

from apps.api import auth
from apps.api.db import init_db, session_scope
from apps.api.models import Conversation, Document
from apps.api.quiz import Attempt
from apps.api.student import STUDENTS, Student


def _password(choose: bool) -> str:
    if not choose:
        return auth.new_password()
    first = getpass.getpass("New password (not shown): ")
    auth.check_new_password(first)
    if getpass.getpass("Same again: ") != first:
        raise ValueError("The two passwords are different.")
    return first


def _email(raw: str) -> str:
    return auth.check_email(raw)


def _free(db, email: str, keep=None) -> None:
    owner = db.scalar(select(Student).where(Student.email == email))
    if owner is not None and owner.id != keep:
        raise ValueError(f"{email} is already the login of {owner.display_name}.")


def _print_login(student: Student, password: str, chosen: bool) -> None:
    print(f"\n  Name:     {student.display_name}")
    print(f"  Email:    {student.email}")
    if not chosen:
        print(f"  Password: {password}")
        print("\n  Write the password down now: it is not shown again.")


def cmd_list(db, _args) -> None:
    rows = db.scalars(select(Student).order_by(Student.created_at)).all()
    for s in rows:
        stand_in = next((k for k, (sid, _) in STUDENTS.items() if sid == s.id), "")
        login = s.email or "(no login yet)"
        last = s.last_login_at.strftime("%d %b %Y %H:%M") if s.last_login_at else "never"
        tag = f"  [{stand_in}]" if stand_in else ""
        print(f"  {s.display_name:<30} {login:<35} last login: {last}{tag}")


def cmd_claim(db, args) -> None:
    student = db.get(Student, STUDENTS[args.who][0])
    if student is None:
        raise ValueError("That student is missing; start the server once first.")
    if student.email:
        raise ValueError(f"It already has a login ({student.email}). "
                         "Use `reset` for a new password.")
    email = _email(args.email)
    _free(db, email)
    password = _password(args.choose)
    student.email, student.password_hash = email, auth.hash_password(password)
    if args.name:
        student.display_name = args.name.strip()[:200]
    db.flush()
    _print_login(student, password, args.choose)


def cmd_add(db, args) -> None:
    email = _email(args.email)
    _free(db, email)
    name = args.name.strip()[:200]
    if not name:
        raise ValueError("Give the student a name with --name.")
    password = _password(args.choose)
    student = Student(id=uuid.uuid4(), display_name=name,
                      email=email, password_hash=auth.hash_password(password))
    db.add(student)
    db.flush()
    _print_login(student, password, args.choose)


def cmd_reset(db, args) -> None:
    email = _email(args.email)
    student = db.scalar(select(Student).where(Student.email == email))
    if student is None:
        raise ValueError(f"No account has the email {email}.")
    password = _password(args.choose)
    student.password_hash = auth.hash_password(password)
    db.flush()
    db.commit()
    logged_out = auth.end_all_sessions(db, student.id)
    _print_login(student, password, args.choose)
    print(f"  Logged out of {logged_out} device{'s' if logged_out != 1 else ''}.")


def cmd_delete(db, args, confirm=input) -> None:
    from sqlalchemy import func

    from apps.api.planner import StudyPlan
    from apps.api.services.figure_service import cache_dir

    email = _email(args.email)
    student = db.scalar(select(Student).where(Student.email == email))
    if student is None:
        raise ValueError(f"No account has the email {email}.")
    if any(sid == student.id for sid, _ in STUDENTS.values()):
        raise ValueError(f"{student.display_name} is a stand-in student and is kept. "
                         "Use `reset` to change its password instead.")

    uploads = db.scalars(select(Document).where(Document.student_id == student.id)).all()
    count = lambda model: db.scalar(  # noqa: E731
        select(func.count()).select_from(model).where(model.student_id == student.id))
    print(f"\n  This deletes {student.display_name} ({email}) and everything that is theirs:")
    print(f"    {count(Attempt)} answers, {count(StudyPlan)} plan, "
          f"{count(Conversation)} chats, {len(uploads)} uploads")
    print("  There is no undo.")
    if confirm("  Type the email again to delete: ").strip().lower() != email:
        raise ValueError("The email did not match, so nothing was deleted.")

    files = [Path(d.storage_path) for d in uploads]
    doc_ids = [str(d.id) for d in uploads]
    name = student.display_name
    db.delete(student)  # the database removes everything linked to it
    db.flush()
    db.commit()
    # Then their files: the uploads, and any figures cut from them.
    for path in files:
        path.unlink(missing_ok=True)
    figures = cache_dir()
    if doc_ids and figures.exists():
        for picture in figures.iterdir():
            if picture.name.split("_", 1)[0] in doc_ids:
                picture.unlink(missing_ok=True)
    auth.clear_cache()
    print(f"\n  Deleted {name}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.accounts",
                                     description="Make and manage student accounts.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="every student and their login")

    p = sub.add_parser("claim", help="give a stand-in student (local or test) a login")
    p.add_argument("who", choices=sorted(STUDENTS))
    p.add_argument("--email", required=True)
    p.add_argument("--name")
    p.add_argument("--choose", action="store_true", help="type the password yourself")

    p = sub.add_parser("add", help="make a new student account")
    p.add_argument("--email", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--choose", action="store_true", help="type the password yourself")

    p = sub.add_parser("reset", help="new password, and log out everywhere")
    p.add_argument("--email", required=True)
    p.add_argument("--choose", action="store_true", help="type the password yourself")

    p = sub.add_parser("delete", help="remove a student and everything that is theirs")
    p.add_argument("--email", required=True)

    args = parser.parse_args(argv)
    init_db()  # the login columns exist even if the server has not run yet
    commands = {"list": cmd_list, "claim": cmd_claim, "add": cmd_add, "reset": cmd_reset,
                "delete": cmd_delete}
    try:
        with session_scope() as db:
            commands[args.command](db, args)
    except ValueError as exc:
        print(f"  Not done: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
