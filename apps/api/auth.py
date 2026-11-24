"""Accounts: passwords, login sessions, and the middleware that knows who
is asking.

Accounts are made by an adult (Ibrahim, or later a teacher) with
`python -m scripts.accounts`; there is no public sign-up. The users are
school students, and an adult involved in every account is the simplest
honest answer to consent for minors.

Passwords are hashed with scrypt from Python's own hashlib: no new compiled
library for Windows Smart App Control to block (it already blocks psycopg).

A login is a random token in an HttpOnly cookie. Only its SHA-256 is stored,
so a copy of the database does not let anyone log in as a student. Logging
out deletes the row; resetting a password deletes all of that student's
rows, so every device is logged out.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import DateTime, ForeignKey, String, delete, func, select
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, Session, mapped_column

from apps.api.db import Base

log = logging.getLogger(__name__)

COOKIE = "eduos_session"
SESSION_DAYS = 30
MIN_PASSWORD = 8


# --- Passwords ------------------------------------------------------------

# scrypt cost: about 50 ms and 16 MB per check. Slow enough that a stolen
# hash is expensive to guess at, fast enough that logging in is not.
_N, _R, _P = 2**14, 8, 1


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return f"scrypt${_N}${_R}${_P}${b64(salt)}${b64(digest)}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        kind, n, r, p, salt, digest = stored.split("$")
        if kind != "scrypt":
            return False
        expected = base64.b64decode(digest)
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt),
                             n=int(n), r=int(r), p=int(p), dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got, expected)


# Checked against when the email is unknown, so a wrong email takes as long
# as a wrong password and the timing does not reveal who has an account.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def new_password() -> str:
    """A starting password to hand over: 12 characters, no look-alikes."""
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
    return "".join(secrets.choice(alphabet) for _ in range(12))


def normalise_email(email: str) -> str:
    return email.strip().lower()


def check_email(raw: str) -> str:
    """The email, lower-cased, if it looks like one; ValueError if not.
    Only the shape is checked: nothing is sent to it yet."""
    email = normalise_email(raw)
    name, at, domain = email.partition("@")
    if not (name and at and "." in domain and " " not in email
            and not domain.startswith(".") and not domain.endswith(".")):
        raise ValueError("That does not look like an email address.")
    return email


def check_new_password(password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"The password must be at least {MIN_PASSWORD} characters.")


# --- Login sessions -------------------------------------------------------


class LoginSession(Base):
    """One logged-in device."""

    __tablename__ = "login_sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def start_session(db: Session, student_id: uuid.UUID) -> str:
    """A new login for this student; the token goes in the cookie."""
    token = secrets.token_urlsafe(32)
    db.add(LoginSession(token_hash=_token_hash(token), student_id=student_id,
                        expires_at=_now() + timedelta(days=SESSION_DAYS)))
    db.commit()
    return token


def end_session(db: Session, token: str) -> None:
    th = _token_hash(token)
    db.execute(delete(LoginSession).where(LoginSession.token_hash == th))
    db.commit()
    _cache_drop(th)


def end_all_sessions(db: Session, student_id: uuid.UUID) -> int:
    """Log a student out everywhere (after a password reset)."""
    rows = db.execute(delete(LoginSession).where(LoginSession.student_id == student_id))
    db.commit()
    with _cache_lock:
        for th in [k for k, (sid, _) in _cache.items() if sid == student_id]:
            _cache.pop(th, None)
    return rows.rowcount or 0


# Each request checks its cookie. Neon is a round trip away, so a token
# that was found is remembered for a minute. A logout on this server forgets
# it at once; a reset from the accounts script, a separate process, takes
# up to that minute to reach a server already running.
_CACHE_SECONDS = 60
_cache: dict[str, tuple[uuid.UUID, float]] = {}
_cache_lock = threading.Lock()


def _cache_drop(th: str) -> None:
    with _cache_lock:
        _cache.pop(th, None)


def student_for_token(db: Session, token: str | None) -> uuid.UUID | None:
    """Who this cookie belongs to, or None if it is missing, unknown or
    expired."""
    if not token:
        return None
    th = _token_hash(token)
    with _cache_lock:
        hit = _cache.get(th)
    if hit and hit[1] > time.monotonic():
        return hit[0]
    row = db.get(LoginSession, th)
    if row is None or row.expires_at <= _now():
        _cache_drop(th)
        return None
    with _cache_lock:
        _cache[th] = (row.student_id, time.monotonic() + _CACHE_SECONDS)
    return row.student_id


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


# --- Logging in -----------------------------------------------------------

# Five wrong passwords for one email in 15 minutes, then a wait. Kept in
# memory: a restart forgets it, which is acceptable for one server.
_FAIL_LIMIT = 5
_FAIL_WINDOW = 15 * 60
_failures: dict[str, list[float]] = {}
_fail_lock = threading.Lock()


class LoginError(Exception):
    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(message)
        self.status = status


def _recent_failures(email: str, now: float) -> list[float]:
    times = [t for t in _failures.get(email, []) if now - t < _FAIL_WINDOW]
    _failures[email] = times
    return times


def log_in(db: Session, email: str, password: str):
    """The student, if the email and password match; LoginError if not."""
    from apps.api.student import Student

    email = normalise_email(email)
    now = time.monotonic()
    with _fail_lock:
        if len(_recent_failures(email, now)) >= _FAIL_LIMIT:
            raise LoginError("Too many wrong tries. Wait 15 minutes, then try again.", 429)

    student = db.scalar(select(Student).where(Student.email == email)) if email else None
    ok = verify_password(password, student.password_hash if student else _DUMMY_HASH)
    if not (ok and student is not None):
        with _fail_lock:
            _failures.setdefault(email, []).append(now)
        raise LoginError("That email and password do not match.")

    with _fail_lock:
        _failures.pop(email, None)
    student.last_login_at = _now()
    db.commit()
    return student


def reset_failures() -> None:
    with _fail_lock:
        _failures.clear()
        _signups.clear()


# --- Signing up -----------------------------------------------------------

# At most 5 new accounts from one address an hour: enough for a family or
# a classroom sharing a connection, not enough to fill the database.
_SIGNUP_LIMIT = 5
_SIGNUP_WINDOW = 3600
_signups: dict[str, list[float]] = {}


def sign_up(db: Session, name: str, email: str, password: str, address: str = ""):
    """A new student who can log in; LoginError if not allowed."""
    import uuid as _uuid

    from apps.api.student import Student

    name = " ".join(name.split())[:200]
    if not name:
        raise LoginError("Please type your name.", 400)
    try:
        email = check_email(email)
        check_new_password(password)
    except ValueError as exc:
        raise LoginError(str(exc), 400) from exc

    now = time.monotonic()
    with _fail_lock:
        recent = [t for t in _signups.get(address, []) if now - t < _SIGNUP_WINDOW]
        if len(recent) >= _SIGNUP_LIMIT:
            raise LoginError("Too many new accounts from here. Try again in an hour.", 429)

    if db.scalar(select(Student).where(Student.email == email)) is not None:
        raise LoginError("An account already uses this email. Log in instead.", 409)
    student = Student(id=_uuid.uuid4(), display_name=name, email=email,
                      password_hash=hash_password(password), last_login_at=_now())
    db.add(student)
    db.commit()
    with _fail_lock:
        _signups[address] = recent + [now]
    return student


# --- Middleware -----------------------------------------------------------

# Reachable without logging in: the page itself, its files, logging in, and
# the health check.
_PUBLIC_EXACT = {"/", "/health", "/auth/login", "/auth/signup", "/auth/logout",
                 "/auth/me", "/favicon.ico"}
# /plan/feed/<secret>.ics: calendar apps fetch it without logging in; the
# secret in the address is the permission (routes/plan.py).
_PUBLIC_PREFIX = ("/static/", "/plan/feed/")
# FastAPI's own API pages: only on a local install.
_DOCS = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}


def is_public(path: str, environment: str) -> bool:
    if path in _PUBLIC_EXACT or path.startswith(_PUBLIC_PREFIX):
        return True
    return environment == "local" and path in _DOCS


def _cookie(scope, name: str) -> str | None:
    for key, value in scope.get("headers", []):
        if key == b"cookie":
            for part in value.decode("latin-1").split(";"):
                k, _, v = part.strip().partition("=")
                if k == name:
                    return v
    return None


class LoginMiddleware:
    """Puts the logged-in student into the request; refuses the rest.

    A plain ASGI middleware rather than Starlette's BaseHTTPMiddleware, so
    the student set here is seen by the route in the same context, sync
    routes included (they run in a thread that copies it).
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        from starlette.concurrency import run_in_threadpool
        from starlette.responses import JSONResponse

        from apps.api.config import get_settings
        from apps.api.db import SessionLocal
        from apps.api.student import _current

        settings = get_settings()
        token = _cookie(scope, COOKIE)

        def lookup():
            with SessionLocal() as db:
                return student_for_token(db, token)

        student_id = await run_in_threadpool(lookup) if token else None
        path = scope.get("path", "")
        if student_id is None and not is_public(path, settings.environment) \
                and settings.login_required():
            response = JSONResponse({"detail": "Please log in.", "login": True},
                                    status_code=401)
            return await response(scope, receive, send)

        reset = _current.set(student_id)
        try:
            await self.app(scope, receive, send)
        finally:
            _current.reset(reset)
