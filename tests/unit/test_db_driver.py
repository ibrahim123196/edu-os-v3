"""Choosing the database driver when Windows blocks psycopg.

Smart App Control started blocking psycopg's compiled library overnight,
with nothing in the project changed. These pin the fallback to pg8000 and
the one thing that would otherwise break with it: the secure-connection
settings in a Neon address.
"""

from __future__ import annotations

import ssl

import pytest

from apps.api import db

NEON = (
    "postgresql+psycopg://user:secret@ep-cool-name.aws.neon.tech/neondb"
    "?sslmode=require&channel_binding=require"
)


@pytest.fixture
def psycopg_blocked(monkeypatch):
    monkeypatch.setattr(db, "_psycopg_loads", lambda: False)


@pytest.fixture
def psycopg_works(monkeypatch):
    monkeypatch.setattr(db, "_psycopg_loads", lambda: True)


def test_psycopg_is_kept_when_it_loads(psycopg_works):
    url, args = db.engine_url_and_args(NEON)
    assert url.drivername == "postgresql+psycopg"
    assert url.query["sslmode"] == "require"  # libpq handles it itself
    assert args == {}


def test_blocked_psycopg_switches_to_pg8000(psycopg_blocked):
    url, _ = db.engine_url_and_args(NEON)
    assert url.drivername == "postgresql+pg8000"
    # Everything else about the address is untouched.
    assert (url.host, url.database, url.username, url.password) == (
        "ep-cool-name.aws.neon.tech", "neondb", "user", "secret",
    )


def test_pg8000_gets_a_verified_tls_connection_instead_of_sslmode(psycopg_blocked):
    """pg8000 fails on sslmode=... in the URL. Dropping it without adding
    TLS would send the password to Neon unencrypted -- or not connect."""
    url, args = db.engine_url_and_args(NEON)
    assert "sslmode" not in url.query
    assert "channel_binding" not in url.query
    context = args["ssl_context"]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


def test_a_local_database_without_ssl_gets_no_tls(psycopg_blocked):
    url, args = db.engine_url_and_args(
        "postgresql+psycopg://eduos:eduos@localhost:5432/eduos"
    )
    assert url.drivername == "postgresql+pg8000"
    assert args == {}


def test_sslmode_disable_means_no_tls(psycopg_blocked):
    _, args = db.engine_url_and_args(
        "postgresql+psycopg://eduos:eduos@localhost/eduos?sslmode=disable"
    )
    assert args == {}


def test_an_address_naming_pg8000_directly_is_handled_too(psycopg_works):
    url, args = db.engine_url_and_args(NEON.replace("+psycopg", "+pg8000"))
    assert url.drivername == "postgresql+pg8000"
    assert "ssl_context" in args
