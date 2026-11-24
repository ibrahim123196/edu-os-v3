"""Passwords and which pages need a login. No database."""

from __future__ import annotations

import pytest

from apps.api import auth
from apps.api.config import Settings


def test_a_password_checks_against_its_own_hash():
    h = auth.hash_password("pencil sharpener")
    assert auth.verify_password("pencil sharpener", h)
    assert not auth.verify_password("pencil sharpenet", h)


def test_the_same_password_hashes_differently_each_time():
    assert auth.hash_password("abc12345") != auth.hash_password("abc12345")


def test_the_password_is_not_in_its_hash():
    assert "pencil" not in auth.hash_password("pencil sharpener")


@pytest.mark.parametrize("stored", [None, "", "garbage", "bcrypt$1$2$3$4$5", "scrypt$x$y$z$a$b"])
def test_a_broken_or_missing_hash_never_matches(stored):
    assert not auth.verify_password("anything", stored)


def test_made_up_passwords_are_long_and_readable():
    p = auth.new_password()
    assert len(p) == 12 and not set(p) & set("01lio")


def test_short_passwords_are_refused():
    with pytest.raises(ValueError):
        auth.check_new_password("short")
    auth.check_new_password("long enough")


def test_emails_are_compared_lower_case():
    assert auth.normalise_email("  Asha@Example.COM ") == "asha@example.com"


def test_only_the_page_login_and_health_are_public():
    for path in ("/", "/health", "/auth/login", "/auth/me", "/static/x.js"):
        assert auth.is_public(path, "production")
    for path in ("/progress", "/plan", "/chat/ask", "/documents", "/quiz", "/figures/a/b.png",
                 "/docs"):
        assert not auth.is_public(path, "production")
    assert auth.is_public("/docs", "local")


def test_login_cannot_be_switched_off_outside_a_local_install():
    assert Settings(require_login=False, environment="local").login_required() is False
    assert Settings(require_login=False, environment="production").login_required() is True
    assert Settings(require_login=False, environment="staging").login_required() is True
    assert Settings.model_fields["require_login"].default is True


@pytest.mark.parametrize("raw", ["asha@example.com", " Asha@Example.com ", "test@eduos.local"])
def test_emails_that_look_right(raw):
    assert auth.check_email(raw) == raw.strip().lower()


@pytest.mark.parametrize("raw", ["", "email", "a@", "@b.com", "a@b", "a b@c.com", "a@.com"])
def test_emails_that_do_not(raw):
    with pytest.raises(ValueError):
        auth.check_email(raw)


def test_sign_up_is_public_but_nothing_else_new_is():
    assert auth.is_public("/auth/signup", "production")
