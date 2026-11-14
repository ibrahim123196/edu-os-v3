"""Keeping test clicks off the real student.

Twenty-one guesses once landed on the real student and the mastery model
believed every one. These tests pin the two things that prevent a repeat:
the setting decides the student, and nothing falls back to the real one.
"""

from __future__ import annotations

import inspect
import uuid
from pathlib import Path

import pytest

from apps.api import config
from apps.api.services.quiz_service import grade
from apps.api.student import (
    LOCAL_STUDENT_ID,
    STUDENTS,
    TEST_STUDENT_ID,
    active_student_id,
)


@pytest.fixture
def setting(monkeypatch):
    """Set ACTIVE_STUDENT for one test, with the settings cache cleared
    either side so no other test sees it."""

    def _set(value: str) -> None:
        monkeypatch.setenv("ACTIVE_STUDENT", value)
        config.get_settings.cache_clear()

    yield _set
    config.get_settings.cache_clear()


def test_the_two_students_are_different():
    assert LOCAL_STUDENT_ID != TEST_STUDENT_ID


def test_the_setting_picks_the_test_student(setting):
    setting("test")
    assert active_student_id() == TEST_STUDENT_ID


def test_the_setting_picks_the_real_student(setting):
    setting("local")
    assert active_student_id() == LOCAL_STUDENT_ID


def test_a_misspelt_setting_fails_rather_than_guessing(setting):
    """"tset" must not silently become either student."""
    with pytest.raises(Exception):
        setting("tset")
        active_student_id()


def test_every_setting_value_has_a_student():
    """A value config accepts but STUDENTS lacks would crash on the first
    answer, not at startup."""
    allowed = set(config.Settings.model_fields["active_student"].annotation.__args__)
    assert allowed == set(STUDENTS)


def test_grade_has_no_default_student():
    """The fallback to the real student is how the guesses got there."""
    param = inspect.signature(grade).parameters["student_id"]
    assert param.default is inspect.Parameter.empty
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_no_route_names_the_real_student_directly():
    """Routes ask active_student_id(). One that hard-codes the real student
    would bypass the setting without anything else noticing."""
    routes = Path(__file__).parents[2] / "apps" / "api" / "routes"
    offenders = [
        p.name for p in routes.glob("*.py") if "LOCAL_STUDENT_ID" in p.read_text()
    ]
    assert offenders == []


def test_student_ids_are_fixed():
    """Changing these UUIDs would orphan every stored attempt."""
    assert LOCAL_STUDENT_ID == uuid.UUID("00000000-0000-0000-0000-000000000001")
    assert TEST_STUDENT_ID == uuid.UUID("00000000-0000-0000-0000-000000000002")
