"""Which rejection reasons mean a question's answer key cannot be trusted.

Answers to a faulty question stop counting toward mastery. Repeats and
misfiled questions were answerable, so their answers must keep counting.
"""

from __future__ import annotations

import pytest

from apps.api.services.flag_service import reason_means_faulty


@pytest.mark.parametrize("reason", [
    "faulty: Says doubling the distance quarters the potential",
    "faulty (flagged): The answer key is wrong",
    "check: key says option 1, checker says 3: For equal ΔV ...",
    "check: 2 options are defensible (1, 3)",
    "check: false premise: it splits equally",
    "check: no option is correct",
])
def test_an_untrustworthy_key_is_faulty(reason):
    assert reason_means_faulty(reason)


@pytest.mark.parametrize("reason", [
    "check: repeats: A parallel-plate capacitor with plate area A",
    "check: wrong concept: this is about drift speed",
    "copy of: Why does the field inside ...",
    "check: reply could not be read: '...'",   # unknown, not proven faulty
    "options 2 and 4 say the same thing",      # caught before anyone saw it
    None,
    "",
])
def test_an_answerable_question_is_not_faulty(reason):
    assert not reason_means_faulty(reason)
