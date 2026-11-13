"""Estimating what a student knows.

Three things make correct-divided-by-total unusable, and each is tested
here: it is overconfident on small samples, it ignores recency, and it
reports a number without saying how much evidence stands behind it.
"""

from __future__ import annotations

from ai.mastery.estimate import (
    WEAK_THRESHOLD,
    MasteryEstimate,
    diagnose,
    estimate,
    practice_weight,
)


def test_no_attempts_sits_at_the_prior():
    est = estimate([])
    assert est.mastery == 0.5
    assert est.confidence == 0.0
    assert not est.is_reliable


def test_one_correct_answer_is_not_full_mastery():
    """Raw proportion would say 100%, and a planner believing it would skip
    a topic the student has barely touched."""
    est = estimate([True])
    assert 0.5 < est.mastery < 0.7
    assert not est.is_reliable


def test_one_wrong_answer_is_not_zero_mastery():
    est = estimate([False])
    assert 0.3 < est.mastery < 0.5


def test_sustained_success_raises_mastery():
    assert estimate([True] * 20).mastery > estimate([True] * 5).mastery


def test_recent_answers_count_for_more_than_old_ones():
    """A student who failed in September and has passed five times since is
    not at 50%."""
    improving = estimate([False] + [True] * 5)
    declining = estimate([True] * 5 + [False])
    assert improving.mastery > declining.mastery


def test_confidence_grows_with_the_number_of_attempts():
    """Recency decides what to believe; sample size decides how sure to be.
    Deriving confidence from the recency-weighted total conflated the two
    and capped it at 0.57 however many questions were answered, because the
    decay sum converges -- so "reliable" was unreachable."""
    assert estimate([True] * 40).confidence > estimate([True] * 10).confidence
    assert estimate([True] * 40).confidence > 0.8


def test_five_attempts_is_enough_to_act_on():
    assert estimate([True] * 5).is_reliable
    assert not estimate([True] * 2).is_reliable


def test_weakness_is_recognised():
    assert estimate([False] * 6).is_weak
    assert not estimate([True] * 10).is_weak


def test_the_estimate_saturates_rather_than_reaching_the_extremes():
    """The prior keeps it off 0 and 1 however one-sided the record, so a
    student is never written off and never declared finished."""
    assert estimate([False] * 50).mastery > 0.15
    assert estimate([True] * 50).mastery < 0.9


def test_describe_admits_a_thin_sample():
    assert "only" in estimate([True]).describe()
    assert "only" not in estimate([True] * 10).describe()
    assert estimate([]).describe() == "not attempted"


# --- what to practise next -----------------------------------------------

def test_weak_concepts_are_favoured_over_strong_ones():
    assert practice_weight(estimate([False] * 8)) > practice_weight(
        estimate([True] * 8)
    )


def test_an_untouched_concept_is_favoured_over_a_strong_one():
    """Finding out is nearly as valuable as remediating, and a topic never
    seen can never be shown to be weak."""
    assert practice_weight(estimate([])) > practice_weight(estimate([True] * 10))


def test_a_mastered_concept_still_gets_some_weight():
    """A strict weakest-first ordering hands the student the same quiz every
    time and stops them revisiting what they have learnt."""
    assert practice_weight(estimate([True] * 20)) > 0


# --- diagnosis ------------------------------------------------------------

def strong() -> MasteryEstimate:
    return estimate([True] * 10)


def shaky() -> MasteryEstimate:
    """Struggling but not hopeless: roughly half right.

    Diagnosis needs headroom. The estimate saturates near the prior bounds,
    so a concept the student gets uniformly wrong sits at about 0.22 and no
    prerequisite can be a tenth lower. A cause can only be named for a
    concept that is weak rather than bottomed out -- which is also when
    naming one is useful.
    """
    return estimate([False] * 4 + [True] * 4)


def test_a_clearly_weaker_prerequisite_is_blamed():
    """Telling a student they are weak on Gauss's law sends them to reread
    Gauss's law, which will not help if the gap is electric field."""
    finding = diagnose("Gauss's law", shaky(), {"Electric field": estimate([False] * 10)})
    assert finding.blamed_prerequisite == "Electric field"
    assert "Likely cause" in finding.explanation


def test_a_strong_prerequisite_is_not_blamed():
    finding = diagnose("Gauss's law", shaky(), {"Electric field": strong()})
    assert finding.blamed_prerequisite is None


def test_an_unreliable_prerequisite_is_not_blamed():
    """Blaming the wrong topic sends the student somewhere useless, which is
    worse than saying nothing."""
    finding = diagnose("Gauss's law", shaky(), {"Electric field": estimate([False])})
    assert finding.blamed_prerequisite is None


def test_the_weakest_prerequisite_is_chosen():
    finding = diagnose(
        "Gauss's law",
        shaky(),
        {
            "Electric flux": estimate([False] * 5 + [True] * 2),
            "Electric field": estimate([False] * 10),
        },
    )
    assert finding.blamed_prerequisite == "Electric field"


def test_a_prerequisite_barely_weaker_is_not_blamed():
    """Lower by noise is not a cause."""
    concept = estimate([False] * 5 + [True] * 3)
    nearly_the_same = estimate([False] * 5 + [True] * 3)
    finding = diagnose("Gauss's law", concept, {"Electric field": nearly_the_same})
    assert finding.blamed_prerequisite is None
