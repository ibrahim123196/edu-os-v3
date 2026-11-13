"""Estimate what a student knows from what they have answered.

Three things make the naive answer -- correct divided by total -- unusable.

A student who has answered one question correctly is not at 100%. Raw
proportion is wildly overconfident on small samples, and a planner that
believes it will skip a topic the student has barely touched. A prior fixes
this: two imaginary correct answers and two imaginary wrong ones are added,
so one correct answer out of one reads as 0.6 rather than 1.0, and only
sustained performance moves the number near either end.

A student who failed Gauss's law in September and has passed it five times
since is not 50%. Recent attempts say more about current understanding than
old ones, so each older attempt counts a little less.

And a number without its sample size is misleading. "42% mastery" from two
attempts and from forty attempts are different claims, so confidence is
reported alongside, and callers are expected to respect it: acting on an
estimate drawn from three answers is how a study plan ends up built on
noise.

The estimator is deliberately simple and explainable. Item response theory
would be better calibrated, and is not worth it before there is enough
attempt data to calibrate against -- which there is not, for anybody, yet.
"""

from __future__ import annotations

from dataclasses import dataclass

# Imaginary prior attempts, half right and half wrong. Larger values make
# the estimate slower to move and more reluctant to claim certainty.
PRIOR_CORRECT = 2.0
PRIOR_TOTAL = 4.0

# How much each older attempt counts relative to the one after it.
# At 0.85 an answer ten attempts ago carries about a fifth of the weight of
# the most recent one.
RECENCY_DECAY = 0.85

# Sets how quickly confidence grows with answered questions: confidence
# reaches 0.5 at this many attempts, 0.67 at twice as many.
CONFIDENCE_SCALE = 5.0

# Below this a concept is treated as weak enough to practise.
WEAK_THRESHOLD = 0.6

# A prerequisite must be this much weaker than the concept itself before it
# is named as the likely cause, rather than being merely lower by noise.
PREREQUISITE_MARGIN = 0.1


@dataclass(frozen=True)
class MasteryEstimate:
    mastery: float
    confidence: float
    attempts: int
    correct: int

    @property
    def is_weak(self) -> bool:
        return self.mastery < WEAK_THRESHOLD

    @property
    def is_reliable(self) -> bool:
        """Whether this is worth acting on.

        An estimate from two attempts is a guess wearing a decimal point.
        """
        return self.confidence >= 0.5

    def describe(self) -> str:
        if self.attempts == 0:
            return "not attempted"
        percent = round(self.mastery * 100)
        if not self.is_reliable:
            return f"{percent}% from only {self.attempts} question(s)"
        return f"{percent}% over {self.attempts} questions"


def estimate(outcomes: list[bool]) -> MasteryEstimate:
    """Estimate mastery from attempts, most recent last.

    Order matters: the last element is treated as the newest answer and
    carries the most weight.
    """
    attempts = len(outcomes)
    correct = sum(1 for o in outcomes if o)

    if not outcomes:
        return MasteryEstimate(
            mastery=PRIOR_CORRECT / PRIOR_TOTAL,
            confidence=0.0,
            attempts=0,
            correct=0,
        )

    # Newest first, so the most recent attempt gets weight 1.
    weighted_correct = 0.0
    weighted_total = 0.0
    for age, outcome in enumerate(reversed(outcomes)):
        weight = RECENCY_DECAY**age
        weighted_total += weight
        if outcome:
            weighted_correct += weight

    mastery = (weighted_correct + PRIOR_CORRECT) / (weighted_total + PRIOR_TOTAL)

    # Confidence counts attempts, not weighted attempts. Recency decides
    # what to believe; sample size decides how sure to be, and the two are
    # different questions. Using the weighted total conflated them and
    # capped confidence at 0.57 however many questions were answered,
    # because the decay sum converges -- so "reliable" was unreachable.
    confidence = attempts / (attempts + CONFIDENCE_SCALE)

    return MasteryEstimate(
        mastery=round(mastery, 4),
        confidence=round(confidence, 4),
        attempts=attempts,
        correct=correct,
    )


@dataclass(frozen=True)
class Diagnosis:
    """Why a concept is weak, when the cause is something earlier."""

    concept: str
    mastery: float
    blamed_prerequisite: str | None = None
    prerequisite_mastery: float | None = None

    @property
    def explanation(self) -> str:
        percent = round(self.mastery * 100)
        if not self.blamed_prerequisite:
            return f"{self.concept} — {percent}%"
        cause = round((self.prerequisite_mastery or 0) * 100)
        return (
            f"{self.concept} — {percent}%. Likely cause: "
            f"{self.blamed_prerequisite} at {cause}%."
        )


def diagnose(
    concept: str,
    concept_estimate: MasteryEstimate,
    prerequisites: dict[str, MasteryEstimate],
) -> Diagnosis:
    """Name the weakest prerequisite, when one is clearly weaker.

    This is the point of storing prerequisites at all. "You are weak on
    Gauss's law" sends a student to reread Gauss's law, which will not help
    if the actual gap is electric field. Telling them where to go instead is
    the difference between a score and a diagnosis.

    A prerequisite is only blamed when it is both weaker by a clear margin
    and backed by enough attempts to believe. Blaming the wrong topic sends
    the student somewhere useless and is worse than saying nothing.
    """
    weakest: str | None = None
    weakest_mastery: float | None = None

    for name, prerequisite in prerequisites.items():
        if not prerequisite.is_reliable:
            continue
        if prerequisite.mastery > concept_estimate.mastery - PREREQUISITE_MARGIN:
            continue
        if weakest_mastery is None or prerequisite.mastery < weakest_mastery:
            weakest, weakest_mastery = name, prerequisite.mastery

    return Diagnosis(
        concept=concept,
        mastery=concept_estimate.mastery,
        blamed_prerequisite=weakest,
        prerequisite_mastery=weakest_mastery,
    )


def practice_weight(est: MasteryEstimate) -> float:
    """How strongly a concept should be favoured when picking questions.

    Weakness earns practice, but so does not knowing. A concept with no
    attempts is ranked just above a weak one: finding out is nearly as
    valuable as remediating, and a student who never sees a topic can never
    be shown to be weak at it.

    Returning a weight rather than sorting keeps selection probabilistic. A
    strict "weakest first" ordering gives the same quiz every time and
    stops the student ever revisiting something they have learnt.
    """
    if est.attempts == 0:
        return 2.5
    # Mastery 0 -> 3.0, mastery 1 -> 0.5.
    weight = 0.5 + (1.0 - est.mastery) * 2.5
    # A shaky estimate is worth testing regardless of what it currently says.
    if not est.is_reliable:
        weight = max(weight, 1.5)
    return round(weight, 3)
