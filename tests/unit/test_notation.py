"""Formula-aware retrieval.

The failure these guard against: asking "formula for coulomb force" and
getting the definition instead, because word-overlap ranking scores a line of
algebra at zero while scoring a paragraph that merely mentions formulas at the
top. The formula lost to prose about the formula by roughly sixteen to one.
"""

from __future__ import annotations

import uuid

import pytest

from ai.ingestion.notation import NOTATION_THRESHOLD, notation_score, profile
from ai.retrieval.hybrid import RetrievedChunk
from ai.retrieval.intent import detect, strip_form_words
from ai.retrieval.rerank import LexicalOverlapReranker

# --- detecting notation in passages --------------------------------------

FORMULA = "F = k q1 q2 / r^2\n\nwhere k = 9 x 10^9 N m^2 / C^2"
UNICODE_FORMULA = "F = kq₁q₂/r²"
REACTION = "6CO2 + 6H2O → C6H12O6 + 6O2"
PROSE = (
    "The Coulomb force is the electrostatic force of attraction or repulsion "
    "between two charged particles acting along the line joining them."
)
PROSE_ABOUT_FORMULAS = (
    "The formula for the Coulomb force is one of the most important equations "
    "in electrostatics and you should memorise it carefully."
)


@pytest.mark.parametrize("text", [FORMULA, UNICODE_FORMULA, REACTION])
def test_notation_is_detected(text):
    assert notation_score(text) >= NOTATION_THRESHOLD


@pytest.mark.parametrize("text", [PROSE, PROSE_ABOUT_FORMULAS])
def test_prose_scores_zero(text):
    """Talking about formulas is not the same as containing one."""
    assert notation_score(text) == 0.0


def test_incidental_symbols_stay_below_the_threshold():
    """A degree sign or a stray Greek letter is not a formula."""
    assert notation_score("The temperature rose to 30° overnight.") < NOTATION_THRESHOLD
    assert notation_score("The angle θ is measured from the vertical.") < NOTATION_THRESHOLD


def test_worked_example_counts_even_when_mostly_prose():
    text = (
        "The Coulomb force is given by\n\nF = kq1q2/r^2\n\nwhere k is the "
        "electrostatic constant. Substituting the values gives 9 N."
    )
    assert notation_score(text) >= NOTATION_THRESHOLD
    assert profile(text).has_formula


def test_equation_requires_terms_on_both_sides():
    assert "equation" not in profile("Let x = the unknown quantity here").signals


def test_empty_text_is_safe():
    assert notation_score("") == 0.0


# --- detecting notation requests in questions ----------------------------

@pytest.mark.parametrize(
    "question",
    [
        "formula for coulomb force",
        "what is the formula for coulomb force",
        "derive the expression for electric field",
        "balanced equation for photosynthesis",
        "calculate the force between two charges",
    ],
)
def test_notation_requests_are_recognised(question):
    assert detect(question).wants_notation


@pytest.mark.parametrize(
    "question",
    [
        "explain coulomb force",
        "difference between mitosis and meiosis",
        "why do charges repel",
    ],
)
def test_explanation_requests_are_not_notation_requests(question):
    assert not detect(question).wants_notation


def test_form_words_are_stripped_from_the_search_text():
    """"formula" matches prose about formulas, so it must not be searched for."""
    assert "formula" not in strip_form_words("formula for coulomb force").lower()
    assert "coulomb" in strip_form_words("formula for coulomb force").lower()


def test_stripping_never_empties_the_query():
    assert strip_form_words("give me the formula").strip()


# --- ranking -------------------------------------------------------------

def chunk(content: str, fused: float = 0.016) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="Physics notes",
        content=content,
        page_start=1,
        page_end=1,
        section=None,
        chunk_index=0,
        fused_score=fused,
        notation_score=notation_score(content),
    )


def test_formula_outranks_prose_when_a_formula_was_asked_for():
    """The original bug: this ordering was reversed."""
    pool = [chunk(PROSE), chunk(PROSE_ABOUT_FORMULAS), chunk(FORMULA)]
    ranked = LexicalOverlapReranker().rerank(
        "formula for coulomb force", pool, 3, wants_notation=True
    )
    assert ranked[0].notation_score >= NOTATION_THRESHOLD


def test_formula_with_surrounding_context_beats_a_bare_formula():
    """A worked example answers the question better than naked algebra."""
    bare = chunk(FORMULA)
    with_context = chunk(
        "The Coulomb force between two charges is given by\n\nF = k q1 q2 / r^2"
        "\n\nwhere k is the electrostatic constant."
    )
    ranked = LexicalOverlapReranker().rerank(
        "formula for coulomb force", [bare, with_context], 2, wants_notation=True
    )
    assert ranked[0] is with_context


def test_explanation_questions_still_prefer_prose():
    """The notation boost must not leak into ordinary questions."""
    definition = chunk(PROSE)
    formula = chunk(FORMULA)
    ranked = LexicalOverlapReranker().rerank(
        "explain coulomb force", [formula, definition], 2, wants_notation=False
    )
    assert ranked[0] is definition


def test_notation_boost_does_not_override_topic():
    """F = ma must not win an electrostatics question just for being a formula."""
    coulomb = chunk(
        "The Coulomb force between two charges is given by\n\nF = k q1 q2 / r^2"
    )
    newton = chunk("F = m a\n\nwhere m is mass and a is acceleration.")
    ranked = LexicalOverlapReranker().rerank(
        "formula for coulomb force", [newton, coulomb], 2, wants_notation=True
    )
    assert ranked[0] is coulomb
