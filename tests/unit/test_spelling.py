"""Typo correction.

The database is stubbed here so the logic can be tested without Postgres. The
real trigram matching is covered in tests/integration.

The risk being guarded against is over-correction: swapping a word the student
meant for a different word that happens to look similar. In a biology context
turning "mitosis" into "meiosis" would be worse than returning nothing, so the
tests below pin both directions -- fix real typos, leave everything else alone.
"""

from __future__ import annotations

import pytest

from ai.retrieval.spelling import (
    MIN_LENGTH,
    _candidates,
    correct_query,
    is_transposition,
)


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def scalar(self):
        return len(self._rows)


class FakeSession:
    """Stands in for a Session, answering the two queries correct_query runs.

    `vocabulary` is the set of words "in the documents"; `matches` maps an
    unknown token to the (term, similarity) the trigram search would return.
    """

    def __init__(self, vocabulary: set[str], matches: dict | None = None):
        self.vocabulary = vocabulary
        self.matches = matches or {}
        self.queries = 0

    def execute(self, statement, params=None):
        self.queries += 1
        sql = str(statement)
        params = params or {}
        if "term = ANY" in sql:
            return FakeResult(
                [(t,) for t in params["tokens"] if t in self.vocabulary]
            )
        if "CROSS JOIN LATERAL" in sql:
            # Postgres now returns candidates loosely; the similarity
            # threshold and transposition rule are applied in Python.
            rows = []
            for token in params["tokens"]:
                found = self.matches.get(token)
                if found:
                    rows.append((token, found[0], found[1]))
            return FakeResult(rows)
        raise AssertionError(f"unexpected query: {sql[:60]}")


# --- tokenising ----------------------------------------------------------

def test_only_words_long_enough_to_match_are_checked():
    found = _candidates("What is the Coulumb force at 10N between 2 charges?")
    assert "coulumb" in found
    assert "charges" in found
    # Numbers, units and short words are not typos.
    assert "is" not in found
    assert "10n" not in found
    assert all(len(w) >= MIN_LENGTH for w in found)


def test_candidates_are_deduplicated():
    assert _candidates("force force force") == ["force"]


# --- correcting ----------------------------------------------------------

def test_misspelled_term_is_corrected_to_a_corpus_word():
    db = FakeSession(
        vocabulary={"force", "between", "charges", "coulomb"},
        matches={"coulumb": ("coulomb", 0.455)},
    )
    corrected, changes = correct_query(db, "coulumb force between charges")

    assert corrected == "coulomb force between charges"
    assert len(changes) == 1
    assert changes[0].original == "coulumb"
    assert changes[0].corrected == "coulomb"


def test_capitalisation_is_preserved():
    db = FakeSession({"coulomb"}, {"coulumb": ("coulomb", 0.455)})
    corrected, _ = correct_query(db, "What is Coulumb's law")
    assert "Coulomb" in corrected


def test_words_already_in_the_corpus_are_never_touched():
    """The whole question is spelled correctly; nothing should change."""
    db = FakeSession(
        vocabulary={"mitosis", "meiosis", "difference", "between"},
        matches={"mitosis": ("meiosis", 0.99)},  # would fire if consulted
    )
    corrected, changes = correct_query(db, "difference between mitosis and meiosis")

    assert corrected == "difference between mitosis and meiosis"
    assert changes == []


def test_unknown_word_with_no_close_match_is_left_alone():
    db = FakeSession(vocabulary={"mitosis"}, matches={})
    corrected, changes = correct_query(db, "explain zwitterion please")
    assert "zwitterion" in corrected
    assert changes == []


def test_match_below_the_threshold_is_rejected():
    """mitosis vs meiosis scores ~0.33 -- too far apart to be a typo."""
    db = FakeSession(vocabulary={"mitosis"}, matches={"meiosis": ("mitosis", 0.333)})
    corrected, changes = correct_query(db, "what is meiosis", min_similarity=0.4)
    assert "meiosis" in corrected
    assert changes == []


def test_several_typos_in_one_question():
    db = FakeSession(
        vocabulary={"coulomb", "electron"},
        matches={"coulumb": ("coulomb", 0.455), "electorn": ("electron", 0.42)},
    )
    corrected, changes = correct_query(db, "coulumb force on an electorn")
    assert "coulomb" in corrected and "electron" in corrected
    assert len(changes) == 2


def test_lookup_is_batched_not_per_word():
    """Two queries regardless of question length -- one to check which words
    are known, one to find matches for the rest."""
    db = FakeSession({"force"}, {"coulumb": ("coulomb", 0.5)})
    correct_query(db, "coulumb force between charged particles nearby")
    assert db.queries <= 2


def test_database_failure_returns_the_question_unchanged():
    """Spelling help must never be the reason a search fails."""

    class Broken:
        def execute(self, *args, **kwargs):
            raise RuntimeError("connection lost")

    corrected, changes = correct_query(Broken(), "coulumb force")
    assert corrected == "coulumb force"
    assert changes == []


def test_question_with_no_checkable_words_skips_the_database():
    db = FakeSession(set())
    corrected, changes = correct_query(db, "is it 10 N?")
    assert corrected == "is it 10 N?"
    assert db.queries == 0


# --- transposed letters --------------------------------------------------

def test_transposed_letters_are_corrected():
    """"electorn" scores only 0.385 on trigrams -- below the threshold --
    but is one adjacent swap away from "electron"."""
    db = FakeSession(
        vocabulary={"electron", "electric"},
        matches={"electorn": ("electron", 0.385)},
    )
    corrected, changes = correct_query(db, "charge of an electorn")
    assert "electron" in corrected
    assert changes[0].corrected == "electron"


@pytest.mark.parametrize(
    "typed,candidate",
    [
        ("electorn", "electron"),
        ("nuetron", "neutron"),
        ("recieve", "receive"),
        ("freind", "friend"),
        ("thier", "their"),
    ],
)
def test_adjacent_swaps_are_recognised(typed, candidate):
    assert is_transposition(typed, candidate)


@pytest.mark.parametrize(
    "typed,candidate",
    [
        ("mitosis", "meiosis"),   # two substitutions, different words
        ("affect", "effect"),     # one substitution, different words
        ("charge", "change"),     # one substitution, different words
        ("force", "forces"),      # different lengths
        ("coulomb", "coulomb"),   # identical
    ],
)
def test_non_swaps_are_rejected(typed, candidate):
    assert not is_transposition(typed, candidate)


def test_a_swap_is_preferred_over_a_closer_looking_word():
    """Trigram similarity alone would pick the wrong candidate here."""
    db = FakeSession(
        vocabulary={"electron", "electric"},
        matches={"electorn": ("electron", 0.385)},
    )
    _, changes = correct_query(db, "electorn mass")
    assert changes[0].corrected == "electron"


def test_mitosis_survives_the_transposition_rule():
    """The rule must not reopen the door this threshold was protecting."""
    db = FakeSession(vocabulary={"mitosis"}, matches={"meiosis": ("mitosis", 0.333)})
    corrected, changes = correct_query(db, "explain meiosis")
    assert "meiosis" in corrected
    assert changes == []
