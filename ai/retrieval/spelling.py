"""Typo tolerance, corrected against the student's own material.

Why this exists: full-text search matches exact tokens, so "coulumb" finds
nothing. The semantic search still runs, but a misspelling shifts the
embedding enough to push the right passage down the ranking, and the model
then abstains. One wrong letter and the system looks broken.

Why correct against the corpus rather than a dictionary: a general spellchecker
has never heard of "Kirchhoff", "NCERT" or "zwitterion" and will happily
replace them with something worse. The words in the student's own documents are
exactly the words they are likely to be asking about, so the vocabulary built
from those documents is both smaller and more accurate than any dictionary.

How: pg_trgm compares words by their three-letter fragments. "coulumb" and
"coulomb" share most of theirs, so similarity lands around 0.45, comfortably
above the floor. "coulomb" and "capacitor" share almost none.

Deliberately conservative. A word already in the corpus is never touched, short
words are skipped (trigram similarity is unreliable below four letters), and a
token with no close match is left exactly as the student typed it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

# Runs of letters only. Numbers, symbols and units are left alone: "10N" and
# "mol/L" are not typos, and trigram matching on them produces nonsense.
#
# Apostrophes and hyphens act as separators, so "Coulumb's" yields "coulumb"
# (correctable) plus "s" (too short to check), and the "’s" survives the
# rewrite untouched. This matches how the vocabulary is built, which splits on
# the same non-letter boundaries.
_WORD = re.compile(r"[A-Za-z]+")

MIN_LENGTH = 4


@dataclass(frozen=True)
class Correction:
    original: str
    corrected: str
    similarity: float


def is_transposition(typed: str, candidate: str) -> bool:
    """True when two words differ only by one adjacent letter swap.

    Trigram similarity is structurally bad at transpositions. Swapping two
    letters destroys three trigrams at once, so "electorn" scores 0.385
    against "electron" -- below the threshold -- while a plain wrong letter
    scores well above it. Lowering the threshold to catch swaps would also
    start merging genuinely different words: "mitosis" and "meiosis" sit at
    0.333, and answering the wrong one of those is far worse than answering
    nothing.

    So swaps get their own rule instead. It is deliberately narrow: exactly
    two differing positions, adjacent, holding each other's letters. That
    admits "nuetron" and "recieve" while rejecting "affect"/"effect" and
    "charge"/"change", which differ by substitution rather than a swap and
    are therefore real words rather than typos.
    """
    if len(typed) != len(candidate):
        return False
    differing = [
        i for i, (a, b) in enumerate(zip(typed, candidate)) if a != b
    ]
    return (
        len(differing) == 2
        and differing[1] == differing[0] + 1
        and typed[differing[0]] == candidate[differing[1]]
        and typed[differing[1]] == candidate[differing[0]]
    )


def _candidates(query: str) -> list[str]:
    """Distinct lowercase words long enough to be worth checking."""
    seen: dict[str, None] = {}
    for match in _WORD.finditer(query):
        word = match.group(0).lower()
        if len(word) >= MIN_LENGTH:
            seen.setdefault(word, None)
    return list(seen)


def _known_terms(db: Session, tokens: list[str]) -> set[str]:
    if not tokens:
        return set()
    rows = db.execute(
        text("SELECT term FROM corpus_terms WHERE term = ANY(:tokens)"),
        {"tokens": tokens},
    ).fetchall()
    return {r[0] for r in rows}


def _best_matches(
    db: Session, tokens: list[str], min_similarity: float, candidates: int = 5
) -> dict[str, tuple[str, float]]:
    """Closest corpus term for each unknown token, in a single query.

    Two stages. Postgres generates candidates with the trigram index, which is
    fast but loose; Python then judges them with both the similarity threshold
    and the transposition rule, which is exact but only affordable on a
    handful of words.

    Several candidates are fetched rather than one because a transposed word
    is not always the closest by trigram -- "electorn" ranks below other
    neighbours of "electron" on that metric alone.

    Ties break on frequency: when two terms are equally close, the one the
    documents use more often is the better guess.
    """
    if not tokens:
        return {}

    rows = db.execute(
        text(
            """
            SELECT t.token, m.term, similarity(m.term, t.token) AS sim
            FROM unnest(CAST(:tokens AS text[])) AS t(token)
            CROSS JOIN LATERAL (
                SELECT term, frequency
                FROM corpus_terms
                WHERE term %% t.token
                ORDER BY similarity(term, t.token) DESC, frequency DESC
                LIMIT :candidates
            ) AS m
            """
        ),
        {"tokens": tokens, "candidates": candidates},
    ).fetchall()

    best: dict[str, tuple[str, float]] = {}
    for token, term, sim in rows:
        sim = float(sim)
        swapped = is_transposition(token, term)
        if sim < min_similarity and not swapped:
            continue
        # A confirmed swap is strong evidence, so it outranks a merely
        # similar-looking word even when that word scores higher on trigrams.
        rank = (1 if swapped else 0, sim)
        current = best.get(token)
        if current is None or rank > (1 if is_transposition(token, current[0]) else 0, current[1]):
            best[token] = (term, sim)

    return best


def correct_query(
    db: Session, query: str, min_similarity: float = 0.4
) -> tuple[str, list[Correction]]:
    """Return the query with likely typos fixed, plus what was changed.

    Any database problem returns the query untouched. Spelling help is a
    nicety; it must never be the reason a search fails.
    """
    try:
        tokens = _candidates(query)
        if not tokens:
            return query, []

        known = _known_terms(db, tokens)
        unknown = [t for t in tokens if t not in known]
        if not unknown:
            return query, []

        matches = _best_matches(db, unknown, min_similarity)
        if not matches:
            return query, []
    except Exception:
        log.warning("Spelling correction unavailable", exc_info=True)
        return query, []

    corrections: list[Correction] = []

    def replace(match: re.Match) -> str:
        word = match.group(0)
        found = matches.get(word.lower())
        if found is None:
            return word
        term, similarity = found
        # Preserve the shape the student typed: "Coulumb" -> "Coulomb".
        corrected = term.capitalize() if word[0].isupper() else term
        corrections.append(Correction(word, corrected, round(similarity, 3)))
        return corrected

    return _WORD.sub(replace, query), corrections


def rebuild_vocabulary(db: Session) -> int:
    """Rebuild the word list from every indexed passage.

    Run after ingestion. A full rebuild rather than an incremental update
    because it is one cheap scan and leaves no room for the vocabulary to
    drift out of step with the passages after a delete or a re-ingest.
    """
    db.execute(text("TRUNCATE corpus_terms"))
    db.execute(
        text(
            """
            INSERT INTO corpus_terms (term, frequency)
            SELECT word, COUNT(*)::int
            FROM (
                SELECT regexp_split_to_table(lower(content), '[^a-z]+') AS word
                FROM document_chunks
            ) AS words
            WHERE length(word) >= :min_length
            GROUP BY word
            """
        ),
        {"min_length": MIN_LENGTH},
    )
    count = db.execute(text("SELECT COUNT(*) FROM corpus_terms")).scalar() or 0
    log.info("Vocabulary rebuilt: %d distinct terms", count)
    return count
