"""Query preparation.
 
A student's question is usually not a good search query. "sir i dont get this
mitosis thing can u explain" contains one useful retrieval term surrounded by
noise. Two cheap transforms fix most of it:
 
* a lexical query for full-text search, stripped of conversational filler
* an optional LLM rewrite that expands abbreviations and adds the vocabulary a
  textbook would actually use
 
The rewrite is optional and degrades to the original question on any failure.
Retrieval must never depend on a second model call succeeding.
"""
 
from __future__ import annotations
 
import logging
import re
from dataclasses import dataclass, field
 
from ai.retrieval.intent import strip_form_words
from ai.retrieval.spelling import correct_query
from apps.api.config import settings
 
log = logging.getLogger(__name__)
 
_FILLER = re.compile(
    r"\b(please|pls|plz|kindly|sir|maam|ma'am|can you|could you|explain to me|"
    r"i (?:dont|don't|do not) (?:get|understand)|help me (?:with|understand)|"
    r"tell me about|what is meant by|i want to know)\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^\w\s\-']")
_WS = re.compile(r"\s+")
 
REWRITE_PROMPT = """You rewrite a student's question into a search query for a \
textbook search engine.
 
Rules:
- Keep every technical term from the question.
- Expand obvious abbreviations (PYQ -> previous year question, LOC -> \
line of charge) only when unambiguous.
- Add at most three synonyms or closely related textbook terms.
- Do not answer the question.
- Do not add terms about topics the question did not mention.
- Reply with the query text only, no quotes and no explanation.
 
Question: {question}"""
 
 
@dataclass
class PreparedQuery:
    original: str
    lexical: str  # for full-text search
    semantic: str  # for embedding
    rewritten: bool = False
    corrected: str | None = None  # the question after typo fixes, if any
    corrections: list = field(default_factory=list)
 
 
def to_lexical(question: str) -> str:
    """Strip filler and punctuation so websearch_to_tsquery gets clean terms."""
    text = _FILLER.sub(" ", question)
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text or question.strip()
 
 
def _is_usable_rewrite(candidate: str, original: str) -> bool:
    """Decide whether a rewritten query is safe to search with.
 
    A length check alone is not enough. The echo provider replies with the
    abstention sentence to any prompt that has no sources, and the rewrite
    prompt has none -- so the search ran against "I couldn't find enough
    information in your uploaded material", which is 14 words and passed a
    2-to-40-word guard without trouble. Every search then came back empty,
    and the trace was the only place that showed why.
 
    The real test is whether the rewrite still concerns the same thing the
    student asked about. A rewrite sharing no substantial word with the
    question has either refused, answered, or wandered off, and searching
    for it is worse than searching for the original.
    """
    if not candidate:
        return False
    if not 2 <= len(candidate.split()) <= 40:
        return False
 
    # An LLM declining to help is not a search query.
    from ai.prompts.grounded_answer import INSUFFICIENT_EVIDENCE
 
    if INSUFFICIENT_EVIDENCE.rstrip(".").lower() in candidate.lower():
        return False
 
    def substantial(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]{4,}", text.lower())}
 
    original_words = substantial(original)
    if not original_words:
        return True  # nothing to check against; length guard already passed
 
    return bool(original_words & substantial(candidate))
 
 
def prepare_query(question: str, llm=None, db=None) -> PreparedQuery:
    question = question.strip()
 
    # Typos first. Both searches suffer from them -- full-text finds nothing,
    # and a misspelling drags the embedding away from the right passage -- so
    # fixing them before anything else improves both halves at once.
    corrections: list = []
    working = question
    if db is not None and settings.enable_spelling_correction:
        working, corrections = correct_query(
            db, question, min_similarity=settings.spelling_min_similarity
        )
 
    # "formula for X" should search for X. The word "formula" matches prose
    # about formulas, which is the opposite of what was asked for.
    lexical = to_lexical(strip_form_words(working))
    semantic = working
    rewritten = False
 
    # The echo provider has no language model behind it, so asking it to
    # rewrite a query produces a canned sentence rather than a better query.
    can_rewrite = llm is not None and getattr(llm, "model", "") != "echo"
 
    if settings.enable_query_rewriting and can_rewrite:
        try:
            candidate = llm.complete(
                system="You rewrite search queries. Reply with the query only.",
                messages=[{"role": "user", "content": REWRITE_PROMPT.format(question=working)}],
                max_tokens=120,
                temperature=0.0,
            ).strip()
            if _is_usable_rewrite(candidate, working):
                semantic = candidate
                lexical = to_lexical(strip_form_words(candidate))
                rewritten = True
        except Exception:
            log.warning("Query rewrite failed; using the original question", exc_info=True)
 
    return PreparedQuery(
        original=question,
        lexical=lexical,
        semantic=semantic,
        rewritten=rewritten,
        corrected=working if corrections else None,
        corrections=[
            {"from": c.original, "to": c.corrected, "similarity": c.similarity}
            for c in corrections
        ],
    )