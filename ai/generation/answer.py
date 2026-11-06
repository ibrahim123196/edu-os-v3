"""Turn retrieved chunks plus an LLM response into a verifiable answer.

The model emits [S1]-style markers. This module maps them back to real chunks,
drops any marker that points at a source that was never supplied, and reports
whether the answer was actually grounded.

Dropping invalid markers matters. A model that cites [S7] when six sources
were provided has stopped tracking its evidence, and a citation that cannot be
opened is worse than no citation -- it looks like proof.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field

from ai.prompts.grounded_answer import INSUFFICIENT_EVIDENCE
from ai.retrieval.hybrid import RetrievedChunk

_MARKER = re.compile(r"\[S(\d+)\]")

SNIPPET_CHARS = 320


@dataclass
class Citation:
    marker: str  # "S1"
    chunk_id: str
    document_id: str
    document_title: str
    page_start: int | None
    page_end: int | None
    page_label: str
    section: str | None
    snippet: str


@dataclass
class GroundedAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True
    used_source_count: int = 0
    retrieved_count: int = 0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [asdict(c) for c in self.citations],
            "grounded": self.grounded,
            "used_source_count": self.used_source_count,
            "retrieved_count": self.retrieved_count,
        }


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # Prefer to end on a sentence, then a word, rather than mid-token.
    for boundary in (". ", "? ", "! "):
        idx = cut.rfind(boundary)
        if idx > limit * 0.6:
            return cut[: idx + 1]
    idx = cut.rfind(" ")
    return (cut[:idx] if idx > 0 else cut) + "…"


def build_answer(raw: str, chunks: list[RetrievedChunk]) -> GroundedAnswer:
    """Parse markers, validate them against the supplied sources, renumber."""
    raw = (raw or "").strip()

    if not raw:
        return GroundedAnswer(
            answer=INSUFFICIENT_EVIDENCE,
            grounded=False,
            retrieved_count=len(chunks),
        )

    if raw.strip().rstrip(".") == INSUFFICIENT_EVIDENCE.rstrip("."):
        return GroundedAnswer(
            answer=INSUFFICIENT_EVIDENCE,
            grounded=False,
            retrieved_count=len(chunks),
        )

    referenced = [int(n) for n in _MARKER.findall(raw)]
    valid = [n for n in referenced if 1 <= n <= len(chunks)]

    # Renumber so the student sees a contiguous 1..n list even if the model
    # skipped sources. Order of first appearance keeps markers reading in
    # sequence down the answer.
    ordering: dict[int, int] = {}
    for n in valid:
        if n not in ordering:
            ordering[n] = len(ordering) + 1

    def replace(match: re.Match) -> str:
        original = int(match.group(1))
        if original in ordering:
            return f"[S{ordering[original]}]"
        return ""  # invalid marker: strip it rather than mislead

    answer = _MARKER.sub(replace, raw)
    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r"\s+([.,;:])", r"\1", answer).strip()

    citations = []
    for original, new_number in sorted(ordering.items(), key=lambda kv: kv[1]):
        chunk = chunks[original - 1]
        citations.append(
            Citation(
                marker=f"S{new_number}",
                chunk_id=str(chunk.chunk_id),
                document_id=str(chunk.document_id),
                document_title=chunk.document_title,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                page_label=chunk.page_label,
                section=chunk.section,
                snippet=_snippet(chunk.content),
            )
        )

    return GroundedAnswer(
        answer=answer,
        citations=citations,
        grounded=bool(citations),
        used_source_count=len(citations),
        retrieved_count=len(chunks),
    )
