"""Hybrid retrieval over document chunks.

Two independent searches run against the same table:

* semantic  -- pgvector cosine distance, catches paraphrase ("cell splitting"
               finds a passage about mitosis)
* lexical   -- Postgres full-text search, catches exact terminology that
               embeddings blur (formula names, "NCERT", "Class 12", symbols)

Their rankings are merged with Reciprocal Rank Fusion. RRF combines *ranks*
rather than scores, which is what makes it safe here: a cosine distance and a
ts_rank are not on comparable scales, so averaging the raw numbers would be
meaningless.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.orm import Session

from ai.ingestion.notation import NOTATION_THRESHOLD
from apps.api.config import settings


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    content: str
    page_start: int | None
    page_end: int | None
    section: str | None
    chunk_index: int
    notation_score: float = 0.0
    # "student_upload" or "platform". A student's own notes are what their
    # teacher set and what they will be examined on, so they lead; the
    # built-in chapters are there for what the notes do not cover.
    source_kind: str = "student_upload"
    publisher: str | None = None

    vector_rank: int | None = None
    notation_rank: int | None = None
    keyword_rank: int | None = None
    vector_score: float | None = None  # cosine similarity, 0..1
    keyword_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        """Whatever the best available ordering signal is."""
        return self.rerank_score if self.rerank_score is not None else self.fused_score

    @property
    def page_label(self) -> str:
        if self.page_start is None:
            return ""
        if self.page_end and self.page_end != self.page_start:
            return f"Pages {self.page_start}-{self.page_end}"
        return f"Page {self.page_start}"


@dataclass
class RetrievalFilters:
    """Curriculum routing. Empty means search everything available."""

    document_ids: list[uuid.UUID] = field(default_factory=list)
    subject: str | None = None
    grade: str | None = None
    # Which pools to search. None means both.
    source_kinds: list[str] | None = None


def _filter_sql(
    filters: RetrievalFilters, params: dict, min_notation: float | None = None
) -> str:
    clauses = ["c.embedding IS NOT NULL"]
    if min_notation is not None:
        clauses.append("c.notation_score >= :min_notation")
        params["min_notation"] = min_notation
    if filters.document_ids:
        clauses.append("c.document_id = ANY(:document_ids)")
        params["document_ids"] = filters.document_ids
    if filters.subject:
        clauses.append("c.subject = :subject")
        params["subject"] = filters.subject
    if filters.grade:
        clauses.append("c.grade = :grade")
        params["grade"] = filters.grade
    if filters.source_kinds is not None:
        clauses.append("d.source_kind = ANY(:source_kinds)")
        params["source_kinds"] = filters.source_kinds
    return " AND ".join(clauses)


def vector_search(
    db: Session,
    query_embedding: list[float],
    filters: RetrievalFilters,
    limit: int,
    min_notation: float | None = None,
) -> list[RetrievedChunk]:
    """Semantic search. With `min_notation`, restricted to passages that
    actually carry formulas or reactions."""
    params: dict = {"limit": limit}
    where = _filter_sql(filters, params, min_notation)
    # pgvector needs the literal vector syntax, not a Python list.
    params["qvec"] = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"

    sql = text(
        f"""
        SELECT c.id, c.document_id, d.title, c.content, c.page_start, c.page_end,
               c.section, c.chunk_index, c.notation_score,
               d.source_kind, d.publisher,
               1 - (c.embedding <=> CAST(:qvec AS vector)) AS similarity
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {where}
        ORDER BY c.embedding <=> CAST(:qvec AS vector)
        LIMIT :limit
        """
    )
    rows = db.execute(sql, params).fetchall()
    return [
        RetrievedChunk(
            chunk_id=r[0], document_id=r[1], document_title=r[2], content=r[3],
            page_start=r[4], page_end=r[5], section=r[6], chunk_index=r[7],
            notation_score=float(r[8] or 0.0),
            source_kind=r[9] or "student_upload", publisher=r[10],
            vector_rank=i + 1, vector_score=float(r[11]),
        )
        for i, r in enumerate(rows)
    ]


def keyword_search(
    db: Session, query: str, filters: RetrievalFilters, limit: int
) -> list[RetrievedChunk]:
    params: dict = {"limit": limit, "q": query}
    where = _filter_sql(filters, params)

    # websearch_to_tsquery tolerates natural phrasing and quoted phrases
    # instead of erroring on stray punctuation the way to_tsquery does.
    sql = text(
        f"""
        SELECT c.id, c.document_id, d.title, c.content, c.page_start, c.page_end,
               c.section, c.chunk_index, c.notation_score,
               d.source_kind, d.publisher,
               ts_rank_cd(c.content_tsv, websearch_to_tsquery('english', :q)) AS rank
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE {where}
          AND c.content_tsv @@ websearch_to_tsquery('english', :q)
        ORDER BY rank DESC
        LIMIT :limit
        """
    )
    rows = db.execute(sql, params).fetchall()
    return [
        RetrievedChunk(
            chunk_id=r[0], document_id=r[1], document_title=r[2], content=r[3],
            page_start=r[4], page_end=r[5], section=r[6], chunk_index=r[7],
            notation_score=float(r[8] or 0.0),
            source_kind=r[9] or "student_upload", publisher=r[10],
            keyword_rank=i + 1, keyword_score=float(r[11]),
        )
        for i, r in enumerate(rows)
    ]


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Merge ranked lists by summing 1 / (k + rank) per document.

    k dampens the advantage of the top position. At the conventional k=60 the
    gap between rank 1 and rank 2 is small, so a chunk both searches agree on
    outranks a chunk that only one search loved.
    """
    merged: dict[uuid.UUID, RetrievedChunk] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            existing = merged.get(chunk.chunk_id)
            if existing is None:
                merged[chunk.chunk_id] = chunk
                existing = chunk
            else:
                # Carry across whichever per-search signals this copy holds.
                if chunk.vector_rank is not None:
                    existing.vector_rank = chunk.vector_rank
                    existing.vector_score = chunk.vector_score
                if chunk.keyword_rank is not None:
                    existing.keyword_rank = chunk.keyword_rank
                    existing.keyword_score = chunk.keyword_score
                if chunk.notation_rank is not None:
                    existing.notation_rank = chunk.notation_rank
            existing.fused_score += 1.0 / (k + rank)

    return sorted(merged.values(), key=lambda c: c.fused_score, reverse=True)


def hybrid_search(
    db: Session,
    query: str,
    query_embedding: list[float],
    filters: RetrievalFilters | None = None,
    vector_k: int | None = None,
    keyword_k: int | None = None,
    wants_notation: bool = False,
) -> list[RetrievedChunk]:
    """Fuse semantic and lexical search, plus a notation arm when the question
    asks for a formula.

    The third arm exists because the first two systematically miss formulas: a
    line of algebra shares no words with the question and embeds nowhere near
    it. Running the same semantic search restricted to passages that contain
    notation guarantees the formula reaches the ranking stage at all, instead
    of being crowded out by prose that merely discusses it.
    """
    filters = filters or RetrievalFilters()
    vector_k = vector_k or settings.vector_top_k
    keyword_k = keyword_k or settings.keyword_top_k

    rankings = [
        vector_search(db, query_embedding, filters, vector_k),
        keyword_search(db, query, filters, keyword_k),
    ]

    if wants_notation:
        formulas = vector_search(
            db,
            query_embedding,
            filters,
            settings.notation_top_k,
            min_notation=NOTATION_THRESHOLD,
        )
        for rank, chunk in enumerate(formulas, start=1):
            chunk.notation_rank = rank
        rankings.append(formulas)

    return reciprocal_rank_fusion(rankings, k=settings.rrf_k)
