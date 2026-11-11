"""Baseline retrieval evaluation.

Answer quality is hard to measure and easy to argue about. Retrieval quality
is neither: either the passage that contains the answer was retrieved, or it
was not. Measuring that first means every later change to chunking, embedding
model, fusion or reranking can be judged rather than guessed at.

The dataset is JSONL, one case per line:

    {"question": "...", "expect_pages": [12, 14], "expect_document": "Biology notes"}
    {"question": "...", "expect_abstain": true}

`expect_abstain` cases matter as much as the rest. A retriever tuned only for
recall will happily surface something for a question the material does not
cover, and the model will then answer it.

Usage:
    python -m ai.evaluation.retrieval_eval ai/evaluation/datasets/sample.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from ai.embeddings.provider import get_embedding_provider
from ai.retrieval.hybrid import RetrievalFilters, hybrid_search
from ai.retrieval.query import to_lexical
from ai.retrieval.rerank import get_reranker
from apps.api.config import settings
from apps.api.db import session_scope


@dataclass
class Case:
    question: str
    expect_pages: list[int]
    expect_document: str | None = None
    expect_abstain: bool = False


def load_dataset(path: Path) -> list[Case]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw = json.loads(line)
        cases.append(
            Case(
                question=raw["question"],
                expect_pages=raw.get("expect_pages", []),
                expect_document=raw.get("expect_document"),
                expect_abstain=raw.get("expect_abstain", False),
            )
        )
    return cases


def _hit(chunk, case: Case) -> bool:
    """A chunk counts as correct if it overlaps an expected page range."""
    if case.expect_document and case.expect_document.lower() not in chunk.document_title.lower():
        return False
    if not case.expect_pages:
        return True
    if chunk.page_start is None:
        return False
    span = range(chunk.page_start, (chunk.page_end or chunk.page_start) + 1)
    return any(p in span for p in case.expect_pages)


def evaluate(dataset: Path, k: int = 5) -> dict:
    """Score retrieval against a dataset of questions with known answers.

    Abstention is measured by similarity, not by whether anything came back.
    Vector search always returns its top k, however poor the matches, so
    "did it return nothing" is never true and would score 0% forever. What
    actually decides abstention upstream is how well the best passage
    matches, which is absolute and comparable across questions -- so that is
    what is reported here, for both kinds of case, and the gap between the
    two distributions is where the threshold belongs.
    """
    cases = load_dataset(dataset)
    provider = get_embedding_provider()
    reranker = get_reranker()

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    answerable_best: list[float] = []
    abstain_best: list[float] = []
    rows = []

    with session_scope() as db:
        for case in cases:
            embedding = provider.embed_query(case.question)
            candidates = hybrid_search(
                db,
                query=to_lexical(case.question),
                query_embedding=embedding,
                filters=RetrievalFilters(),
            )
            best = max((c.vector_score or 0.0 for c in candidates), default=0.0)
            top = reranker.rerank(case.question, candidates, k)

            if case.expect_abstain:
                abstain_best.append(best)
                rows.append((case.question, "abstain", f"best {best:.3f}"))
                continue

            answerable_best.append(best)
            hits = [i for i, c in enumerate(top, start=1) if _hit(c, case)]
            recalls.append(1.0 if hits else 0.0)
            reciprocal_ranks.append(1.0 / hits[0] if hits else 0.0)
            rows.append(
                (
                    case.question,
                    f"rank {hits[0]}" if hits else "MISS",
                    f"{top[0].page_label if top else '-'}  best {best:.3f}",
                )
            )

    threshold = settings.own_material_threshold
    correct_abstentions = sum(1 for b in abstain_best if b < threshold)
    false_abstentions = sum(1 for b in answerable_best if b < threshold)

    def mean(values: list[float]) -> float | None:
        return round(statistics.fmean(values), 3) if values else None

    return {
        "cases": len(cases),
        "k": k,
        f"recall@{k}": mean(recalls),
        "mrr": mean(reciprocal_ranks),
        "similarity": {
            "answerable_mean": mean(answerable_best),
            "answerable_min": round(min(answerable_best), 3) if answerable_best else None,
            "unanswerable_mean": mean(abstain_best),
            "unanswerable_max": round(max(abstain_best), 3) if abstain_best else None,
        },
        "at_threshold": {
            "threshold": threshold,
            "correctly_abstained": f"{correct_abstentions}/{len(abstain_best)}",
            "wrongly_abstained": f"{false_abstentions}/{len(answerable_best)}",
        },
        "config": {
            "embedding_model": settings.embedding_model,
            "reranker": type(reranker).__name__,
            "chunk_target_words": settings.chunk_target_words,
            "rerank_top_n": settings.rerank_top_n,
        },
        "rows": rows,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"No dataset at {path}")
        return 1

    report = evaluate(path)
    rows = report.pop("rows")

    print(json.dumps(report, indent=2))
    print("\nPer case:")
    for question, outcome, detail in rows:
        print(f"  {outcome:<10} {detail:<22} {question[:56]}")
    print(
        "\nA good threshold sits between unanswerable_max and "
        "answerable_min. If they overlap, similarity alone cannot separate "
        "them and the abstention decision needs another signal."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())