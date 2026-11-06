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
    cases = load_dataset(dataset)
    provider = get_embedding_provider()
    reranker = get_reranker()

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    abstain_correct = 0
    abstain_total = 0
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
            top = reranker.rerank(case.question, candidates, k)

            if case.expect_abstain:
                abstain_total += 1
                # Correct behaviour is retrieving nothing convincing.
                got_nothing = not top
                abstain_correct += int(got_nothing)
                rows.append((case.question, "abstain", "ok" if got_nothing else "leaked"))
                continue

            hits = [i for i, c in enumerate(top, start=1) if _hit(c, case)]
            recalls.append(1.0 if hits else 0.0)
            reciprocal_ranks.append(1.0 / hits[0] if hits else 0.0)
            rows.append(
                (
                    case.question,
                    f"rank {hits[0]}" if hits else "miss",
                    top[0].page_label if top else "-",
                )
            )

    return {
        "cases": len(cases),
        "k": k,
        f"recall@{k}": round(statistics.fmean(recalls), 3) if recalls else None,
        "mrr": round(statistics.fmean(reciprocal_ranks), 3) if reciprocal_ranks else None,
        "abstention_accuracy": (
            round(abstain_correct / abstain_total, 3) if abstain_total else None
        ),
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
        print(f"  {outcome:<10} {detail:<14} {question[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
