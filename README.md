# EduOS — Milestone 1: Grounded Study Chat

Upload your own material, ask questions about it, get answers that cite the
page they came from — and an honest "not in your notes" when the material does
not cover the question.

This is the first vertical slice of the EduOS roadmap. It implements the full
pipeline end to end rather than a prototype of one stage, because everything
later (tutor, quiz engine, student model) sits on top of this retrieval
foundation.

```
upload → extract → chunk → embed → store
                                     ↓
question → prepare → route → hybrid retrieve → fuse → rerank → context → LLM → cite
```

## Running it

You need Docker. Nothing else.

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY, or set LLM_PROVIDER=echo
docker compose up --build
```

Open http://localhost:8000. API docs are at `/docs`.

The first boot downloads a ~130 MB embedding model. After that, embedding runs
offline.

### Without Docker

```bash
docker compose up -d db                       # Postgres with pgvector
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn apps.api.main:app --reload
```

### Without an API key

Set `LLM_PROVIDER=echo`. Ingestion, retrieval, fusion, reranking and citation
parsing all run; only the wording of the answer is stubbed. Useful for testing
the pipeline and for CI.

## What is in here

| Path | What it does |
|---|---|
| `ai/ingestion/` | PDF/text extraction and page-aware chunking |
| `ai/embeddings/` | Embedding providers behind one interface |
| `ai/retrieval/` | Hybrid search, reciprocal rank fusion, reranking, query prep |
| `ai/generation/` | LLM clients and citation validation |
| `ai/prompts/` | The grounding and abstention prompt |
| `ai/evaluation/` | Retrieval eval harness and dataset format |
| `apps/api/` | FastAPI routes, SQLAlchemy models, service layer |
| `web/` | Single-page interface |

## Design decisions worth knowing

**Hybrid retrieval, not embeddings alone.** Semantic search finds paraphrase
("cell splitting" → mitosis). Full-text search finds exact terminology that
embeddings blur — formula names, "NCERT", "Class 12", notation. Indian exam
material is dense with terms of the second kind. The two rankings merge with
Reciprocal Rank Fusion, which combines *ranks* rather than scores, because a
cosine distance and a `ts_rank` are not on comparable scales.

**Reranking is a separate stage.** Retrieval optimises for recall (pull 20
plausible passages), reranking for precision (decide which 6 answer the
question). One retriever tuned to do both misses too much.

**Citations are validated, not trusted.** If the model cites `[S9]` when six
sources were supplied, the marker is stripped rather than shown. A citation
that cannot be opened is worse than none — it looks like proof.

**Abstention is a feature.** A model that quietly answers from general
knowledge looks identical to one that read the notes, right up until it is
confidently wrong about something the student is about to be examined on.

**Every stage is traced.** `POST /chat/ask` with `include_trace: true` returns
what each stage did and how long it took. When an answer is wrong, the trace
says whether retrieval missed the passage or the model ignored it. That
distinction is most of debugging a RAG system and very hard to recover later.

## Tuning

Chunk size is the setting most worth experimenting with. Dense textbook prose
wants larger chunks; sparse bullet-point notes want smaller ones.

```bash
# change CHUNK_TARGET_WORDS in .env, then
curl -X POST localhost:8000/documents/{id}/reingest
curl localhost:8000/documents/{id}/chunks | jq   # inspect the split
```

## Testing

```bash
pytest -m "not integration"      # 39 tests, no database needed
docker compose up -d db
pytest tests/integration -m integration
```

The integration tests cover the raw SQL — the vector cast and
`websearch_to_tsquery` — which unit tests cannot reach.

## Evaluation

Build a baseline before changing anything, per the roadmap.

```bash
python -m ai.evaluation.retrieval_eval ai/evaluation/datasets/sample.jsonl
```

Reports recall@k, MRR, and abstention accuracy. Write your own dataset against
material you have actually uploaded — 30–50 cases, at least a quarter of them
`expect_abstain`. A retriever that never abstains scores well on recall and is
useless in a classroom.

## What Milestone 1 deliberately leaves out

- The curriculum graph. Documents carry flat `subject` and `grade` strings,
  which the retrieval filters already use. Those become foreign keys to
  Exam/Subject/Chapter/Topic/Concept without changing retrieval code.
- Multi-turn conversation. Messages are stored; the pipeline is single-turn
  until Milestone 2.
- Authentication and multi-tenancy. Required before any real pilot.
- The Next.js frontend. `web/` is one static page so the loop is testable
  today; `apps/web/` is where the real client goes.

## Before a real pilot

Not deployable to students as-is. Needed first: authentication and per-student
data isolation, consent and privacy handling for minors, rate limiting, upload
scanning, content licensing review for uploaded material, and a human
oversight path for reporting wrong answers.
