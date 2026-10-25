# hybrid-rag-service

A small, dependency-light retrieval-augmented generation (RAG) service: documents go in, questions come out with **cited** answers. It combines BM25 keyword search (SQLite FTS5) with vector search and merges the two rankings with reciprocal rank fusion (RRF).

This is a personal portfolio project, written to show how I structure a RAG backend: clear module boundaries, testable pieces, a measurable evaluation, and no hidden infrastructure. It is not a production system and has no users.

## What is in it

| Module | Responsibility |
| --- | --- |
| `ragkit/chunking.py` | sentence-aware chunking with overlap |
| `ragkit/embedders.py` | `HashingEmbedder` (offline, deterministic) and `OpenAIEmbedder` (any OpenAI-compatible endpoint) |
| `ragkit/store.py` | SQLite storage: chunks, FTS5 index, float32 vectors |
| `ragkit/retrieval.py` | BM25, vector and hybrid (RRF) search; injection-safe FTS5 query building |
| `ragkit/answer.py` | `ExtractiveAnswerer` (no LLM) and `OpenAIAnswerer` (cites sources as `[n]`) |
| `ragkit/service.py` | `RagService`: add / delete documents, query |
| `ragkit/api.py` | FastAPI endpoints |
| `ragkit/eval.py` | recall@k and MRR benchmark |

## Quick start

```bash
pip install -r requirements.txt
uvicorn ragkit.api:app --reload

curl -X POST localhost:8000/documents -H 'content-type: application/json' \
  -d '{"id":"retries","title":"Celery retries","text":"Celery tasks retry with exponential backoff. Tasks must be idempotent."}'

curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"Why must tasks be idempotent?"}'
```

Set `OPENAI_API_KEY` to switch to OpenAI embeddings and generated answers. Without it everything runs offline.

```bash
docker build -t hybrid-rag . && docker run -p 8000:8000 -v rag-data:/data hybrid-rag
```

## Evaluation

`python -m ragkit.eval --k 3` indexes `data/corpus.jsonl` (20 short technical notes) and scores 40 questions from `data/queries.jsonl`. A question counts as a hit if a chunk of its labelled document is in the top k.

| mode | recall@3 | MRR |
| --- | --- | --- |
| BM25 | 1.000 | 0.988 |
| vector (hashing embedder) | 0.800 | 0.783 |
| hybrid (RRF) | 0.975 | 0.921 |

**Read these numbers carefully.** The corpus and the questions were both written by me, the corpus is tiny, and the default embedder is lexical (hashed word n-grams), not semantic. So BM25 wins and fusion does not help here. That is the expected outcome for this setup, not a claim that BM25 is better in general. The point of the benchmark is to have a repeatable harness: swap in `OpenAIEmbedder` or your own `Embedder` and your own corpus, and the same command tells you whether hybrid retrieval pays off. A regression test fails CI if retrieval quality on the bundled set drops.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Core tests (chunking, embeddings, retrieval, service, eval regression) use only the standard library and numpy. API tests are skipped automatically if FastAPI is not installed.

## Known limits

- Vector search loads all vectors into memory and does a brute-force dot product. Fine up to tens of thousands of chunks; beyond that use an ANN index or pgvector.
- SQLite, single process, no auth, no rate limiting.
- `OpenAIEmbedder` and `OpenAIAnswerer` are thin stdlib clients and are not covered by automated tests (no network in CI).
- Chunking is English-oriented (sentence splitter and stop-word list).

## License

MIT
