"""Retrieval evaluation: recall@k and MRR for bm25, vector and hybrid.

    python -m ragkit.eval [--k 3] [--corpus data/corpus.jsonl] [--queries data/queries.jsonl]

Relevance is judged at document level (a hit is any chunk of a relevant document).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import RagService

MODES = ("bm25", "vector", "hybrid")


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(svc: RagService, queries: list[dict], k: int) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for mode in MODES:
        hits_at_k = 0
        rr_sum = 0.0
        for item in queries:
            relevant = set(item["relevant"])
            hits = svc.retriever.search(item["question"], k=k, mode=mode)
            chunks = svc.store.get_chunks([cid for cid, _ in hits])
            docs = []
            for cid, _ in hits:
                if cid in chunks and chunks[cid].doc_id not in docs:
                    docs.append(chunks[cid].doc_id)
            rank = next((i for i, d in enumerate(docs, start=1) if d in relevant), None)
            if rank is not None:
                hits_at_k += 1
                rr_sum += 1.0 / rank
        n = len(queries)
        results[mode] = {f"recall@{k}": hits_at_k / n, "mrr": rr_sum / n}
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    svc = RagService()
    for d in load_jsonl(args.corpus):
        svc.add_document(d["id"], d["title"], d["text"])
    queries = load_jsonl(args.queries)
    res = evaluate(svc, queries, args.k)
    print(f"{len(queries)} questions, {svc.store.count_chunks()} chunks")
    print(f"{'mode':8} {'recall@'+str(args.k):>9} {'MRR':>7}")
    for mode, m in res.items():
        print(f"{mode:8} {m[f'recall@{args.k}']:>9.3f} {m['mrr']:>7.3f}")


if __name__ == "__main__":
    main()
