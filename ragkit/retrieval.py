"""BM25, vector and hybrid (reciprocal rank fusion) retrieval."""
from __future__ import annotations

import numpy as np

from .embedders import Embedder, tokenize
from .store import Store

STOP = frozenset("a an and are as at be by for from how i in is it of on or that the to what when which who why with do does can".split())


def fts_query(question: str) -> str:
    """Turn free text into a safe FTS5 query: quoted terms joined by OR."""
    terms = [t for t in tokenize(question) if t not in STOP]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion. Each ranking is a list of ids, best first."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, min_similarity: float = 0.1):
        self.store, self.embedder = store, embedder
        self.min_similarity = min_similarity  # drops chance hash-collision matches

    def bm25(self, question: str, k: int) -> list[tuple[int, float]]:
        q = fts_query(question)
        return self.store.bm25(q, k) if q else []

    def vector(self, question: str, k: int) -> list[tuple[int, float]]:
        ids, mat = self.store.all_vectors()
        if len(ids) == 0:
            return []
        qv = self.embedder.embed([question])[0]
        sims = mat @ qv
        top = np.argsort(-sims)[:k]
        return [(int(ids[i]), float(sims[i])) for i in top if sims[i] >= self.min_similarity]

    def hybrid(self, question: str, k: int, fetch: int = 30) -> list[tuple[int, float]]:
        a = [i for i, _ in self.bm25(question, fetch)]
        b = [i for i, _ in self.vector(question, fetch)]
        return rrf([a, b])[:k]

    def search(self, question: str, k: int = 5, mode: str = "hybrid") -> list[tuple[int, float]]:
        if mode == "bm25":
            return self.bm25(question, k)
        if mode == "vector":
            return self.vector(question, k)
        if mode == "hybrid":
            return self.hybrid(question, k)
        raise ValueError(f"unknown mode: {mode}")
