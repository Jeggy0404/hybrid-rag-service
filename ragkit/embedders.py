"""Embedding backends.

HashingEmbedder is deterministic, offline and dependency-free (feature hashing of
word unigrams and bigrams). It is lexical, not semantic: it exists so the project
runs and is testable without API keys. For real semantic search use OpenAIEmbedder
or plug in any object with `dim` and `embed(texts) -> np.ndarray`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = tokenize(text)
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            counts: dict[str, int] = {}
            for f in feats:
                counts[f] = counts.get(f, 0) + 1
            for f, c in counts.items():
                idx, sign = self._bucket(f)
                out[i, idx] += sign * (1.0 + math.log(c))
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


class OpenAIEmbedder:
    """OpenAI-compatible /v1/embeddings client (stdlib only)."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536,
                 base_url: str = "https://api.openai.com/v1", api_key: str | None = None):
        self.model, self.dim, self.base_url = model, dim, base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def embed(self, texts: list[str]) -> np.ndarray:
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": texts}).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)["data"]
        arr = np.array([d["embedding"] for d in sorted(data, key=lambda d: d["index"])], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)
