"""Answer generation with numbered citations."""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Protocol

from .embedders import tokenize
from .retrieval import STOP
from .store import Chunk

_SENT = re.compile(r"(?<=[.!?])\s+")
NO_ANSWER = "I could not find this in the indexed documents."


class Answerer(Protocol):
    def answer(self, question: str, chunks: list[Chunk]) -> str: ...


class ExtractiveAnswerer:
    """No LLM: returns the best-matching sentences, each tagged with its source [n]."""

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return NO_ANSWER
        q = {t for t in tokenize(question) if t not in STOP}
        scored = []
        for n, ch in enumerate(chunks, start=1):
            for s in _SENT.split(ch.text):
                overlap = len(q & set(tokenize(s)))
                if overlap:
                    scored.append((overlap, -n, s.strip(), n))
        if not scored:
            return NO_ANSWER
        scored.sort(reverse=True)
        return " ".join(f"{s} [{n}]" for _, _, s, n in scored[: self.max_sentences])


PROMPT = (
    "Answer the question using ONLY the numbered context. Cite sources like [1] after each claim. "
    "If the context does not contain the answer, reply exactly: " + NO_ANSWER
)


class OpenAIAnswerer:
    """OpenAI-compatible chat completions client (stdlib only)."""

    def __init__(self, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1",
                 api_key: str | None = None):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return NO_ANSWER
        context = "\n\n".join(f"[{n}] ({c.title}) {c.text}" for n, c in enumerate(chunks, start=1))
        body = {"model": self.model, "temperature": 0, "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)["choices"][0]["message"]["content"].strip()
