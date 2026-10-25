"""High-level service tying chunking, storage, retrieval and answering together."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .answer import Answerer, ExtractiveAnswerer
from .chunking import chunk_text
from .embedders import Embedder, HashingEmbedder
from .retrieval import Retriever
from .store import Store


@dataclass
class Source:
    n: int
    doc_id: str
    title: str
    chunk_id: int
    score: float
    text: str


class RagService:
    def __init__(self, db_path: str = ":memory:", embedder: Embedder | None = None,
                 answerer: Answerer | None = None, max_chars: int = 600):
        self.embedder = embedder or HashingEmbedder()
        self.answerer = answerer or ExtractiveAnswerer()
        self.max_chars = max_chars
        self.store = Store(db_path, dim=self.embedder.dim)
        self.retriever = Retriever(self.store, self.embedder)

    def add_document(self, doc_id: str, title: str, text: str, source: str = "") -> int:
        if not doc_id.strip():
            raise ValueError("doc_id must not be empty")
        pieces = chunk_text(text, max_chars=self.max_chars)
        if not pieces:
            raise ValueError("document has no text")
        return self.store.add_document(doc_id, title, source, pieces, self.embedder.embed(pieces))

    def query(self, question: str, k: int = 4, mode: str = "hybrid") -> dict:
        if not question.strip():
            raise ValueError("question must not be empty")
        hits = self.retriever.search(question, k=k, mode=mode)
        chunks = self.store.get_chunks([cid for cid, _ in hits])
        ordered = [(chunks[cid], score) for cid, score in hits if cid in chunks]
        sources = [Source(n, c.doc_id, c.title, c.id, round(s, 4), c.text)
                   for n, (c, s) in enumerate(ordered, start=1)]
        answer = self.answerer.answer(question, [c for c, _ in ordered])
        return {"answer": answer, "mode": mode, "sources": [asdict(s) for s in sources]}
