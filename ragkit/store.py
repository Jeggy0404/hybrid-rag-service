"""SQLite storage: chunks + FTS5 index (BM25) + float32 vectors as BLOBs."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np


@dataclass
class Chunk:
    id: int
    doc_id: str
    title: str
    ord: int
    text: str


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
    id TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='porter unicode61');
"""


class Store:
    def __init__(self, path: str = ":memory:", dim: int = 1024):
        self.dim = dim
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)

    def delete_document(self, doc_id: str) -> None:
        ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
        self.db.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
        self.db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.db.commit()

    def add_document(self, doc_id: str, title: str, source: str,
                     texts: list[str], vectors: np.ndarray) -> int:
        """Insert (or replace) a document with its chunks. Returns the chunk count."""
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must have the same length")
        if len(texts) and vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of dim {self.dim}, got {vectors.shape[1]}")
        self.delete_document(doc_id)
        self.db.execute("INSERT INTO documents(id,title,source) VALUES(?,?,?)", (doc_id, title, source))
        for i, (t, v) in enumerate(zip(texts, vectors)):
            cur = self.db.execute(
                "INSERT INTO chunks(doc_id,ord,text,embedding) VALUES(?,?,?,?)",
                (doc_id, i, t, v.astype(np.float32).tobytes()))
            self.db.execute("INSERT INTO chunks_fts(rowid,text) VALUES(?,?)", (cur.lastrowid, t))
        self.db.commit()
        return len(texts)

    def count_chunks(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def get_chunks(self, ids: list[int]) -> dict[int, Chunk]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        rows = self.db.execute(
            f"SELECT c.id,c.doc_id,d.title,c.ord,c.text FROM chunks c "
            f"JOIN documents d ON d.id=c.doc_id WHERE c.id IN ({q})", ids)
        return {r[0]: Chunk(*r) for r in rows}

    def all_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        rows = self.db.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
        if not rows:
            return np.zeros(0, dtype=np.int64), np.zeros((0, self.dim), dtype=np.float32)
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        return ids, mat

    def bm25(self, match_query: str, limit: int) -> list[tuple[int, float]]:
        # bm25() returns lower = better; negate so higher = better.
        rows = self.db.execute(
            "SELECT rowid, -bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?", (match_query, limit)).fetchall()
        return [(r[0], r[1]) for r in rows]
