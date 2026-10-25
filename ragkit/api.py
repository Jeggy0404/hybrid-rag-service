"""FastAPI wrapper. Run: uvicorn ragkit.api:app

Environment:
  RAG_DB_PATH      SQLite file (default: ./rag.db)
  OPENAI_API_KEY   if set, OpenAI embeddings + chat are used; otherwise the offline
                   hashing embedder and extractive answerer are used.
"""
from __future__ import annotations

import os
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .answer import ExtractiveAnswerer, OpenAIAnswerer
from .embedders import HashingEmbedder, OpenAIEmbedder
from .service import RagService


class DocumentIn(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1)
    source: str = ""


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)
    mode: Literal["hybrid", "bm25", "vector"] = "hybrid"


def build_service() -> RagService:
    if os.environ.get("OPENAI_API_KEY"):
        return RagService(os.environ.get("RAG_DB_PATH", "rag.db"),
                          embedder=OpenAIEmbedder(), answerer=OpenAIAnswerer())
    return RagService(os.environ.get("RAG_DB_PATH", "rag.db"),
                      embedder=HashingEmbedder(), answerer=ExtractiveAnswerer())


def create_app(service: RagService | None = None) -> FastAPI:
    app = FastAPI(title="hybrid-rag-service", version="0.1.0")
    svc = service or build_service()

    def get_service() -> RagService:
        return svc

    @app.get("/health")
    def health(s: RagService = Depends(get_service)) -> dict:
        return {"status": "ok", "chunks": s.store.count_chunks()}

    @app.post("/documents", status_code=201)
    def add_document(doc: DocumentIn, s: RagService = Depends(get_service)) -> dict:
        try:
            n = s.add_document(doc.id, doc.title, doc.text, doc.source)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"id": doc.id, "chunks": n}

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str, s: RagService = Depends(get_service)) -> None:
        s.store.delete_document(doc_id)

    @app.post("/query")
    def query(q: QueryIn, s: RagService = Depends(get_service)) -> dict:
        return s.query(q.question, k=q.k, mode=q.mode)

    return app


app = create_app() if os.environ.get("RAG_EAGER_APP", "1") == "1" else None
