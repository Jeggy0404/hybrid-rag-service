import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

os.environ["RAG_EAGER_APP"] = "0"
from ragkit.api import create_app  # noqa: E402
from ragkit.service import RagService  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(create_app(RagService()))


def test_health_empty(client):
    assert client.get("/health").json() == {"status": "ok", "chunks": 0}


def test_add_and_query(client):
    r = client.post("/documents", json={"id": "d1", "title": "Retries",
                                        "text": "Celery tasks retry with exponential backoff."})
    assert r.status_code == 201 and r.json()["chunks"] == 1
    q = client.post("/query", json={"question": "How do tasks retry with backoff?"}).json()
    assert q["sources"][0]["doc_id"] == "d1"
    assert "[1]" in q["answer"]


def test_query_validation(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "x", "mode": "magic"}).status_code == 422
    assert client.post("/query", json={"question": "x", "k": 99}).status_code == 422


def test_blank_document_rejected(client):
    r = client.post("/documents", json={"id": "d", "title": "t", "text": "   "})
    assert r.status_code == 422


def test_delete(client):
    client.post("/documents", json={"id": "d1", "title": "T", "text": "Redis caches query results."})
    assert client.delete("/documents/d1").status_code == 204
    assert client.get("/health").json()["chunks"] == 0
