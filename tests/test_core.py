import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragkit.answer import NO_ANSWER, ExtractiveAnswerer
from ragkit.chunking import chunk_text
from ragkit.embedders import HashingEmbedder
from ragkit.eval import evaluate, load_jsonl
from ragkit.retrieval import fts_query, rrf
from ragkit.service import RagService

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ChunkingTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("One. Two."), ["One. Two."])

    def test_empty_text(self):
        self.assertEqual(chunk_text("   "), [])

    def test_chunks_respect_limit_and_overlap(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(30))
        chunks = chunk_text(text, max_chars=120, overlap_sentences=1)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c) <= 120 + 40 for c in chunks))
        # last sentence of a chunk is the first of the next
        for a, b in zip(chunks, chunks[1:]):
            self.assertEqual(a.split(". ")[-1].rstrip("."), b.split(". ")[0].rstrip("."))

    def test_never_loses_sentences(self):
        sents = [f"Fact {i} stays." for i in range(20)]
        joined = " ".join(chunk_text(" ".join(sents), max_chars=100))
        for s in sents:
            self.assertIn(s, joined)

    def test_rejects_tiny_limit(self):
        with self.assertRaises(ValueError):
            chunk_text("x", max_chars=10)


class EmbedderTests(unittest.TestCase):
    def test_deterministic_and_normalised(self):
        e = HashingEmbedder(256)
        a = e.embed(["the quick brown fox"])
        b = e.embed(["the quick brown fox"])
        np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(float(np.linalg.norm(a[0])), 1.0, places=5)

    def test_similar_text_scores_higher(self):
        e = HashingEmbedder(1024)
        q, near, far = e.embed(["postgres index speed", "a postgres index makes lookups fast", "sunny weather in prague"])
        self.assertGreater(float(q @ near), float(q @ far))

    def test_empty_text_is_zero_vector(self):
        self.assertEqual(float(np.linalg.norm(HashingEmbedder(64).embed([""])[0])), 0.0)


class RetrievalTests(unittest.TestCase):
    def test_fts_query_quotes_and_drops_stopwords(self):
        self.assertEqual(fts_query("What is a cache stampede?"), '"cache" OR "stampede"')

    def test_fts_query_is_injection_safe(self):
        q = fts_query('foo" OR NEAR(bar) AND (baz')
        self.assertNotIn("(", q)
        self.assertNotIn("NEAR", q.replace('"near"', ""))

    def test_rrf_prefers_items_ranked_by_both(self):
        fused = [i for i, _ in rrf([[1, 2, 3], [3, 1, 4]])]
        self.assertEqual(fused[0], 1)
        self.assertEqual(set(fused), {1, 2, 3, 4})

    def test_rrf_is_deterministic_on_ties(self):
        self.assertEqual(rrf([[1], [2]]), rrf([[1], [2]]))


def make_service():
    svc = RagService()
    svc.add_document("pg", "Postgres indexes", "A B-tree index speeds up lookups. Indexes slow down writes.")
    svc.add_document("celery", "Celery retries", "Celery tasks retry with exponential backoff. Tasks must be idempotent.")
    return svc


class ServiceTests(unittest.TestCase):
    def test_query_returns_cited_answer(self):
        r = make_service().query("Why must tasks be idempotent?")
        self.assertEqual(r["sources"][0]["doc_id"], "celery")
        self.assertIn("[1]", r["answer"])

    def test_unknown_topic_is_not_answered(self):
        r = make_service().query("zebra migration patterns")
        self.assertEqual(r["answer"], NO_ANSWER)
        self.assertEqual(r["sources"], [])

    def test_reindexing_replaces_old_chunks(self):
        svc = make_service()
        before = svc.store.count_chunks()
        svc.add_document("pg", "Postgres indexes", "Completely different text about vacuum.")
        self.assertEqual(svc.store.count_chunks(), before)
        self.assertEqual(svc.query("B-tree index lookups", mode="bm25")["sources"], [])

    def test_delete_document_removes_it_from_all_indexes(self):
        svc = make_service()
        svc.store.delete_document("celery")
        for mode in ("bm25", "vector", "hybrid"):
            ids = {s["doc_id"] for s in svc.query("exponential backoff retry", mode=mode)["sources"]}
            self.assertNotIn("celery", ids)

    def test_validation(self):
        svc = make_service()
        with self.assertRaises(ValueError):
            svc.query("   ")
        with self.assertRaises(ValueError):
            svc.add_document("x", "t", "   ")
        with self.assertRaises(ValueError):
            svc.query("q", mode="nope")

    def test_empty_index_does_not_crash(self):
        r = RagService().query("anything at all")
        self.assertEqual(r["sources"], [])

    def test_extractive_answerer_without_chunks(self):
        self.assertEqual(ExtractiveAnswerer().answer("q", []), NO_ANSWER)


class EvalRegressionTests(unittest.TestCase):
    """Guards retrieval quality on the bundled benchmark against silent regressions."""

    def test_bundled_benchmark(self):
        svc = RagService()
        for d in load_jsonl(os.path.join(ROOT, "data", "corpus.jsonl")):
            svc.add_document(d["id"], d["title"], d["text"])
        res = evaluate(svc, load_jsonl(os.path.join(ROOT, "data", "queries.jsonl")), k=3)
        self.assertGreaterEqual(res["bm25"]["recall@3"], 0.95)
        self.assertGreaterEqual(res["hybrid"]["recall@3"], 0.90)
        self.assertGreaterEqual(res["vector"]["recall@3"], 0.75)


if __name__ == "__main__":
    unittest.main()
