from __future__ import annotations

import asyncio
import unittest

import numpy as np

from xinyidai_agent.rag.citation import CitationGenerator
from xinyidai_agent.rag.reranker import CrossEncoderReranker
from xinyidai_agent.rag.retriever import (
    DenseRetriever,
    HybridRetriever,
    ProductionRAGRetriever,
    QueryRewriter,
    QueryRouter,
    RetrievalResult,
    SparseRetriever,
)


class FakeEmbedder:
    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


class FakeStore:
    async def search_similar(self, query_embedding, top_k: int = 10, filters=None):
        return [
            {
                "chunk_id": "chunk-a",
                "doc_id": "doc-1",
                "content": "小微税贷支持小微企业申请融资。",
                "similarity": 0.91,
                "metadata": {"doc_type": "product"},
                "document_title": "小微税贷产品说明",
                "source_path": "knowledge/product.md",
                "start_char": 0,
                "end_char": 18,
            },
            {
                "chunk_id": "chunk-b",
                "doc_id": "doc-2",
                "content": "企业需要完成数据授权后才能查询额度。",
                "similarity": 0.72,
                "metadata": {"doc_type": "policy"},
                "document_title": "授权规则",
                "source_path": "knowledge/auth.md",
                "start_char": 0,
                "end_char": 19,
            },
        ][:top_k]

    async def search_sparse(self, query: str, top_k: int = 10, filters=None):
        return [
            {
                "chunk_id": "chunk-c",
                "doc_id": "doc-3",
                "content": "小微税贷年化利率按企业信用等级分层确定。",
                "similarity": 2.1,
                "metadata": {"doc_type": "rate"},
                "document_title": "小微税贷利率规则",
                "source_path": "knowledge/rate.md",
                "start_char": 0,
                "end_char": 21,
            },
            {
                "chunk_id": "chunk-a",
                "doc_id": "doc-1",
                "content": "小微税贷支持小微企业申请融资。",
                "similarity": 1.5,
                "metadata": {"doc_type": "product"},
                "document_title": "小微税贷产品说明",
                "source_path": "knowledge/product.md",
                "start_char": 0,
                "end_char": 18,
            },
        ][:top_k]


class FakeCrossEncoder:
    def predict(self, pairs, batch_size: int = 32):
        return [3.0 if "年化利率" in content else 1.0 for _, content in pairs]


class RetrieverRerankerCitationTest(unittest.TestCase):
    def test_query_rewrite_and_route(self) -> None:
        rewritten = QueryRewriter().rewrite("信易贷利率")
        self.assertIn("年化利率", rewritten)
        self.assertEqual(QueryRouter().route("AUTH_REQUIRED"), "sparse")
        self.assertEqual(QueryRouter().route("如何申请信易贷"), "dense")

    def test_hybrid_retriever_uses_rrf(self) -> None:
        retriever = HybridRetriever(DenseRetriever(FakeEmbedder(), FakeStore()), SparseRetriever(FakeStore()))
        results = asyncio.run(retriever.retrieve("小微税贷利率", top_k=3, mode="hybrid"))

        self.assertEqual([result.chunk_id for result in results], ["chunk-a", "chunk-c", "chunk-b"])
        self.assertEqual(results[0].retrieval_source, "dense+sparse")
        self.assertIn("rrf_score", results[0].metadata)

    def test_cross_encoder_reranker_promotes_best_candidate(self) -> None:
        results = [
            RetrievalResult("chunk-a", "doc-1", "普通产品说明", 0.9),
            RetrievalResult("chunk-c", "doc-3", "小微税贷年化利率规则", 0.6),
        ]
        reranker = CrossEncoderReranker(model=FakeCrossEncoder(), batch_size=2)

        reranked = reranker.rerank("小微税贷利率", results, top_k=2)

        self.assertEqual(reranked[0].chunk_id, "chunk-c")
        self.assertEqual(reranked[0].metadata["retrieval_score"], 0.6)
        self.assertGreaterEqual(reranker.get_metrics()["total_requests"], 1)

    def test_citation_generator_formats_prompt_context(self) -> None:
        result = RetrievalResult(
            chunk_id="chunk-c",
            doc_id="doc-3",
            content="小微税贷年化利率按企业信用等级分层确定。",
            score=0.88,
            document_title="小微税贷利率规则",
            source_path="knowledge/rate.md",
        )
        context = CitationGenerator(max_snippet_chars=20).format_context([result])

        self.assertIn("[1] 小微税贷利率规则 score=0.8800", context)
        self.assertIn("小微税贷年化利率", context)

    def test_production_rag_retriever_returns_sources_and_trace(self) -> None:
        hybrid = HybridRetriever(DenseRetriever(FakeEmbedder(), FakeStore()), SparseRetriever(FakeStore()))
        reranker = CrossEncoderReranker(model=FakeCrossEncoder())
        retriever = ProductionRAGRetriever(hybrid, reranker=reranker, index_version="test-index")

        sources, trace = retriever.retrieve("小微税贷利率", top_k=1)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].title, "小微税贷利率规则")
        self.assertTrue(trace.rerank_applied)
        self.assertEqual(trace.index_version, "test-index")
        self.assertEqual(trace.steps[0]["name"], "hybrid_retrieval")


if __name__ == "__main__":
    unittest.main()
