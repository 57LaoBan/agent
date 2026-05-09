from __future__ import annotations

from typing import Protocol

from xinyidai_agent.protocol import RetrievalTrace, SourceDocument
from xinyidai_agent.rag.async_runtime import AsyncRuntime
from xinyidai_agent.rag.document_loader import Document, DocumentLoader
from xinyidai_agent.rag.embedder import BGEEmbedder
from xinyidai_agent.rag.citation import Citation, CitationGenerator
from xinyidai_agent.rag.factory import (
    ProductionRAGRuntime,
    RAGConfig,
    build_production_rag_runtime,
)
from xinyidai_agent.rag.reranker import CrossEncoderReranker
from xinyidai_agent.rag.retriever import (
    DenseRetriever,
    HybridRetriever,
    ProductionRAGRetriever,
    QueryRewriter,
    QueryRouter,
    RetrievalResult,
    SparseRetriever,
    build_hybrid_retriever,
)
from xinyidai_agent.rag.storage import PgVectorStore, VectorSearchResult


class Retriever(Protocol):
    """检索器协议，供工具层和运行时注入真实 RAG 检索实现。"""

    def retrieve(self, query: str, top_k: int) -> tuple[list[SourceDocument], RetrievalTrace]:
        """返回证据片段和检索 trace。"""


class EmptyRetriever:
    """第一阶段占位检索器，后续替换为真实 RAG。"""

    def retrieve(self, query: str, top_k: int) -> tuple[list[SourceDocument], RetrievalTrace]:
        """返回空证据，并显式标记当前仍为未配置的占位检索器。"""
        trace = RetrievalTrace(
            query=query,
            top_k=top_k,
            results_count=0,
            retriever_type="empty",
            index_version="not_configured",
            mock=True,
            rerank_applied=False,
            steps=[{"name": "empty_retriever", "reason": "真实 RAG 尚未接入"}],
        )
        return [], trace


__all__ = [
    "AsyncRuntime",
    "BGEEmbedder",
    "Citation",
    "CitationGenerator",
    "CrossEncoderReranker",
    "DenseRetriever",
    "Document",
    "DocumentLoader",
    "EmptyRetriever",
    "HybridRetriever",
    "PgVectorStore",
    "ProductionRAGRetriever",
    "ProductionRAGRuntime",
    "QueryRewriter",
    "QueryRouter",
    "RAGConfig",
    "Retriever",
    "RetrievalResult",
    "SparseRetriever",
    "VectorSearchResult",
    "build_hybrid_retriever",
    "build_production_rag_runtime",
]
