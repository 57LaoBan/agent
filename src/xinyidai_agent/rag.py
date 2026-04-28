from __future__ import annotations

from typing import Protocol

from xinyidai_agent.protocol import RetrievalTrace, SourceDocument


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> tuple[list[SourceDocument], RetrievalTrace]:
        """返回证据片段和检索 trace。"""


class EmptyRetriever:
    """第一阶段占位检索器，后续替换为真实 RAG。"""

    def retrieve(self, query: str, top_k: int) -> tuple[list[SourceDocument], RetrievalTrace]:
        trace = RetrievalTrace(
            query=query,
            top_k=top_k,
            results_count=0,
            rerank_applied=False,
            steps=[{"name": "empty_retriever", "reason": "真实 RAG 尚未接入"}],
        )
        return [], trace
