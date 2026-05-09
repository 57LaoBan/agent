"""生产级 RAG 检索器。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import inspect
import re
from time import perf_counter
from typing import Any, Literal

from xinyidai_agent.protocol import RetrievalTrace, SourceDocument
from xinyidai_agent.rag.async_runtime import AsyncRuntime
from xinyidai_agent.rag.embedder import BGEEmbedder
from xinyidai_agent.rag.storage import PgVectorStore

RetrievalMode = Literal["dense", "sparse", "hybrid"]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """检索结果，承载排序分数、来源文档和链路元数据。"""

    chunk_id: str
    doc_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    document_title: str | None = None
    source_path: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    retrieval_source: str = "unknown"

    def to_source_document(self) -> SourceDocument:
        """转换为工具层统一证据模型。"""
        title = self.document_title or str(self.metadata.get("title") or self.doc_id)
        metadata = {
            **self.metadata,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "retrieval_source": self.retrieval_source,
        }
        if self.start_char is not None:
            metadata["start_char"] = self.start_char
        if self.end_char is not None:
            metadata["end_char"] = self.end_char
        return SourceDocument(
            source_id=self.chunk_id,
            title=title,
            content=self.content,
            score=self.score,
            url=self.source_path,
            metadata=metadata,
        )


class QueryRewriter:
    """信易贷领域查询改写器。"""

    _synonyms: Mapping[str, tuple[str, ...]] = {
        "信易贷": ("企业信用贷款", "小微信贷", "融资服务"),
        "利率": ("年化利率", "贷款利率", "APR"),
        "额度": ("授信额度", "贷款额度"),
        "准入": ("准入条件", "申请条件", " eligibility "),
        "还款": ("还款方式", "还款计划"),
        "授权": ("企业授权", "数据授权", "AUTH_REQUIRED"),
    }

    def rewrite(self, query: str) -> str:
        """扩展缩写和领域同义词，提升混合检索召回。"""
        normalized = " ".join(str(query or "").split())
        if not normalized:
            raise ValueError("检索查询不能为空")

        additions: list[str] = []
        for keyword, synonyms in self._synonyms.items():
            if keyword in normalized:
                additions.extend(synonym for synonym in synonyms if synonym not in normalized)
        return normalized if not additions else f"{normalized} {' '.join(additions)}"


class QueryRouter:
    """根据查询形态选择 Dense、Sparse 或 Hybrid 检索策略。"""

    _exact_pattern = re.compile(r"([A-Z]{2,}[_-]?\d+|\d{6,}|[A-Z_]{4,}|发票|编号|代码|统一社会信用代码)")
    _conceptual_pattern = re.compile(r"(如何|怎么|为什么|原因|建议|区别|流程|解释|说明)")

    def route(self, query: str) -> RetrievalMode:
        """返回适合当前查询的检索模式。"""
        if self._exact_pattern.search(query):
            return "sparse"
        if self._conceptual_pattern.search(query):
            return "dense"
        return "hybrid"


class DenseRetriever:
    """Dense 检索器，负责查询向量化和 pgvector 相似度召回。"""

    def __init__(self, embedder: BGEEmbedder, store: PgVectorStore) -> None:
        """初始化 Dense 检索依赖。"""
        self._embedder = embedder
        self._store = store

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """执行向量相似度检索。"""
        if top_k <= 0:
            return []
        query_embedding = await asyncio.to_thread(self._embedder.embed_query, query)
        rows = await self._store.search_similar(query_embedding, top_k=top_k, filters=filters)
        return [_result_from_row(row, source="dense") for row in rows]


class SparseRetriever:
    """Sparse 检索器，负责 PostgreSQL 全文索引召回。"""

    def __init__(self, store: PgVectorStore) -> None:
        """初始化 Sparse 检索依赖。"""
        self._store = store

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """执行关键词全文检索。"""
        if top_k <= 0:
            return []
        rows = await self._store.search_sparse(query, top_k=top_k, filters=filters)
        return [_result_from_row(row, source="sparse") for row in rows]


class HybridRetriever:
    """混合检索器，按路由并行执行 Dense/Sparse 并使用 RRF 融合。"""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        rrf_k: int = 60,
        query_rewriter: QueryRewriter | None = None,
        query_router: QueryRouter | None = None,
        fanout_multiplier: int = 4,
    ) -> None:
        """初始化混合检索配置。"""
        if rrf_k <= 0:
            raise ValueError("rrf_k 必须大于 0")
        if fanout_multiplier <= 0:
            raise ValueError("fanout_multiplier 必须大于 0")
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._rrf_k = rrf_k
        self._query_rewriter = query_rewriter or QueryRewriter()
        self._query_router = query_router or QueryRouter()
        self._fanout_multiplier = fanout_multiplier

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[RetrievalResult]:
        """按查询路由执行检索并返回排序后的结果。"""
        if top_k <= 0:
            return []
        rewritten_query = self._query_rewriter.rewrite(query)
        resolved_mode = mode or self._query_router.route(rewritten_query)
        fanout_top_k = max(top_k, top_k * self._fanout_multiplier)

        if resolved_mode == "dense":
            return (await self._dense_retriever.retrieve(rewritten_query, fanout_top_k, filters))[:top_k]
        if resolved_mode == "sparse":
            return (await self._sparse_retriever.retrieve(rewritten_query, fanout_top_k, filters))[:top_k]

        dense_results, sparse_results = await asyncio.gather(
            self._dense_retriever.retrieve(rewritten_query, fanout_top_k, filters),
            self._sparse_retriever.retrieve(rewritten_query, fanout_top_k, filters),
        )
        return self.reciprocal_rank_fusion(dense_results, sparse_results)[:top_k]

    def reciprocal_rank_fusion(
        self,
        dense_results: Sequence[RetrievalResult],
        sparse_results: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        """使用 Reciprocal Rank Fusion 合并两路结果。"""
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, RetrievalResult] = {}
        dense_rank: dict[str, int] = {}
        sparse_rank: dict[str, int] = {}

        for rank, result in enumerate(dense_results, start=1):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (self._rrf_k + rank)
            result_map.setdefault(result.chunk_id, result)
            dense_rank[result.chunk_id] = rank

        for rank, result in enumerate(sparse_results, start=1):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (self._rrf_k + rank)
            result_map.setdefault(result.chunk_id, result)
            sparse_rank[result.chunk_id] = rank

        fused: list[RetrievalResult] = []
        for chunk_id, score in rrf_scores.items():
            base = result_map[chunk_id]
            sources = []
            if chunk_id in dense_rank:
                sources.append("dense")
            if chunk_id in sparse_rank:
                sources.append("sparse")
            metadata = {
                **base.metadata,
                "rrf_score": score,
                "dense_rank": dense_rank.get(chunk_id),
                "sparse_rank": sparse_rank.get(chunk_id),
            }
            fused.append(
                replace(
                    base,
                    score=score,
                    metadata=metadata,
                    retrieval_source="+".join(sources) if sources else "hybrid",
                )
            )

        fused.sort(key=lambda result: result.score, reverse=True)
        return fused


class ProductionRAGRetriever:
    """工具层适配器，串联混合检索、可选重排和引用生成。"""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Any | None = None,
        index_version: str = "pgvector",
        candidate_multiplier: int = 4,
        async_runtime: AsyncRuntime | None = None,
        retrieve_timeout_seconds: float = 30.0,
    ) -> None:
        """初始化可注入到 RagSearchTool 的生产检索器。

        Args:
            async_runtime: 后台异步运行时，注入后同步入口 retrieve() 会通过它把协程
                调度到固定 loop，避免 FastAPI sync handler 每次新建 loop 时与
                pgvector 连接池跨 loop 冲突。
            retrieve_timeout_seconds: 单次同步等待的最长秒数，命中后抛超时异常。
        """
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier 必须大于 0")
        if retrieve_timeout_seconds <= 0:
            raise ValueError("retrieve_timeout_seconds 必须大于 0")
        self._hybrid_retriever = hybrid_retriever
        self._reranker = reranker
        self._index_version = index_version
        self._candidate_multiplier = candidate_multiplier
        self._async_runtime = async_runtime
        self._retrieve_timeout_seconds = retrieve_timeout_seconds

    def retrieve(self, query: str, top_k: int) -> tuple[list[SourceDocument], RetrievalTrace]:
        """同步工具协议入口，内部执行完整异步检索链路。"""
        if self._async_runtime is not None:
            # 走后台 loop，连接池/asyncio.Queue 与该 loop 严格绑定。
            return self._async_runtime.run_coroutine(
                self.retrieve_async(query, top_k),
                timeout=self._retrieve_timeout_seconds,
            )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.retrieve_async(query, top_k))
        raise RuntimeError(
            "ProductionRAGRetriever.retrieve 在已运行事件循环中被调用且未注入 AsyncRuntime，"
            "请改用 retrieve_async 或在构造时传入 async_runtime。"
        )

    async def retrieve_async(self, query: str, top_k: int) -> tuple[list[SourceDocument], RetrievalTrace]:
        """异步执行完整检索流程并生成工具层 trace。"""
        if top_k <= 0:
            trace = RetrievalTrace(query=query, top_k=top_k, results_count=0, retriever_type="hybrid")
            return [], trace

        total_start = perf_counter()
        candidate_top_k = max(top_k, top_k * self._candidate_multiplier)
        steps: list[dict[str, Any]] = []

        retrieve_start = perf_counter()
        candidates = await self._hybrid_retriever.retrieve(query, top_k=candidate_top_k, mode="hybrid")
        steps.append(
            {
                "name": "hybrid_retrieval",
                "candidate_top_k": candidate_top_k,
                "results_count": len(candidates),
                "duration_ms": _elapsed_ms(retrieve_start),
            }
        )

        final_results = candidates[:top_k]
        rerank_applied = False
        if self._reranker is not None and candidates:
            rerank_start = perf_counter()
            final_results = await _call_reranker(self._reranker, query, candidates, top_k)
            rerank_applied = True
            steps.append(
                {
                    "name": "rerank",
                    "model": getattr(self._reranker, "model_name", "cross_encoder"),
                    "results_count": len(final_results),
                    "duration_ms": _elapsed_ms(rerank_start),
                }
            )

        sources = [result.to_source_document() for result in final_results]
        steps.append({"name": "citation_context", "results_count": len(sources), "duration_ms": 0.0})
        steps.append({"name": "total", "duration_ms": _elapsed_ms(total_start)})
        trace = RetrievalTrace(
            query=query,
            top_k=top_k,
            results_count=len(sources),
            retriever_type="hybrid_rrf",
            index_version=self._index_version,
            mock=False,
            rerank_applied=rerank_applied,
            steps=steps,
        )
        return sources, trace


def build_hybrid_retriever(
    embedder: BGEEmbedder,
    store: PgVectorStore,
    rrf_k: int = 60,
) -> HybridRetriever:
    """构建默认生产混合检索器。"""
    return HybridRetriever(DenseRetriever(embedder, store), SparseRetriever(store), rrf_k=rrf_k)


def _result_from_row(row: Mapping[str, Any], source: str) -> RetrievalResult:
    """将存储层字典结果转换为检索结果。"""
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = dict(metadata)
    similarity = row.get("similarity", row.get("score", 0.0))
    return RetrievalResult(
        chunk_id=str(row["chunk_id"]),
        doc_id=str(row["doc_id"]),
        content=str(row["content"]),
        score=float(similarity),
        metadata=metadata,
        document_title=_optional_str(row.get("document_title")),
        source_path=_optional_str(row.get("source_path")),
        start_char=_optional_int(row.get("start_char")),
        end_char=_optional_int(row.get("end_char")),
        retrieval_source=source,
    )


def _optional_str(value: Any) -> str | None:
    """规范化可空字符串。"""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_int(value: Any) -> int | None:
    """规范化可空整数。"""
    if value is None:
        return None
    return int(value)


def _elapsed_ms(start: float) -> float:
    """计算毫秒耗时。"""
    return (perf_counter() - start) * 1000


async def _call_reranker(
    reranker: Any,
    query: str,
    candidates: list[RetrievalResult],
    top_k: int,
) -> list[RetrievalResult]:
    """在线程池或事件循环中调用重排序器，避免阻塞异步检索链路。"""
    rerank = reranker.rerank
    if inspect.iscoroutinefunction(rerank):
        return await rerank(query, candidates, top_k=top_k)
    return await asyncio.to_thread(rerank, query, candidates, top_k=top_k)
