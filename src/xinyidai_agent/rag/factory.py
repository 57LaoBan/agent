"""生产级 RAG 检索器工厂。

负责把 env / 配置文件转化为可直接注入到工具注册表的检索器；
对外提供单一入口 ``build_production_rag_runtime``，把 AsyncRuntime、
PgVectorStore、BGE-M3 Embedder、HybridRetriever、ProductionRAGRetriever
组装在一起，并保证以下生命周期约束：

- ``PgVectorStore.initialize()`` 只在 AsyncRuntime 内部 loop 中执行一次；
- 关闭时反向释放：先关 store（断连接池），再关 AsyncRuntime（停 loop / join 线程）。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from xinyidai_agent.rag.async_runtime import AsyncRuntime
from xinyidai_agent.rag.embedder import BGEEmbedder
from xinyidai_agent.rag.reranker import CrossEncoderReranker
from xinyidai_agent.rag.retriever import (
    HybridRetriever,
    ProductionRAGRetriever,
    build_hybrid_retriever,
)
from xinyidai_agent.rag.storage import PgVectorStore


@dataclass(frozen=True)
class RAGConfig:
    """生产 RAG 检索器的运行参数。"""

    dsn: str
    table_prefix: str = "rag_v2_"
    embedder_device: str = "cpu"
    rerank_enabled: bool = False
    reranker_device: str = "cpu"
    pool_min_size: int = 1
    pool_max_size: int = 8
    retrieve_timeout_seconds: float = 90.0
    index_version: str = "v2"
    warmup_on_startup: bool = True

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """从环境变量构造配置；DSN 缺失时返回 None 由调用方决定回退策略。"""
        dsn = (
            os.getenv("RAG_PGVECTOR_DSN")
            or os.getenv("SESSION_DB_DSN")
            or ""
        ).strip()
        if not dsn:
            raise RuntimeError(
                "未配置 RAG DSN，请在 config/.env 中设置 RAG_PGVECTOR_DSN 或 SESSION_DB_DSN。"
            )
        return cls(
            dsn=dsn,
            table_prefix=os.getenv("RAG_TABLE_PREFIX", "rag_v2_"),
            embedder_device=os.getenv("RAG_EMBEDDER_DEVICE", "cpu"),
            rerank_enabled=_to_bool(os.getenv("RAG_RERANKER_ENABLED"), default=False),
            reranker_device=os.getenv("RAG_RERANKER_DEVICE", "cpu"),
            pool_min_size=int(os.getenv("RAG_PG_POOL_MIN", "1")),
            pool_max_size=int(os.getenv("RAG_PG_POOL_MAX", "8")),
            retrieve_timeout_seconds=float(os.getenv("RAG_RETRIEVE_TIMEOUT_SECONDS", "90")),
            index_version=os.getenv("RAG_INDEX_VERSION", "v2"),
            warmup_on_startup=_to_bool(os.getenv("RAG_WARMUP_ON_STARTUP"), default=True),
        )


@dataclass
class ProductionRAGRuntime:
    """绑定生命周期的生产 RAG 资源集合。"""

    runtime: AsyncRuntime
    embedder: BGEEmbedder
    store: PgVectorStore
    hybrid_retriever: HybridRetriever
    retriever: ProductionRAGRetriever
    reranker: CrossEncoderReranker | None = None
    _closed: bool = False

    def close(self) -> None:
        """按反向依赖顺序释放资源。"""
        if self._closed:
            return
        self._closed = True
        # 先在后台 loop 内 close 连接池，再停 loop / join 线程。
        try:
            self.runtime.run_coroutine(self.store.close(), timeout=10.0)
        except Exception:  # noqa: BLE001
            # 关闭过程异常不应阻断 loop 终止，记录由调用方负责。
            pass
        self.runtime.close(join_timeout=10.0)


def build_production_rag_runtime(config: RAGConfig | None = None) -> ProductionRAGRuntime:
    """构造完整的生产 RAG 检索器并完成 store 初始化。"""
    cfg = config or RAGConfig.from_env()

    runtime = AsyncRuntime()
    try:
        embedder = BGEEmbedder(device=cfg.embedder_device)
        store = PgVectorStore(
            cfg.dsn,
            min_pool_size=cfg.pool_min_size,
            max_pool_size=cfg.pool_max_size,
            embedding_dimension=embedder.dimension,
            table_prefix=cfg.table_prefix,
        )
        # 连接池必须在后台 loop 中创建，确保后续协程使用的 asyncio.Queue 同源。
        runtime.run_coroutine(store.initialize(), timeout=30.0)

        reranker: CrossEncoderReranker | None = None
        if cfg.rerank_enabled:
            reranker = CrossEncoderReranker(device=cfg.reranker_device)

        hybrid = build_hybrid_retriever(embedder, store)
        retriever = ProductionRAGRetriever(
            hybrid,
            reranker=reranker,
            index_version=cfg.index_version,
            async_runtime=runtime,
            retrieve_timeout_seconds=cfg.retrieve_timeout_seconds,
        )
        if cfg.warmup_on_startup:
            # fire-and-forget：把首次模型加载/推理放到后台 loop，不阻塞 lifespan startup；
            # 第一请求若来得早会与 warmup 共用 embedder 的内部锁，仍能拿到结果。
            _schedule_warmup(runtime, embedder)
    except Exception:
        # 任何阶段失败都要回收已启动的后台线程，避免泄漏。
        runtime.close(join_timeout=5.0)
        raise

    return ProductionRAGRuntime(
        runtime=runtime,
        embedder=embedder,
        store=store,
        hybrid_retriever=hybrid,
        retriever=retriever,
        reranker=reranker,
    )


def _schedule_warmup(runtime: AsyncRuntime, embedder: BGEEmbedder) -> None:
    """在后台 loop 提交一次 dummy embed，提前完成模型懒加载与首次推理。"""
    import asyncio

    async def _warmup() -> None:
        try:
            await asyncio.to_thread(embedder.embed_query, "warmup")
        except Exception:  # noqa: BLE001
            # 预热失败不影响线上链路；首请求时仍会自然触发加载。
            pass

    asyncio.run_coroutine_threadsafe(_warmup(), runtime.loop)


def _to_bool(value: Any, default: bool) -> bool:
    """把 env 字符串转换为布尔值，兼容常见写法。"""
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default
