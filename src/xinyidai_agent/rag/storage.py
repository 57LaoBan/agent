"""pgvector 存储层。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
import json
from time import perf_counter
from typing import Any, AsyncIterator

import numpy as np
import psycopg
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """向量检索结果。"""

    chunk_id: str
    doc_id: str
    content: str
    similarity: float
    start_char: int | None
    end_char: int | None
    metadata: dict[str, Any]
    document_title: str | None = None
    source_path: str | None = None


class PgVectorStore:
    """PostgreSQL + pgvector 存储层。

    使用 psycopg 异步连接和内部连接池，提供表结构初始化、文档/分块 upsert、
    批量分块插入、Dense 相似度检索与基础 Sparse 检索。
    """

    def __init__(
        self,
        connection_string: str,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
        embedding_dimension: int = 1024,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 40,
        table_prefix: str = "",
    ) -> None:
        """初始化 pgvector 存储层配置。

        Args:
            table_prefix: 表名命名空间前缀，用于在同库下隔离不同版本的 RAG 索引
                （例如 baseline 与 v2）。默认为空，表名仍为 rag_documents/rag_chunks；
                传入 "rag_v2_" 时表名变为 rag_v2_documents/rag_v2_chunks。
        """
        if min_pool_size <= 0:
            raise ValueError("min_pool_size 必须大于 0")
        if max_pool_size < min_pool_size:
            raise ValueError("max_pool_size 必须大于等于 min_pool_size")
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension 必须大于 0")
        normalized_prefix = self._normalize_table_prefix(table_prefix)

        self._connection_string = connection_string
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._embedding_dimension = embedding_dimension
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search
        self._table_prefix = normalized_prefix
        # 表名与索引名一次性确定，后续 SQL 直接复用，避免每条语句重复拼接。
        prefix_or_default = normalized_prefix or "rag_"
        self._documents_table = f"{prefix_or_default}documents" if normalized_prefix else "rag_documents"
        self._chunks_table = f"{prefix_or_default}chunks" if normalized_prefix else "rag_chunks"
        index_stem = (normalized_prefix or "rag_").rstrip("_")
        self._idx_chunks_hnsw = f"idx_{index_stem}_chunks_embedding_hnsw"
        self._idx_chunks_doc_id = f"idx_{index_stem}_chunks_doc_id"
        self._idx_chunks_metadata = f"idx_{index_stem}_chunks_metadata"
        self._idx_chunks_content_tsv = f"idx_{index_stem}_chunks_content_tsv"
        self._pool: _AsyncPsycopgPool | None = None
        self._metrics = {
            "search_count": 0,
            "inserted_documents": 0,
            "inserted_chunks": 0,
            "total_search_duration": 0.0,
        }

    async def initialize(self) -> None:
        """初始化连接池、pgvector 扩展、表结构和索引。"""
        self._pool = _AsyncPsycopgPool(
            self._connection_string,
            min_size=self._min_pool_size,
            max_size=self._max_pool_size,
        )
        await self._pool.open()

        async with self._acquire() as conn:
            async with conn.transaction():
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await self._create_tables(conn)
                await self._create_indexes(conn)

    @property
    def table_prefix(self) -> str:
        """返回当前命名空间前缀。"""
        return self._table_prefix

    @property
    def documents_table(self) -> str:
        """返回文档表全名（含前缀）。"""
        return self._documents_table

    @property
    def chunks_table(self) -> str:
        """返回分块表全名（含前缀）。"""
        return self._chunks_table

    async def insert_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        source_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """插入或更新文档。"""
        self._ensure_initialized()
        async with self._acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._documents_table} (doc_id, title, content, source_path, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE
                SET title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    source_path = EXCLUDED.source_path,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                (doc_id, title, content, source_path, Jsonb(metadata or {})),
            )
        self._metrics["inserted_documents"] += 1

    async def insert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        embedding: np.ndarray | list[float],
        start_char: int,
        end_char: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """插入或更新单个文档分块。"""
        await self.insert_chunks_batch(
            [
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "content": content,
                    "embedding": embedding,
                    "start_char": start_char,
                    "end_char": end_char,
                    "metadata": metadata or {},
                }
            ]
        )

    async def insert_chunks_batch(self, chunks: list[dict[str, Any]]) -> None:
        """批量插入或更新分块，使用 unnest 降低数据库往返次数。"""
        self._ensure_initialized()
        if not chunks:
            return

        chunk_ids: list[str] = []
        doc_ids: list[str] = []
        contents: list[str] = []
        embeddings: list[str] = []
        start_chars: list[int] = []
        end_chars: list[int] = []
        metadata_rows: list[str] = []

        for chunk in chunks:
            chunk_ids.append(str(chunk["chunk_id"]))
            doc_ids.append(str(chunk["doc_id"]))
            contents.append(str(chunk["content"]))
            embeddings.append(self._vector_literal(chunk["embedding"]))
            start_chars.append(int(chunk.get("start_char") or 0))
            end_chars.append(int(chunk.get("end_char") or len(str(chunk["content"]))))
            metadata_rows.append(json.dumps(chunk.get("metadata") or {}, ensure_ascii=False))

        async with self._acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    f"""
                    INSERT INTO {self._chunks_table} (
                        chunk_id, doc_id, content, embedding, start_char, end_char, metadata
                    )
                    SELECT
                        input.chunk_id,
                        input.doc_id,
                        input.content,
                        input.embedding::vector({self._embedding_dimension}),
                        input.start_char,
                        input.end_char,
                        input.metadata::jsonb
                    FROM unnest(
                        %s::text[],
                        %s::text[],
                        %s::text[],
                        %s::text[],
                        %s::integer[],
                        %s::integer[],
                        %s::text[]
                    ) AS input(chunk_id, doc_id, content, embedding, start_char, end_char, metadata)
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET doc_id = EXCLUDED.doc_id,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        start_char = EXCLUDED.start_char,
                        end_char = EXCLUDED.end_char,
                        metadata = EXCLUDED.metadata,
                        updated_at = now()
                    """,
                    (chunk_ids, doc_ids, contents, embeddings, start_chars, end_chars, metadata_rows),
                )
        self._metrics["inserted_chunks"] += len(chunks)

    async def search_similar(
        self,
        query_embedding: np.ndarray | list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Dense 向量相似度检索。"""
        self._ensure_initialized()
        if top_k <= 0:
            return []

        start_time = perf_counter()
        vector = self._vector_literal(query_embedding)
        where_sql, params = self._build_filter_clause(filters)

        async with self._acquire() as conn:
            # 用 SET LOCAL 包在事务里，避免连接复用时 GUC 污染后续会话。
            async with conn.transaction():
                # SET 语句不支持参数化协议；ef_search 已经是 int，直接拼接安全。
                await conn.execute(
                    f"SET LOCAL hnsw.ef_search = {int(self._hnsw_ef_search)}"
                )
                rows = await conn.execute(
                    f"""
                    SELECT
                        c.chunk_id,
                        c.doc_id,
                        c.content,
                        1 - (c.embedding <=> %s::vector({self._embedding_dimension})) AS similarity,
                        c.start_char,
                        c.end_char,
                        c.metadata,
                        d.title AS document_title,
                        d.source_path
                    FROM {self._chunks_table} c
                    JOIN {self._documents_table} d ON d.doc_id = c.doc_id
                    WHERE c.embedding IS NOT NULL {where_sql}
                    ORDER BY c.embedding <=> %s::vector({self._embedding_dimension})
                    LIMIT %s
                    """,
                    (vector, *params, vector, top_k),
                )
                result_rows = await rows.fetchall()

        self._metrics["search_count"] += 1
        self._metrics["total_search_duration"] += perf_counter() - start_time
        return [asdict(self._row_to_result(row)) for row in result_rows]

    async def search_sparse(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """基于 PostgreSQL 全文索引的 Sparse 检索。"""
        self._ensure_initialized()
        if top_k <= 0:
            return []
        where_sql, params = self._build_filter_clause(filters)

        async with self._acquire() as conn:
            rows = await conn.execute(
                f"""
                SELECT
                    c.chunk_id,
                    c.doc_id,
                    c.content,
                    ts_rank_cd(
                        to_tsvector('simple', c.content),
                        plainto_tsquery('simple', %s)
                    ) AS similarity,
                    c.start_char,
                    c.end_char,
                    c.metadata,
                    d.title AS document_title,
                    d.source_path
                FROM {self._chunks_table} c
                JOIN {self._documents_table} d ON d.doc_id = c.doc_id
                WHERE to_tsvector('simple', c.content) @@ plainto_tsquery('simple', %s)
                {where_sql}
                ORDER BY similarity DESC
                LIMIT %s
                """,
                (query, query, *params, top_k),
            )
            result_rows = await rows.fetchall()
        return [asdict(self._row_to_result(row)) for row in result_rows]

    async def count_documents(self) -> int:
        """统计文档数。"""
        self._ensure_initialized()
        async with self._acquire() as conn:
            row = await (
                await conn.execute(f"SELECT COUNT(*) AS count FROM {self._documents_table}")
            ).fetchone()
        return int(row["count"])

    async def count_chunks(self) -> int:
        """统计分块数。"""
        self._ensure_initialized()
        async with self._acquire() as conn:
            row = await (
                await conn.execute(f"SELECT COUNT(*) AS count FROM {self._chunks_table}")
            ).fetchone()
        return int(row["count"])

    def get_metrics(self) -> dict[str, float | int]:
        """获取存储层指标快照。"""
        search_count = int(self._metrics["search_count"])
        avg_search_ms = (
            float(self._metrics["total_search_duration"]) / search_count * 1000
            if search_count > 0
            else 0.0
        )
        return {
            "search_count": search_count,
            "inserted_documents": int(self._metrics["inserted_documents"]),
            "inserted_chunks": int(self._metrics["inserted_chunks"]),
            "avg_search_ms": avg_search_ms,
            "pool_size": self._pool.size if self._pool else 0,
        }

    async def close(self) -> None:
        """关闭连接池。"""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _create_tables(self, conn: AsyncConnection) -> None:
        """创建 RAG 文档表和分块表（受 table_prefix 影响）。"""
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._documents_table} (
                doc_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source_path TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._chunks_table} (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES {self._documents_table}(doc_id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                embedding vector({self._embedding_dimension}),
                start_char INTEGER,
                end_char INTEGER,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

    async def _create_indexes(self, conn: AsyncConnection) -> None:
        """创建向量、元数据和全文索引（索引名亦受 table_prefix 影响）。"""
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self._idx_chunks_hnsw}
            ON {self._chunks_table}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = {self._hnsw_m}, ef_construction = {self._hnsw_ef_construction})
            """
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self._idx_chunks_doc_id} ON {self._chunks_table}(doc_id)"
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self._idx_chunks_metadata} "
            f"ON {self._chunks_table} USING gin(metadata)"
        )
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self._idx_chunks_content_tsv}
            ON {self._chunks_table}
            USING gin(to_tsvector('simple', content))
            """
        )

    @staticmethod
    def _normalize_table_prefix(prefix: str) -> str:
        """校验并规范化表前缀，避免 SQL 注入与非法标识符。

        允许字符：小写字母、数字、下划线；以字母开头；最长 32 字符。
        """
        normalized = (prefix or "").strip()
        if not normalized:
            return ""
        if len(normalized) > 32:
            raise ValueError("table_prefix 长度不能超过 32")
        if not normalized[0].isalpha():
            raise ValueError("table_prefix 必须以字母开头")
        for char in normalized:
            if not (char.isalnum() or char == "_"):
                raise ValueError(f"table_prefix 含非法字符: {char!r}")
        return normalized.lower()

    def _ensure_initialized(self) -> None:
        """确保连接池已初始化。"""
        if self._pool is None:
            raise RuntimeError("PgVectorStore 尚未 initialize")

    @asynccontextmanager
    async def _acquire(self) -> AsyncIterator[AsyncConnection]:
        """从内部连接池获取连接。"""
        self._ensure_initialized()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            yield conn

    def _vector_literal(self, embedding: np.ndarray | list[float]) -> str:
        """将向量转换为 pgvector 文本字面量。"""
        array = np.asarray(embedding, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"向量必须是一维数组，实际维度：{array.ndim}")
        if array.shape[0] != self._embedding_dimension:
            raise ValueError(f"向量维度不匹配：期望 {self._embedding_dimension}，实际 {array.shape[0]}")
        return "[" + ",".join(f"{float(value):.8g}" for value in array.tolist()) + "]"

    def _build_filter_clause(self, filters: dict[str, Any] | None) -> tuple[str, tuple[Any, ...]]:
        """构建安全的 doc_id 和 metadata JSONB 过滤条件。"""
        if not filters:
            return "", ()

        clauses: list[str] = []
        params: list[Any] = []
        doc_id = filters.get("doc_id")
        if doc_id is not None:
            clauses.append("AND c.doc_id = %s")
            params.append(str(doc_id))

        metadata_filter = {key: value for key, value in filters.items() if key != "doc_id"}
        if metadata_filter:
            clauses.append("AND c.metadata @> %s::jsonb")
            params.append(json.dumps(metadata_filter, ensure_ascii=False))

        return " " + " ".join(clauses) if clauses else "", tuple(params)

    def _row_to_result(self, row: dict[str, Any]) -> VectorSearchResult:
        """数据库行转检索结果。"""
        metadata = row["metadata"] or {}
        if not isinstance(metadata, dict):
            metadata = dict(metadata)
        return VectorSearchResult(
            chunk_id=str(row["chunk_id"]),
            doc_id=str(row["doc_id"]),
            content=str(row["content"]),
            similarity=float(row["similarity"]),
            start_char=row["start_char"],
            end_char=row["end_char"],
            metadata=metadata,
            document_title=row["document_title"],
            source_path=row["source_path"],
        )


class _AsyncPsycopgPool:
    """最小异步连接池，避免为当前模块额外引入 psycopg_pool 依赖。"""

    def __init__(self, dsn: str, min_size: int, max_size: int) -> None:
        """初始化连接池配置。"""
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._queue: asyncio.Queue[AsyncConnection] = asyncio.Queue(maxsize=max_size)
        self._created = 0
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def size(self) -> int:
        """当前已创建连接数。"""
        return self._created

    async def open(self) -> None:
        """预创建最小连接数。"""
        async with self._lock:
            for _ in range(self._min_size):
                conn = await self._new_connection()
                await self._queue.put(conn)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[AsyncConnection]:
        """获取并自动归还连接。"""
        if self._closed:
            raise RuntimeError("连接池已关闭")

        conn = await self._acquire_connection()
        try:
            yield conn
        except (psycopg.Error, asyncio.CancelledError):
            await self._discard_connection(conn)
            raise
        except Exception:
            await self._queue.put(conn)
            raise
        else:
            await self._queue.put(conn)

    async def close(self) -> None:
        """关闭所有空闲连接。"""
        self._closed = True
        while not self._queue.empty():
            conn = await self._queue.get()
            await conn.close()
            self._created -= 1

    async def _acquire_connection(self) -> AsyncConnection:
        """从队列取连接，必要时创建新连接。"""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            async with self._lock:
                if self._created < self._max_size:
                    return await self._new_connection()
            return await self._queue.get()

    async def _new_connection(self) -> AsyncConnection:
        """创建新的异步连接。

        统一开启 autocommit：避免单条 execute 因隐式事务未 commit 而在连接归还/关闭时被回滚；
        需要原子性的写操作仍可通过 `async with conn.transaction()` 显式开启事务。
        """
        conn = await AsyncConnection.connect(self._dsn, row_factory=dict_row, autocommit=True)
        self._created += 1
        return conn

    async def _discard_connection(self, conn: AsyncConnection) -> None:
        """丢弃异常连接。"""
        await conn.close()
        async with self._lock:
            self._created -= 1
            if not self._closed and self._created < self._min_size:
                replacement = await self._new_connection()
                await self._queue.put(replacement)
