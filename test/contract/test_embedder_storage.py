from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.rag import BGEEmbedder, PgVectorStore  # noqa: E402


class FakeEmbeddingModel:
    """测试用向量模型。"""

    def __init__(self) -> None:
        """初始化调用记录。"""
        self.calls: list[object] = []

    def encode(self, texts, **_kwargs):
        """返回可预测的 3 维归一化向量。"""
        self.calls.append(texts)
        if isinstance(texts, list):
            return np.array([self._vector(text) for text in texts], dtype=np.float32)
        return np.array(self._vector(texts), dtype=np.float32)

    def _vector(self, text: str) -> list[float]:
        """按文本长度构造测试向量。"""
        base = float(len(text) or 1)
        vector = np.array([base, base + 1.0, base + 2.0], dtype=np.float32)
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()


class BGEEmbedderTest(unittest.TestCase):
    """BGE 向量化器测试。"""

    def test_embed_query_uses_cache_and_returns_copy(self) -> None:
        model = FakeEmbeddingModel()
        embedder = BGEEmbedder(model=model, query_cache_size=2)

        first = embedder.embed_query("  信易贷产品  ")
        embedder.embed_query("信易贷产品")
        first[0] = 999.0
        third = embedder.embed_query("信易贷产品")

        self.assertEqual(len(model.calls), 1)
        self.assertNotEqual(float(third[0]), 999.0)
        self.assertEqual(embedder.dimension, 3)
        self.assertEqual(embedder.get_metrics()["cache_hits"], 2)

    def test_embed_documents_batches_and_truncates(self) -> None:
        model = FakeEmbeddingModel()
        embedder = BGEEmbedder(model=model, max_length=4)

        embeddings = embedder.embed_documents(["abcdef", "xy"], batch_size=2, show_progress=False)

        self.assertEqual(embeddings.shape, (2, 3))
        self.assertEqual(model.calls[0], ["abcd", "xy"])
        self.assertEqual(embedder.get_metrics()["total_requests"], 2)

    def test_empty_text_is_rejected(self) -> None:
        embedder = BGEEmbedder(model=FakeEmbeddingModel())

        with self.assertRaises(ValueError):
            embedder.embed_query("  ")


class PgVectorStoreTest(unittest.TestCase):
    """pgvector 存储层纯逻辑测试。"""

    def test_vector_literal_validates_dimension(self) -> None:
        store = PgVectorStore("postgresql://user:pass@localhost/db", embedding_dimension=3)

        self.assertEqual(store._vector_literal([1, 2, 3]), "[1,2,3]")
        with self.assertRaises(ValueError):
            store._vector_literal([1, 2])

    def test_filter_clause_supports_doc_id_and_metadata(self) -> None:
        store = PgVectorStore("postgresql://user:pass@localhost/db", embedding_dimension=3)

        sql, params = store._build_filter_clause({"doc_id": "doc_1", "category": "policy"})

        self.assertIn("c.doc_id = %s", sql)
        self.assertIn("c.metadata @> %s::jsonb", sql)
        self.assertEqual(params[0], "doc_1")
        self.assertIn("policy", params[1])

    def test_insert_chunks_batch_uses_unnest_sql(self) -> None:
        store = PgVectorStore("postgresql://user:pass@localhost/db", embedding_dimension=3)
        fake_pool = FakePool()
        store._pool = fake_pool

        asyncio.run(
            store.insert_chunks_batch(
                [
                    {
                        "chunk_id": "chunk_1",
                        "doc_id": "doc_1",
                        "content": "内容",
                        "embedding": [1, 0, 0],
                        "start_char": 0,
                        "end_char": 2,
                        "metadata": {"type": "text"},
                    }
                ]
            )
        )

        executed_sql = fake_pool.conn.executed[0][0]
        self.assertIn("FROM unnest", executed_sql)
        self.assertIn("ON CONFLICT (chunk_id) DO UPDATE", executed_sql)
        self.assertEqual(store.get_metrics()["inserted_chunks"], 1)

    def test_search_similar_maps_rows_to_result_dicts(self) -> None:
        store = PgVectorStore("postgresql://user:pass@localhost/db", embedding_dimension=3)
        fake_pool = FakePool(
            rows=[
                {
                    "chunk_id": "chunk_1",
                    "doc_id": "doc_1",
                    "content": "内容",
                    "similarity": 0.91,
                    "start_char": 0,
                    "end_char": 2,
                    "metadata": {"type": "text"},
                    "document_title": "文档",
                    "source_path": "doc.md",
                }
            ]
        )
        store._pool = fake_pool

        results = asyncio.run(store.search_similar([1, 0, 0], top_k=1))

        self.assertEqual(results[0]["chunk_id"], "chunk_1")
        self.assertEqual(results[0]["similarity"], 0.91)
        self.assertEqual(results[0]["document_title"], "文档")


class FakePool:
    """测试用连接池。"""

    def __init__(self, rows: list[dict] | None = None) -> None:
        """初始化 fake 连接。"""
        self.conn = FakeConnection(rows or [])
        self.size = 1

    def acquire(self):
        """返回异步上下文管理器。"""
        return FakeAcquire(self.conn)


class FakeAcquire:
    """测试用 acquire 上下文。"""

    def __init__(self, conn: "FakeConnection") -> None:
        """保存连接。"""
        self.conn = conn

    async def __aenter__(self) -> "FakeConnection":
        """进入上下文。"""
        return self.conn

    async def __aexit__(self, *_args) -> None:
        """退出上下文。"""


class FakeConnection:
    """测试用异步连接。"""

    def __init__(self, rows: list[dict]) -> None:
        """初始化执行记录。"""
        self.rows = rows
        self.executed: list[tuple[str, tuple]] = []

    async def execute(self, sql: str, params: tuple | None = None) -> "FakeCursor":
        """记录 SQL 并返回游标。"""
        self.executed.append((sql, params or ()))
        return FakeCursor(self.rows)

    def transaction(self):
        """返回事务上下文。"""
        return FakeTransaction()


class FakeCursor:
    """测试用游标。"""

    def __init__(self, rows: list[dict]) -> None:
        """保存查询结果。"""
        self.rows = rows

    async def fetchall(self) -> list[dict]:
        """返回全部行。"""
        return self.rows

    async def fetchone(self) -> dict | None:
        """返回单行。"""
        return self.rows[0] if self.rows else None


class FakeTransaction:
    """测试用事务上下文。"""

    async def __aenter__(self) -> None:
        """进入事务。"""

    async def __aexit__(self, *_args) -> None:
        """退出事务。"""


if __name__ == "__main__":
    unittest.main()
