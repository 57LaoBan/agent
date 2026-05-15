"""长期记忆 PG 存储层：一张 user_facts 表。

复用项目已有的 PostgreSQL + pgvector 基础设施，不引入额外依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_facts (
    fact_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT NOT NULL,
    session_id  TEXT,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding   vector(1024),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_user_facts_user ON user_facts(user_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_user_facts_key ON user_facts(user_id, key) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_user_facts_embedding
    ON user_facts USING ivfflat (embedding vector_cosine_ops) WHERE deleted_at IS NULL;
"""


@dataclass(frozen=True)
class FactRecord:
    """一条用户事实。"""

    fact_id: str
    user_id: str
    key: str
    value: str
    metadata: dict[str, Any]
    created_at: str


class PgFactStore:
    """用户事实的 CRUD 存储。

    设计原则：
    - 同 (user_id, key) 写入新值时，旧记录标 deleted_at，新记录 INSERT（可审计）
    - 查询只返回 deleted_at IS NULL 的活跃记录
    - 向量检索复用 pgvector ivfflat 索引
    """

    def __init__(self, dsn: str, ensure_schema: bool = True) -> None:
        self._dsn = dsn
        if ensure_schema:
            self._ensure_schema()

    def _ensure_schema(self) -> None:
        """幂等创建 user_facts 表。"""
        with psycopg.connect(self._dsn) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()

    def upsert(
        self,
        *,
        user_id: str,
        key: str,
        value: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | np.ndarray | None = None,
    ) -> None:
        """写入或更新一条事实。同 key 旧记录标记失效。"""
        emb_literal = self._vector_literal(embedding) if embedding is not None else None

        with psycopg.connect(self._dsn) as conn:
            with conn.transaction():
                # 标记旧记录失效
                conn.execute(
                    """
                    UPDATE user_facts
                    SET deleted_at = now(), updated_at = now()
                    WHERE user_id = %s AND key = %s AND deleted_at IS NULL
                    """,
                    (user_id, key),
                )
                # 插入新记录
                conn.execute(
                    """
                    INSERT INTO user_facts (user_id, session_id, key, value, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        session_id,
                        key,
                        value,
                        Jsonb(metadata or {}),
                        emb_literal,
                    ),
                )

    def search_by_embedding(
        self,
        *,
        user_id: str,
        query_embedding: list[float] | np.ndarray,
        limit: int = 8,
    ) -> list[FactRecord]:
        """按向量相似度检索用户事实。"""
        emb_literal = self._vector_literal(query_embedding)
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            rows = conn.execute(
                """
                SELECT fact_id::text, user_id, key, value, metadata, created_at::text
                FROM user_facts
                WHERE user_id = %s AND deleted_at IS NULL AND embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (user_id, emb_literal, limit),
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def list_active(self, *, user_id: str, limit: int = 20) -> list[FactRecord]:
        """列出用户全部活跃事实。"""
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            rows = conn.execute(
                """
                SELECT fact_id::text, user_id, key, value, metadata, created_at::text
                FROM user_facts
                WHERE user_id = %s AND deleted_at IS NULL
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def delete_all(self, *, user_id: str) -> int:
        """软删除用户全部事实（显式遗忘）。"""
        with psycopg.connect(self._dsn) as conn:
            cur = conn.execute(
                """
                UPDATE user_facts SET deleted_at = now(), updated_at = now()
                WHERE user_id = %s AND deleted_at IS NULL
                """,
                (user_id,),
            )
            conn.commit()
            return cur.rowcount

    def _vector_literal(self, embedding: list[float] | np.ndarray) -> str:
        """将向量转换为 pgvector 文本字面量。"""
        arr = np.asarray(embedding, dtype=np.float32).ravel()
        return "[" + ",".join(f"{x:.6f}" for x in arr) + "]"

    def _to_record(self, row: dict[str, Any]) -> FactRecord:
        return FactRecord(
            fact_id=row["fact_id"],
            user_id=row["user_id"],
            key=row["key"],
            value=row["value"],
            metadata=row.get("metadata") or {},
            created_at=str(row["created_at"]),
        )
