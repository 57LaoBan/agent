"""对比 baseline 与 v2 两套 RAG 索引的基础统计指标。

用于直观展示同一份知识库在两种切片/向量化策略下的差异，
便于面试讲解和后续评测脚本（evaluate_rag.py）的对照基准。
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import psycopg


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.config import load_env_file  # noqa: E402


# 两份索引的物理表配置；baseline 没有 documents 表，doc_id 直接落在 chunks。
INDEX_DEFINITIONS = [
    {
        "label": "baseline (固定切片)",
        "documents_table": None,
        "chunks_table": "rag_chunks",
        "metadata_column": "metadata_json",
    },
    {
        "label": "v2 (production_router + bge-m3)",
        "documents_table": "rag_v2_documents",
        "chunks_table": "rag_v2_chunks",
        "metadata_column": "metadata",
    },
]


def main() -> None:
    """连接 PG 并打印各索引的统计对照。"""
    load_env_file(ROOT / "config" / ".env")
    dsn = os.getenv("RAG_PGVECTOR_DSN") or os.getenv("SESSION_DB_DSN")
    if not dsn:
        raise SystemExit("缺少 PostgreSQL DSN，请在 config/.env 配置 RAG_PGVECTOR_DSN。")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            print("=" * 78)
            print(f"{'指标':22s} | " + " | ".join(f"{idx['label']:32s}" for idx in INDEX_DEFINITIONS))
            print("-" * 78)

            stats = [_collect_index_stats(cur, idx) for idx in INDEX_DEFINITIONS]

            metric_keys = [
                ("table_exists", "表是否存在"),
                ("doc_count", "文档数"),
                ("chunk_count", "分块数"),
                ("avg_chunks_per_doc", "平均每文档分块"),
                ("min_content_len", "最短分块字符数"),
                ("max_content_len", "最长分块字符数"),
                ("avg_content_len", "平均分块字符数"),
                ("embedding_dim", "向量维度"),
                ("with_embedding_ratio", "已向量化比例"),
            ]
            for key, label in metric_keys:
                values = [_format_stat(stat.get(key)) for stat in stats]
                print(f"{label:22s} | " + " | ".join(f"{value:32s}" for value in values))

            print("=" * 78)


def _collect_index_stats(cur: psycopg.Cursor[Any], idx: dict[str, Any]) -> dict[str, Any]:
    """收集单个索引的统计指标。"""
    chunks_table = idx["chunks_table"]
    documents_table = idx["documents_table"]
    metadata_column = idx["metadata_column"]

    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = %s
        )
        """,
        (chunks_table,),
    )
    chunks_exists = bool(cur.fetchone()[0])
    if not chunks_exists:
        return {"table_exists": False}

    stats: dict[str, Any] = {"table_exists": True}

    cur.execute(f'SELECT COUNT(*) FROM "{chunks_table}"')
    chunk_count = int(cur.fetchone()[0])
    stats["chunk_count"] = chunk_count

    if documents_table:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            )
            """,
            (documents_table,),
        )
        if cur.fetchone()[0]:
            cur.execute(f'SELECT COUNT(*) FROM "{documents_table}"')
            stats["doc_count"] = int(cur.fetchone()[0])
        else:
            stats["doc_count"] = None
    else:
        cur.execute(f'SELECT COUNT(DISTINCT doc_id) FROM "{chunks_table}"')
        stats["doc_count"] = int(cur.fetchone()[0])

    if stats.get("doc_count"):
        stats["avg_chunks_per_doc"] = round(chunk_count / stats["doc_count"], 2)

    cur.execute(
        f"""
        SELECT
            MIN(LENGTH(content)),
            MAX(LENGTH(content)),
            AVG(LENGTH(content))::int
        FROM "{chunks_table}"
        """
    )
    min_len, max_len, avg_len = cur.fetchone()
    stats["min_content_len"] = min_len
    stats["max_content_len"] = max_len
    stats["avg_content_len"] = avg_len

    cur.execute(
        f"""
        SELECT embedding::text
        FROM "{chunks_table}"
        WHERE embedding IS NOT NULL
        LIMIT 1
        """
    )
    sample = cur.fetchone()
    if sample and sample[0]:
        stats["embedding_dim"] = sample[0].count(",") + 1

    cur.execute(f'SELECT COUNT(*) FROM "{chunks_table}" WHERE embedding IS NOT NULL')
    with_embedding = int(cur.fetchone()[0])
    stats["with_embedding_ratio"] = (
        f"{with_embedding}/{chunk_count} ({with_embedding / chunk_count:.0%})"
        if chunk_count
        else "0/0"
    )

    # 留作扩展占位，避免单元格为空（后续若需要展示 metadata 抽样可在此扩展）。
    _ = metadata_column

    return stats


def _format_stat(value: Any) -> str:
    """统一格式化单元格输出。"""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    return str(value)


if __name__ == "__main__":
    main()
