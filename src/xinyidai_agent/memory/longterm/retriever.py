"""事实检索器：每轮开始时按 user_id + 向量相似度召回历史事实。

复用已有的 pgvector 索引，一条 SQL 搞定。
"""

from __future__ import annotations

import logging
from typing import Callable

from xinyidai_agent.memory.longterm.store import FactRecord, PgFactStore


logger = logging.getLogger(__name__)


class FactRetriever:
    """按 user_id + query 向量召回历史事实。

    Args:
        store: 事实存储
        embed_fn: 文本 → 向量的函数（复用已有的 BGEEmbedder.embed_query）
        max_results: 最大召回条数
    """

    def __init__(
        self,
        store: PgFactStore,
        embed_fn: Callable[[str], list[float]] | None = None,
        max_results: int = 8,
    ) -> None:
        self._store = store
        self._embed_fn = embed_fn
        self._max_results = max_results

    def retrieve(self, *, user_id: str, query: str) -> list[FactRecord]:
        """按向量相似度检索。embed_fn 不可用时回退到全量列表。"""
        if self._embed_fn is not None:
            try:
                query_embedding = self._embed_fn(query)
                return self._store.search_by_embedding(
                    user_id=user_id,
                    query_embedding=query_embedding,
                    limit=self._max_results,
                )
            except Exception as exc:
                logger.debug("向量检索失败，回退到全量列表：%s", exc)

        # 回退：按时间倒序返回最近的事实
        return self._store.list_active(user_id=user_id, limit=self._max_results)

    def retrieve_text(self, *, user_id: str, query: str) -> str:
        """检索并拼成可注入 prompt 的纯文本。"""
        facts = self.retrieve(user_id=user_id, query=query)
        if not facts:
            return ""
        lines = [f"- {fact.key}: {fact.value}" for fact in facts]
        return "用户历史记忆：\n" + "\n".join(lines)
