"""事实写入器：turn 结束时把 confirmed_slots 中有值的字段持久化。

设计原则：
- 只写有业务价值的字段（company_name、product_name、credit_amount 等）
- 同步写入，不引入队列（当前 QPS 不需要）
- 写入时生成 embedding，供后续向量检索
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from xinyidai_agent.memory.longterm.store import PgFactStore
from xinyidai_agent.protocol import SessionStateSnapshot


logger = logging.getLogger(__name__)

# 值得持久化的 slot 白名单
_PERSIST_SLOTS = {"company_name", "product_name", "application_id"}


class FactWriter:
    """从 SessionStateSnapshot 中提取有价值的事实并写入 PG。

    Args:
        store: 事实存储
        embed_fn: 文本 → 向量的函数（复用已有的 BGEEmbedder.embed_query）
    """

    def __init__(
        self,
        store: PgFactStore,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._store = store
        self._embed_fn = embed_fn

    def write_from_state(self, state: SessionStateSnapshot) -> int:
        """从会话状态中提取事实并写入。返回写入条数。"""
        user_id = getattr(state, "ltm_user_id", None) or state.session_id
        written = 0

        # 1. 持久化 confirmed_slots 中的白名单字段
        for key, value in state.confirmed_slots.items():
            if key not in _PERSIST_SLOTS or not value:
                continue
            self._write_fact(
                user_id=user_id,
                session_id=state.session_id,
                key=key,
                value=str(value),
            )
            written += 1

        # 2. 持久化额度查询结果
        if state.last_credit_amount:
            company = state.last_credit_amount.get("company_name") or state.selected_company_name
            amount = state.last_credit_amount.get("credit_amount")
            if company and amount:
                self._write_fact(
                    user_id=user_id,
                    session_id=state.session_id,
                    key=f"credit_amount:{company}",
                    value=str(amount),
                    metadata=state.last_credit_amount,
                )
                written += 1

        # 3. 持久化申请 ID
        if state.last_application_id:
            self._write_fact(
                user_id=user_id,
                session_id=state.session_id,
                key="last_application_id",
                value=state.last_application_id,
            )
            written += 1

        return written

    def _write_fact(
        self,
        *,
        user_id: str,
        session_id: str,
        key: str,
        value: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """写入单条事实，带可选 embedding。"""
        embedding = None
        if self._embed_fn is not None:
            try:
                # 用 "key: value" 格式生成 embedding，便于后续语义检索
                embedding = self._embed_fn(f"{key}: {value}")
            except Exception as exc:
                logger.debug("生成 embedding 失败，跳过：%s", exc)

        try:
            self._store.upsert(
                user_id=user_id,
                key=key,
                value=value,
                session_id=session_id,
                metadata=metadata,
                embedding=embedding,
            )
        except Exception as exc:
            logger.warning("写入事实失败 [%s=%s]：%s", key, value, exc)
