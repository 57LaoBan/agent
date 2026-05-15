"""多层记忆注入器：STM + LTM → request.metadata。

每轮调用时：
1. 估算 STM 已占 token
2. 用剩余 budget 调 FactRetriever 检索 LTM
3. 裁剪到 budget 内
4. 注入 request.metadata["_ltm_context"]
"""

from __future__ import annotations

import logging

from xinyidai_agent.memory.compression.budget import estimate_state_tokens, estimate_tokens
from xinyidai_agent.memory.longterm.retriever import FactRetriever
from xinyidai_agent.protocol import ChatRequest, SessionStateSnapshot


logger = logging.getLogger(__name__)


class MemoryInjector:
    """多层记忆注入器。"""

    def __init__(
        self,
        *,
        retriever: FactRetriever,
        max_total_tokens: int = 1200,
    ) -> None:
        self._retriever = retriever
        self._max_total_tokens = max_total_tokens

    def compose_and_attach(
        self,
        request: ChatRequest,
        state: SessionStateSnapshot,
    ) -> ChatRequest:
        """召回 LTM 并注入 request.metadata，不改 user_message。"""
        user_id = getattr(state, "ltm_user_id", None) or state.session_id

        # 估算 STM 已占 token
        stm_tokens = estimate_state_tokens(state.short_summary, state.recent_turns)
        remaining = self._max_total_tokens - stm_tokens
        if remaining <= 100:
            return request  # 预算不足，跳过 LTM 注入

        # 调自研 FactRetriever 检索
        try:
            ltm_text = self._retriever.retrieve_text(
                user_id=user_id,
                query=request.user_message,
            )
        except Exception as exc:
            logger.warning("LTM 检索失败，跳过注入：%s", exc)
            return request

        if not ltm_text:
            return request

        # 裁剪到 budget
        ltm_tokens = estimate_tokens(ltm_text)
        if ltm_tokens > remaining:
            lines = ltm_text.split("\n")
            trimmed: list[str] = []
            used = 0
            for line in lines:
                line_tokens = estimate_tokens(line)
                if used + line_tokens > remaining:
                    break
                trimmed.append(line)
                used += line_tokens
            ltm_text = "\n".join(trimmed)

        if not ltm_text.strip():
            return request

        metadata = dict(request.metadata)
        metadata["_ltm_context"] = ltm_text
        return request.model_copy(update={"metadata": metadata})
