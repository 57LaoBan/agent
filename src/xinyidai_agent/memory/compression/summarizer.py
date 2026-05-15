"""受控 LLM 摘要器，仅在 token 超阈值时触发。

输出为 schema 校验后的 SummaryDelta，由 MemoryManager 决定是否落库。
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from xinyidai_agent.llm.protocol import ChatModel, JSON_OBJECT_RESPONSE_FORMAT


class SummaryDelta(BaseModel):
    """LLM 摘要输出，schema-bound。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    new_summary: str = Field(max_length=600)
    dropped_turn_count: int = 0


class SessionSummarizer:
    """受控 LLM 摘要器。

    仅当 recent_turns 触顶且估算 token 超阈值时由 MemoryManager 调用。
    """

    def __init__(self, model: ChatModel, max_summary_tokens: int = 350) -> None:
        self._model = model
        self._max_summary_tokens = max_summary_tokens

    def compress(
        self,
        *,
        previous_summary: str,
        overflow_turns: list[dict[str, Any]],
    ) -> SummaryDelta:
        """把溢出轮次压缩为新摘要。"""
        messages = self._build_messages(previous_summary, overflow_turns)
        raw = self._model.complete(messages, response_format=JSON_OBJECT_RESPONSE_FORMAT)
        return self._parse(raw, len(overflow_turns))

    def _build_messages(
        self,
        previous_summary: str,
        overflow_turns: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        turns_text = json.dumps(overflow_turns, ensure_ascii=False, default=str)
        return [
            {
                "role": "system",
                "content": (
                    "你是会话摘要器。把以下溢出轮次压缩为一段简短的业务事实摘要，"
                    f"不超过 {self._max_summary_tokens} token。"
                    "只输出 JSON：{\"new_summary\": \"...\", \"dropped_turn_count\": N}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"之前的摘要：{previous_summary or '无'}\n"
                    f"溢出轮次：{turns_text}"
                ),
            },
        ]

    def _parse(self, raw: str, turn_count: int) -> SummaryDelta:
        """解析 LLM 输出，失败时回退为简单截断。"""
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                payload = json.loads(text[start : end + 1])
                return SummaryDelta.model_validate(payload)
        except (json.JSONDecodeError, ValidationError):
            pass
        # 回退：直接用原始文本截断
        return SummaryDelta(
            new_summary=raw[:600] if raw else "",
            dropped_turn_count=turn_count,
        )
