"""LLM 调用协议，所有上层模块只依赖此接口。"""

from __future__ import annotations

from typing import Any, Protocol


ResponseFormat = dict[str, Any]
JSON_OBJECT_RESPONSE_FORMAT: ResponseFormat = {"type": "json_object"}


class ChatModel(Protocol):
    """LLM 调用协议。"""

    def complete(
        self,
        messages: list[dict[str, str]],
        response_format: ResponseFormat | None = None,
    ) -> str:
        """根据消息列表生成回答。"""
        ...
