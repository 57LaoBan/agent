"""L0 守卫：纯规则前置过滤，命中立刻返回。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from xinyidai_agent.protocol import ChatRequest

GuardKind = Literal["pass", "reject_empty", "reject_invalid", "continuation_confirm"]


@dataclass(frozen=True)
class GuardResult:
    """规则守卫结果。"""

    kind: GuardKind
    user_visible_message: str = ""
    reason: str = ""


class RulesGuard:
    """L0 规则守卫，处理空输入、无意义输入和确认续接。"""

    _CONFIRM_REPLIES = frozenset({"确认", "好", "好的", "ok", "yes", "y", "是", "对", "同意"})

    def guard(self, request: ChatRequest, *, has_pending_action: bool) -> GuardResult:
        """守卫入口，命中确定性场景时返回短路结果。"""
        message = (request.user_message or "").strip()
        if not message:
            return GuardResult(
                kind="reject_empty",
                user_visible_message="请发送您的问题或诉求，我会为您解答。",
                reason="empty_message",
            )
        if not self._has_meaningful_content(message):
            return GuardResult(
                kind="reject_invalid",
                user_visible_message="您的问题我没看明白，能详细描述一下吗？",
                reason="no_chinese_or_letter",
            )
        if has_pending_action and self._is_confirmation_reply(message):
            return GuardResult(kind="continuation_confirm", reason="user_confirmed_pending_action")
        return GuardResult(kind="pass")

    @staticmethod
    def _has_meaningful_content(message: str) -> bool:
        """判断是否包含中文、字母或数字。"""
        for char in message:
            if "\u4e00" <= char <= "\u9fff" or char.isalnum():
                return True
        return False

    @classmethod
    def _is_confirmation_reply(cls, message: str) -> bool:
        """判断是否是肯定确认。"""
        return message.strip().lower() in cls._CONFIRM_REPLIES
