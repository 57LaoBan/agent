"""L1 场景直达：高确定性输入跳过模型路由。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re

from xinyidai_agent.capabilities.catalog import CapabilityCatalog
from xinyidai_agent.protocol import ChatRequest


@dataclass(frozen=True)
class SceneDirectHit:
    """L1 直达命中结果。"""

    handler_name: str
    final_answer: str
    stop_reason: str = "answered_without_tool"


class SceneDirectDispatcher:
    """L1 直达分发器，处理问候、感谢和能力介绍。"""

    def __init__(self, catalog: CapabilityCatalog) -> None:
        """初始化直达规则，catalog 只用于生成能力清单。"""
        self._catalog = catalog
        self._rules: list[tuple[re.Pattern[str], Callable[[ChatRequest], SceneDirectHit]]] = [
            (
                re.compile(r"^(你是(谁|什么)|介绍.*你|你能(做|干).*什么)\s*[？?。.！!]*$"),
                self._handle_introduction,
            ),
            (
                re.compile(r"(支持|能办|有什么|有哪些).*(业务|功能|服务|能力)"),
                self._handle_capability_list,
            ),
            (
                re.compile(r"^(ping|测试|在吗|你好|您好|hi|hello|嗨)[\s！!？?。.~]*$", re.I),
                self._handle_greeting,
            ),
            (
                re.compile(r"^(谢谢|多谢|thanks?|thx|辛苦了|麻烦了|感谢)[\s！!。.~]*$", re.I),
                self._handle_thanks,
            ),
        ]

    def dispatch(self, request: ChatRequest) -> SceneDirectHit | None:
        """命中直达规则时返回结果，否则返回 None。"""
        message = (request.user_message or "").strip()
        if not message:
            return None
        for pattern, handler in self._rules:
            if pattern.search(message):
                return handler(request)
        return None

    def _handle_introduction(self, _request: ChatRequest) -> SceneDirectHit:
        """回答助手身份介绍。"""
        answer = (
            "我是信易贷智能助手，可以帮您：\n"
            f"{self._render_capability_brief()}\n"
            "您可以直接告诉我业务诉求，例如：信易贷的准入条件是什么？"
        )
        return SceneDirectHit(handler_name="introduction", final_answer=answer)

    def _handle_capability_list(self, _request: ChatRequest) -> SceneDirectHit:
        """回答能力清单。"""
        return SceneDirectHit(
            handler_name="capability_list",
            final_answer="我目前支持以下业务能力：\n" + self._render_capability_brief(),
        )

    def _handle_greeting(self, _request: ChatRequest) -> SceneDirectHit:
        """回答问候。"""
        return SceneDirectHit(
            handler_name="greeting",
            final_answer="您好，我是信易贷智能助手。请问有什么可以帮您？",
        )

    def _handle_thanks(self, _request: ChatRequest) -> SceneDirectHit:
        """回答感谢。"""
        return SceneDirectHit(
            handler_name="thanks",
            final_answer="不客气，有任何信易贷相关的问题都可以问我。",
        )

    def _render_capability_brief(self) -> str:
        """从 catalog 动态生成用户可见能力清单。"""
        lines = [f"- {policy.description}" for policy in self._catalog.user_visible_capabilities()]
        return "\n".join(lines)
