from __future__ import annotations

from typing import Protocol

from xinyidai_agent.protocol import ChatRequest, RouteDecision


class IntentRouter(Protocol):
    def route(self, request: ChatRequest) -> RouteDecision:
        """识别业务场景、意图、槽位、风险和可用工具范围。"""
