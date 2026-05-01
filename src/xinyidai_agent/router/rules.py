from __future__ import annotations

import re

from xinyidai_agent.protocol import ChatRequest, RouteDecision


class RuleBasedRouter:
    """系统级本地守卫，只处理无业务语义输入和会话续接。"""

    def match(self, request: ChatRequest) -> RouteDecision | None:
        message = request.user_message.strip()
        resume_route = self._resume_route(request)
        if resume_route is not None:
            return resume_route

        if self._is_low_information(message):
            return unknown_route(
                request,
                reason="用户输入缺少可识别的业务语义，直接追问确认意图。",
                confidence=0.2,
            )

        return None

    def _is_low_information(self, message: str) -> bool:
        if len(message) < 2:
            return True
        return re.fullmatch(r"[\W\d_]+", message, flags=re.UNICODE) is not None

    def _resume_route(self, request: ChatRequest) -> RouteDecision | None:
        raw_route = request.metadata.get("_session_resume_route")
        if not isinstance(raw_route, dict):
            return None

        payload = dict(raw_route)
        filled_slots = dict(payload.get("filled_slots") or {})
        required_slots = list(payload.get("required_slots") or [])
        missing_slots = list(payload.get("missing_slots") or [])
        for slot in [*required_slots, *missing_slots]:
            if request.metadata.get(slot):
                filled_slots[slot] = request.metadata[slot]

        payload["filled_slots"] = filled_slots
        payload["missing_slots"] = []
        payload["route_source"] = "session_memory"
        payload["route_reason"] = "根据上一轮待补槽位恢复业务流程。"
        return RouteDecision.model_validate(payload)


def default_knowledge_route(request: ChatRequest, reason: str = "未配置模型路由，回退到知识问答。") -> RouteDecision:
    return RouteDecision(
        scene="KNOWLEDGE_QA",
        intent="POLICY_OR_PRODUCT_QA",
        raw_intent="POLICY_OR_PRODUCT_QA",
        confidence=0.8,
        allowed_tools=["rag_search"],
        allowed_tool_categories=["knowledge"],
        risk_level="read_only",
        route_reason=reason,
        route_source="fallback",
        should_call_model=True,
        should_call_tool=True,
    )


def unknown_route(request: ChatRequest, reason: str, confidence: float = 0.0) -> RouteDecision:
    return RouteDecision(
        scene="UNKNOWN",
        intent="UNKNOWN",
        raw_intent="UNKNOWN",
        confidence=confidence,
        missing_slots=["user_intent"],
        allowed_tools=[],
        allowed_tool_categories=[],
        risk_level="read_only",
        route_reason=reason,
        route_source="local_guard",
        should_call_model=True,
        should_call_tool=False,
    )
