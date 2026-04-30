from __future__ import annotations

import re

from xinyidai_agent.protocol import ChatRequest, RouteDecision


class RuleBasedRouter:
    """规则优先路由器，处理确定性强的业务表达。"""

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

        if any(keyword in message for keyword in ("申请", "办理", "贷款链接")):
            filled_slots = {
                slot: request.metadata[slot]
                for slot in ("company_name", "product_name")
                if request.metadata.get(slot)
            }
            if "product_name" not in filled_slots and "小微税贷" in message:
                filled_slots["product_name"] = "小微税贷"
            return RouteDecision(
                scene="LOAN_APPLY",
                intent="CREATE_APPLICATION",
                raw_intent="CREATE_APPLICATION",
                confidence=0.88,
                required_slots=["company_name", "product_name"],
                filled_slots=filled_slots,
                allowed_tools=["search_product", "create_application", "create_authorization_link"],
                allowed_tool_categories=["knowledge", "application", "authorization"],
                risk_level="state_create",
                route_reason="规则命中贷款申请关键词，创建申请前必须进行执行确认。",
                route_source="rule",
                should_call_model=True,
                should_call_tool=True,
            )

        if any(keyword in message for keyword in ("额度", "能贷", "多少钱", "授信")):
            filled_slots = {}
            if request.metadata.get("company_name"):
                filled_slots["company_name"] = request.metadata["company_name"]
            return RouteDecision(
                scene="DATA_QUERY",
                intent="CREDIT_LIMIT_QUERY",
                raw_intent="CREDIT_LIMIT_QUERY",
                confidence=0.91,
                required_slots=["company_name"],
                filled_slots=filled_slots,
                allowed_tools=["query_credit_amount"],
                allowed_tool_categories=["data_query"],
                risk_level="read_only",
                route_reason="规则命中授信额度查询关键词，数值类答案必须通过只读工具查询。",
                route_source="rule",
                should_call_model=True,
                should_call_tool=True,
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


def default_knowledge_route(request: ChatRequest, reason: str = "未命中强规则，回退到知识问答。") -> RouteDecision:
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
