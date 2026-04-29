from __future__ import annotations

from typing import Any

from xinyidai_agent.protocol import ChatRequest, RouteDecision


SCENE_DEFAULTS: dict[str, dict[str, Any]] = {
    "KNOWLEDGE_QA": {
        "intent": "POLICY_OR_PRODUCT_QA",
        "risk_level": "read_only",
        "allowed_tools": ["rag_search"],
        "allowed_tool_categories": ["knowledge"],
        "should_call_tool": True,
    },
    "DATA_QUERY": {
        "intent": "CREDIT_LIMIT_QUERY",
        "risk_level": "read_only",
        "required_slots": ["company_name"],
        "allowed_tools": ["query_credit_amount"],
        "allowed_tool_categories": ["data_query"],
        "should_call_tool": True,
    },
    "LOAN_APPLY": {
        "intent": "CREATE_APPLICATION",
        "risk_level": "state_create",
        "required_slots": ["company_name", "product_name"],
        "allowed_tools": ["search_product", "create_application", "create_authorization_link"],
        "allowed_tool_categories": ["knowledge", "application", "authorization"],
        "should_call_tool": True,
    },
    "AUTHORIZATION": {
        "intent": "CREATE_AUTHORIZATION_LINK",
        "risk_level": "link_create",
        "required_slots": ["company_name"],
        "allowed_tools": ["create_authorization_link"],
        "allowed_tool_categories": ["authorization"],
        "should_call_tool": True,
    },
    "APPLICATION_STATUS": {
        "intent": "APPLICATION_STATUS_QUERY",
        "risk_level": "read_only",
        "required_slots": ["application_id"],
        "allowed_tools": ["query_application_status"],
        "allowed_tool_categories": ["status"],
        "should_call_tool": True,
    },
    "SMALLTALK": {
        "intent": "SMALLTALK",
        "risk_level": "read_only",
        "allowed_tools": [],
        "allowed_tool_categories": [],
        "should_call_tool": False,
    },
    "UNKNOWN": {
        "intent": "UNKNOWN",
        "risk_level": "read_only",
        "allowed_tools": [],
        "allowed_tool_categories": [],
        "should_call_tool": False,
    },
}


class RoutePolicy:
    def __init__(self, min_confidence: float = 0.7) -> None:
        self._min_confidence = min_confidence

    def apply(self, request: ChatRequest, route: RouteDecision) -> RouteDecision:
        defaults = SCENE_DEFAULTS.get(route.scene, SCENE_DEFAULTS["UNKNOWN"])
        filled_slots = {
            **route.filled_slots,
            **self._metadata_slots(request, defaults.get("required_slots", [])),
        }
        required_slots = route.required_slots or list(defaults.get("required_slots", []))
        missing_slots = [
            slot for slot in required_slots if not filled_slots.get(slot)
        ]

        if route.confidence < self._min_confidence:
            return RouteDecision(
                scene="UNKNOWN",
                intent="LOW_CONFIDENCE",
                confidence=route.confidence,
                filled_slots=filled_slots,
                missing_slots=["user_intent"],
                route_reason=f"意图识别置信度低于阈值 {self._min_confidence}，需要追问确认。",
                route_source=route.route_source,
                should_call_model=True,
                should_call_tool=False,
            )

        return RouteDecision(
            scene=route.scene,
            intent=route.intent or str(defaults["intent"]),
            confidence=route.confidence,
            required_slots=required_slots,
            filled_slots=filled_slots,
            missing_slots=route.missing_slots or missing_slots,
            allowed_tools=list(defaults.get("allowed_tools", [])),
            allowed_tool_categories=list(defaults.get("allowed_tool_categories", [])),
            risk_level=route.risk_level or str(defaults["risk_level"]),
            route_reason=route.route_reason,
            route_source=route.route_source,
            should_call_model=route.should_call_model,
            should_call_tool=route.should_call_tool or bool(defaults.get("should_call_tool")),
        )

    def _metadata_slots(self, request: ChatRequest, slots: list[str]) -> dict[str, Any]:
        return {
            slot: request.metadata[slot]
            for slot in slots
            if slot in request.metadata and request.metadata[slot]
        }
