from __future__ import annotations

from typing import Any

from xinyidai_agent.capabilities import CapabilityResolver
from xinyidai_agent.protocol import ChatRequest, RouteDecision


class RoutePolicy:
    def __init__(
        self,
        min_confidence: float = 0.7,
        capability_resolver: CapabilityResolver | None = None,
    ) -> None:
        self._min_confidence = min_confidence
        self._capability_resolver = capability_resolver or CapabilityResolver()

    def apply(self, request: ChatRequest, route: RouteDecision) -> RouteDecision:
        raw_intent = route.raw_intent or route.intent
        if route.confidence < self._min_confidence:
            return self._clarify(
                route=route,
                raw_intent=raw_intent,
                filled_slots=route.filled_slots,
                reason=f"意图识别置信度低于阈值 {self._min_confidence}，需要追问确认。",
                capability_source="low_confidence",
            )

        resolution = self._capability_resolver.resolve(request, route)
        if resolution.policy is None:
            return self._clarify(
                route=route,
                raw_intent=raw_intent,
                filled_slots=route.filled_slots,
                reason=resolution.reason,
                capability_source=resolution.source,
            )

        capability = resolution.policy
        filled_slots = {
            **route.filled_slots,
            **self._metadata_slots(request, capability.required_slots),
        }
        missing_slots = self._missing_slots(route, capability.required_slots, filled_slots)

        if capability.capability_id == "unknown.clarify":
            return self._clarify(
                route=route,
                raw_intent=raw_intent,
                filled_slots=filled_slots,
                reason=route.route_reason,
                capability_source=resolution.source,
            )

        return RouteDecision(
            scene=capability.scene,
            intent=capability.standard_intent,
            raw_intent=raw_intent,
            capability_id=capability.capability_id,
            capability_source=resolution.source,
            confirmation_required=capability.confirmation_required,
            confidence=route.confidence,
            required_slots=capability.required_slots,
            filled_slots=filled_slots,
            missing_slots=missing_slots,
            allowed_tools=capability.allowed_tools,
            allowed_tool_categories=capability.allowed_tool_categories,
            risk_level=capability.risk_level,
            route_reason=route.route_reason,
            route_source=route.route_source,
            should_call_model=route.should_call_model,
            should_call_tool=bool(capability.allowed_tools),
        )

    def _clarify(
        self,
        route: RouteDecision,
        raw_intent: str,
        filled_slots: dict[str, Any],
        reason: str,
        capability_source: str,
    ) -> RouteDecision:
        return RouteDecision(
            scene="UNKNOWN",
            intent="LOW_CONFIDENCE",
            raw_intent=raw_intent,
            capability_id="unknown.clarify",
            capability_source=capability_source,
            confidence=route.confidence,
            filled_slots=filled_slots,
            missing_slots=["user_intent"],
            allowed_tools=[],
            allowed_tool_categories=[],
            risk_level="read_only",
            route_reason=reason,
            route_source=route.route_source,
            should_call_model=True,
            should_call_tool=False,
        )

    def _metadata_slots(self, request: ChatRequest, slots: list[str]) -> dict[str, Any]:
        return {
            slot: request.metadata[slot]
            for slot in slots
            if slot in request.metadata and request.metadata[slot]
        }

    def _missing_slots(
        self,
        route: RouteDecision,
        required_slots: list[str],
        filled_slots: dict[str, Any],
    ) -> list[str]:
        missing: list[str] = []
        for slot in required_slots:
            if not filled_slots.get(slot):
                missing.append(slot)
        for slot in route.missing_slots:
            if slot not in required_slots and not filled_slots.get(slot):
                missing.append(slot)
        return missing
