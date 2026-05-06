from __future__ import annotations

from typing import Any

from xinyidai_agent.capabilities import CapabilityResolver
from xinyidai_agent.policies import RiskPolicy
from xinyidai_agent.protocol import ChatRequest, RouteDecision, SessionStateSnapshot, ToolCall
from xinyidai_agent.router.action_validator import ActionValidationResult, PendingActionValidator
from xinyidai_agent.tools.base import ToolSpec


class RoutePolicy:
    def __init__(
        self,
        min_confidence: float = 0.7,
        capability_resolver: CapabilityResolver | None = None,
    ) -> None:
        self._min_confidence = min_confidence
        self._capability_resolver = capability_resolver or CapabilityResolver()
        self._pending_action_validator = PendingActionValidator()
        self._risk_policy = RiskPolicy()

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

        resolution = self._capability_resolver.resolve(route)
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
        missing_slots = self._missing_slots(capability.required_slots, filled_slots)

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
            confirmation_required=self._risk_policy.requires_confirmation(
                capability.risk_level,
                capability.confirmation_required,
            ),
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

    def validate_tool_action(
        self,
        route: RouteDecision,
        tool_call: ToolCall,
        tool_specs: list[ToolSpec],
        session_state: SessionStateSnapshot | None = None,
        allow_unconfirmed: bool = False,
    ) -> ActionValidationResult:
        """统一校验工具动作是否仍处于当前路由和会话状态允许范围内。"""
        spec_map = {spec.name: spec for spec in tool_specs}
        spec = spec_map.get(tool_call.tool_name)
        if spec is None:
            return ActionValidationResult(False, f"工具未在当前分类中声明：{tool_call.tool_name}", "TOOL_NOT_DECLARED")

        if tool_call.tool_name not in route.allowed_tools:
            return ActionValidationResult(False, f"工具不在当前能力允许列表中：{tool_call.tool_name}", "TOOL_NOT_ALLOWED")

        if spec.category not in route.allowed_tool_categories:
            return ActionValidationResult(False, f"工具分类不在当前能力允许列表中：{spec.category}", "CATEGORY_NOT_ALLOWED")

        if tool_call.tool_category and tool_call.tool_category != spec.category:
            return ActionValidationResult(
                False,
                f"工具分类不匹配：调用方={tool_call.tool_category}，策略={spec.category}",
                "TOOL_CATEGORY_MISMATCH",
            )

        if tool_call.risk_level != spec.risk_level:
            return ActionValidationResult(
                False,
                f"工具风险等级不匹配：调用方={tool_call.risk_level}，策略={spec.risk_level}",
                "RISK_LEVEL_MISMATCH",
            )

        missing_route_slots = [slot for slot in route.required_slots if not route.filled_slots.get(slot)]
        if missing_route_slots:
            return ActionValidationResult(False, f"缺少能力必填槽位：{', '.join(missing_route_slots)}", "ROUTE_SLOTS_MISSING")

        input_errors = self._validate_input_slots(tool_call.arguments, spec)
        if input_errors:
            return ActionValidationResult(False, f"工具入参校验失败：{'; '.join(input_errors)}", "TOOL_INPUT_INVALID")

        needs_confirmation = self._risk_policy.requires_confirmation(
            spec.risk_level,
            spec.requires_confirmation or route.confirmation_required or tool_call.confirmation_required,
        )
        if needs_confirmation and not allow_unconfirmed:
            if session_state is None or session_state.pending_action is None:
                return ActionValidationResult(False, "确认状态缺少 pending_action 快照。", "PENDING_ACTION_MISSING")
            return self._pending_action_validator.validate_pending_action_snapshot(
                state=session_state,
                route=route,
                action=session_state.pending_action,
                tool_call=tool_call,
                spec=spec,
            )

        return ActionValidationResult(True, "工具动作通过策略校验。", "OK", tool_call)

    def _metadata_slots(self, request: ChatRequest, slots: list[str]) -> dict[str, Any]:
        return {
            slot: request.metadata[slot]
            for slot in slots
            if slot in request.metadata and request.metadata[slot]
        }

    def _missing_slots(
        self,
        required_slots: list[str],
        filled_slots: dict[str, Any],
    ) -> list[str]:
        missing: list[str] = []
        for slot in required_slots:
            if not filled_slots.get(slot):
                missing.append(slot)
        return missing

    def _validate_input_slots(self, arguments: dict[str, Any], spec: ToolSpec) -> list[str]:
        errors: list[str] = []
        for slot in spec.input_slots:
            value = arguments.get(slot.name)
            if slot.required and not self._has_value(value, slot.allow_empty):
                errors.append(f"{slot.name} 缺失")
                continue
            if not self._has_value(value, slot.allow_empty):
                continue
            if not self._matches_type(value, slot.value_type):
                errors.append(f"{slot.name} 类型应为 {slot.value_type}，实际为 {type(value).__name__}")
        return errors

    def _has_value(self, value: object, allow_empty: bool) -> bool:
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        if isinstance(value, list | dict | tuple | set) and not value:
            return allow_empty
        return True

    def _matches_type(self, value: object, value_type: str) -> bool:
        if value_type == "string":
            return isinstance(value, str)
        if value_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if value_type == "number":
            return isinstance(value, int | float) and not isinstance(value, bool)
        if value_type == "boolean":
            return isinstance(value, bool)
        if value_type == "object":
            return isinstance(value, dict)
        if value_type == "array":
            return isinstance(value, list)
        return True
