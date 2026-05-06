from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from xinyidai_agent.policies import RiskPolicy
from xinyidai_agent.protocol import PendingAction, RouteDecision, SessionStateSnapshot, ToolCall
from xinyidai_agent.runtime_state import build_pending_action_hash


@dataclass(frozen=True)
class ActionValidationResult:
    allowed: bool
    reason: str
    code: str
    tool_call: ToolCall | None = None
    details: dict[str, Any] = field(default_factory=dict)


class PendingActionValidator:
    """确认后的工具执行前校验 pending action snapshot 是否仍与当前状态一致。"""

    def __init__(self, risk_policy: RiskPolicy | None = None) -> None:
        self._risk_policy = risk_policy or RiskPolicy()

    def validate_pending_action_snapshot(
        self,
        *,
        state: SessionStateSnapshot,
        route: RouteDecision,
        action: PendingAction,
        tool_call: ToolCall,
        spec: Any,
        now: datetime | None = None,
    ) -> ActionValidationResult:
        if state.pending_action is None:
            return self._blocked("PENDING_ACTION_MISSING", "当前会话没有待确认动作。")

        if action.action_id != state.pending_action.action_id:
            return self._blocked("ACTION_ID_MISMATCH", "确认动作与会话中的待确认动作不一致。")

        if action.session_id != state.session_id:
            return self._blocked("SESSION_MISMATCH", "确认动作所属会话不匹配。")

        if action.scene != route.scene:
            return self._blocked("SCENE_MISMATCH", "确认动作场景与当前路由不一致。")

        if action.capability_id != route.capability_id:
            return self._blocked("CAPABILITY_MISMATCH", "确认动作能力与当前路由不一致。")

        if action.stage and state.current_stage and action.stage != state.current_stage:
            return self._blocked("STAGE_MISMATCH", "确认动作阶段与当前会话阶段不一致。")

        if action.tool_call.tool_name != tool_call.tool_name:
            return self._blocked("TOOL_NAME_MISMATCH", "确认动作工具与当前执行工具不一致。")

        if action.tool_call.arguments != tool_call.arguments:
            return self._blocked("TOOL_ARGUMENTS_MISMATCH", "确认动作参数与当前执行参数不一致。")

        if action.risk_level != spec.risk_level or tool_call.risk_level != spec.risk_level:
            return self._blocked("RISK_LEVEL_MISMATCH", "确认动作风险等级与工具声明不一致。")

        if self._risk_policy.requires_confirmation(spec.risk_level) and state.confirmation_status != "confirmed":
            return self._blocked("CONFIRMATION_REQUIRED", "该动作尚未获得用户确认。")

        conflict = self._slot_snapshot_conflict(state, route, tool_call, action)
        if conflict:
            return self._blocked(
                "SLOT_SNAPSHOT_CONFLICT",
                "确认动作的业务快照与当前会话槽位冲突。",
                {"slot": conflict},
            )

        if action.expires_at:
            expires_at = datetime.fromisoformat(action.expires_at)
            current = now or datetime.now(UTC)
            if expires_at < current:
                return self._blocked("ACTION_EXPIRED", "待确认动作已过期。")

        expected_hash = build_pending_action_hash(
            session_id=action.session_id,
            scene=action.scene,
            capability_id=action.capability_id,
            stage=action.stage,
            slot_snapshot=action.slot_snapshot,
            tool_name=action.tool_call.tool_name,
        )
        if action.precondition_hash != expected_hash:
            return self._blocked("PRECONDITION_HASH_MISMATCH", "确认动作前置条件 hash 不一致。")

        return ActionValidationResult(True, "确认动作快照校验通过。", "OK", tool_call)

    def _slot_snapshot_conflict(
        self,
        state: SessionStateSnapshot,
        route: RouteDecision,
        tool_call: ToolCall,
        action: PendingAction,
    ) -> str | None:
        for slot, snapshot_value in action.slot_snapshot.items():
            state_value = state.confirmed_slots.get(slot)
            if state_value is not None and state_value != snapshot_value:
                return slot
            route_value = route.filled_slots.get(slot)
            if route_value is not None and route_value != snapshot_value:
                return slot
        return None

    def _blocked(
        self,
        code: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> ActionValidationResult:
        return ActionValidationResult(False, reason, code, details=details or {})
