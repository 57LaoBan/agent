from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from xinyidai_agent.policies import RiskPolicy
from xinyidai_agent.protocol import (
    ChatRequest,
    PendingAction,
    RouteDecision,
    RuntimeStepDecision,
    SessionStateSnapshot,
    SourceDocument,
    ToolResult,
)


class RuntimeStepContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_index: int
    max_steps: int
    request: ChatRequest
    session_state: SessionStateSnapshot
    route: RouteDecision
    last_tool_result: ToolResult | None = None
    last_sources: list[SourceDocument] = Field(default_factory=list)
    pending_action: PendingAction | None = None
    executed_tool_names: list[str] = Field(default_factory=list)


class RuntimeStepPolicy:
    """硬编码 runtime step 安全规则，不把下一步裁决交给模型。"""

    def __init__(self, min_confidence: float = 0.7) -> None:
        self._min_confidence = min_confidence
        self._risk_policy = RiskPolicy()

    def decide_next_step(self, context: RuntimeStepContext) -> RuntimeStepDecision:
        route = context.route
        last = context.last_tool_result

        if context.step_index > context.max_steps:
            return self._stop("达到最大受控步数。")

        if route.missing_slots:
            return RuntimeStepDecision(
                step_type="ASK_USER",
                reason="缺少必要业务槽位。",
                confidence=route.confidence,
                missing_slots=route.missing_slots,
            )

        if route.confidence < self._min_confidence:
            return RuntimeStepDecision(
                step_type="ASK_USER",
                reason=f"路由置信度低于阈值 {self._min_confidence}。",
                confidence=route.confidence,
                missing_slots=["user_intent"],
            )

        if context.pending_action is not None or context.session_state.confirmation_status == "waiting":
            return RuntimeStepDecision(
                step_type="WAIT_CONFIRMATION",
                reason="当前会话存在待确认动作，不能继续自动推进。",
                confidence=route.confidence,
                pending_action=context.pending_action or context.session_state.pending_action,
            )

        if last is not None:
            return self._after_tool_result(context, last)

        if self._risk_policy.requires_handoff(route.risk_level):
            return RuntimeStepDecision(
                step_type="HANDOFF",
                reason=f"{route.risk_level} 风险动作默认转人工或强确认，不自动执行。",
                confidence=route.confidence,
                handoff_reason=f"{route.risk_level}_requires_handoff",
            )

        if self._risk_policy.requires_confirmation(route.risk_level, route.confirmation_required):
            return RuntimeStepDecision(
                step_type="PROPOSE_PENDING_ACTION",
                reason="非只读或需要确认的业务动作必须先生成 pending_action。",
                confidence=route.confidence,
            )

        repeated = self._would_repeat_only_allowed_tool(route.allowed_tools, context.executed_tool_names)
        if repeated:
            return self._stop("当前能力只剩已执行过的工具，阻断重复执行。")

        if route.should_call_tool and route.allowed_tools:
            return RuntimeStepDecision(
                step_type="PROPOSE_TOOL",
                reason="只读能力允许规划并执行工具。",
                confidence=route.confidence,
            )

        return RuntimeStepDecision(
            step_type="ANSWER_WITHOUT_TOOL",
            reason="当前能力不需要工具。",
            confidence=route.confidence,
        )

    def _after_tool_result(
        self,
        context: RuntimeStepContext,
        result: ToolResult,
    ) -> RuntimeStepDecision:
        if result.status != "success":
            return RuntimeStepDecision(
                step_type="GENERATE_FINAL_ANSWER",
                reason="工具失败，生成安全提示。",
                confidence=context.route.confidence,
            )

        if result.business_status in {"PARTIAL_DATA", "NOT_FOUND"}:
            return RuntimeStepDecision(
                step_type="GENERATE_FINAL_ANSWER",
                reason="证据不足或未找到结果，不能编造。",
                confidence=context.route.confidence,
            )

        if result.terminal:
            return RuntimeStepDecision(
                step_type="GENERATE_FINAL_ANSWER",
                reason="工具结果为 terminal，生成最终回答。",
                confidence=context.route.confidence,
            )

        if not result.allowed_next_tools:
            return RuntimeStepDecision(
                step_type="GENERATE_FINAL_ANSWER",
                reason="工具未声明允许的下一步工具，安全收敛。",
                confidence=context.route.confidence,
            )

        repeated_next = all(tool in context.executed_tool_names for tool in result.allowed_next_tools)
        if repeated_next:
            return self._stop("下一步候选工具均已执行过，阻断重复执行。")

        return RuntimeStepDecision(
            step_type="PROPOSE_TOOL",
            reason="工具结果声明了同能力内允许的下一步只读工具。",
            confidence=context.route.confidence,
        )

    def _would_repeat_only_allowed_tool(
        self,
        allowed_tools: list[str],
        executed_tool_names: list[str],
    ) -> bool:
        return len(allowed_tools) == 1 and allowed_tools[0] in executed_tool_names

    def _stop(self, reason: str) -> RuntimeStepDecision:
        return RuntimeStepDecision(step_type="STOP", reason=reason, confidence=0.0)
