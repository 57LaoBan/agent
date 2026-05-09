"""ReAct 单步执行器：模型 reasoning、规则 guardrails 和工具执行。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Literal
from uuid import uuid4

from xinyidai_agent.llm import JSON_OBJECT_RESPONSE_FORMAT, ChatModel
from xinyidai_agent.policies import RiskPolicy
from xinyidai_agent.protocol import ReactAction, ReactStepDecision, ToolCall, ToolExecutionObservation
from xinyidai_agent.react_observation import ReactObservation
from xinyidai_agent.runtime.react_prompt import ReactPromptBuilder
from xinyidai_agent.tools.base import ToolExecution, ToolSpec
from xinyidai_agent.tools.registry import ToolRegistry

StepKind = Literal["call_tool_executed", "terminal", "invalid_action"]


@dataclass(frozen=True)
class StepOutcome:
    """ReactStepEngine 单步执行结果。"""

    kind: StepKind
    decision: ReactStepDecision
    tool_observation: ToolExecutionObservation | None = None
    tool_execution: ToolExecution | None = None
    invalid_hint: str | None = None
    pending_action_required: bool = False
    pending_tool_call: ToolCall | None = None


class ReactStepEngine:
    """单步 ReAct 执行：reason → guardrail → execute。"""

    _REPEAT_LIMIT = 2

    def __init__(
        self,
        model: ChatModel,
        tool_registry: ToolRegistry,
        prompt_builder: ReactPromptBuilder | None = None,
        risk_policy: RiskPolicy | None = None,
        max_json_retries: int = 1,
    ) -> None:
        """初始化模型、工具注册表和策略依赖。"""
        self._model = model
        self._tools = tool_registry
        self._prompt = prompt_builder or ReactPromptBuilder()
        self._risk = risk_policy or RiskPolicy()
        self._max_json_retries = max_json_retries

    def step(self, observation: ReactObservation) -> StepOutcome:
        """执行一次 ReAct 单步。"""
        decision = self._reason(observation)
        action = decision.action
        if action.type == "call_tool":
            return self._handle_call_tool(observation, decision)
        if action.type == "answer":
            return self._validate_answer(observation, decision)
        if action.type == "ask_user":
            return self._validate_ask_user(decision)
        if action.type == "handoff":
            return self._validate_handoff(decision)
        return self._reject(decision, "action.type 必须是 call_tool / answer / ask_user / handoff。")

    def _reason(self, observation: ReactObservation) -> ReactStepDecision:
        """调用模型并解析为 ReactStepDecision，JSON 错误最多修复重试一次。"""
        messages = self._prompt.build_messages(observation)
        last_error = ""
        for attempt in range(self._max_json_retries + 1):
            raw = self._model.complete(messages, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            try:
                payload = self._parse_json(raw)
                return ReactStepDecision.model_validate(payload)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                last_error = str(exc)
                if attempt < self._max_json_retries:
                    messages = self._prompt.build_messages(
                        observation.model_copy(
                            update={
                                "invalid_action_hint": (
                                    f"上一次输出不是合法 JSON：{last_error}，请严格按 schema 输出。"
                                )
                            }
                        )
                    )
        raise RuntimeError(f"模型 JSON 输出无法解析：{last_error}")

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """从模型输出中提取 JSON 对象。"""
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"未找到 JSON 对象：{raw[:200]}")
        return json.loads(text[start : end + 1])

    def _handle_call_tool(self, observation: ReactObservation, decision: ReactStepDecision) -> StepOutcome:
        """校验工具调用动作并在允许时执行工具。"""
        action = decision.action
        if not action.tool_name:
            return self._reject(decision, "选择 call_tool 时必须给出 tool_name。")
        if action.tool_name not in observation.capability.allowed_tools:
            return self._reject(
                decision,
                (
                    f"{action.tool_name} 不在当前能力 {observation.capability.capability_id} 的工具白名单："
                    f"{observation.capability.allowed_tools}"
                ),
            )
        repeat = self._count_same_calls(observation.executed_tools, action)
        if repeat >= self._REPEAT_LIMIT:
            return self._reject(
                decision,
                f"同 (tool_name, arguments) 已执行 {repeat} 次，请改用 answer / ask_user / handoff。",
            )

        spec = self._tools.spec_for_name(action.tool_name)
        if spec is None:
            tool_call = self._build_tool_call(action, None, observation)
            execution, tool_observation = self._tools.execute_observed(
                observation.request,
                observation.route,
                tool_call,
                capability=observation.capability,
                session_state=observation.session_state,
            )
            return StepOutcome(
                kind="call_tool_executed",
                decision=decision,
                tool_observation=tool_observation,
                tool_execution=execution,
            )

        if self._risk.requires_handoff(spec.risk_level):
            return self._reject(
                decision,
                f"{action.tool_name} 风险级别 {spec.risk_level} 必须人工处理，请改用 handoff。",
            )
        if self._risk.requires_confirmation(spec.risk_level, spec.requires_confirmation):
            tool_call = self._build_tool_call(action, spec, observation)
            return StepOutcome(
                kind="terminal",
                decision=decision,
                pending_action_required=True,
                pending_tool_call=tool_call,
            )

        tool_call = self._build_tool_call(action, spec, observation)
        execution, tool_observation = self._tools.execute_observed(
            observation.request,
            observation.route,
            tool_call,
            capability=observation.capability,
            session_state=observation.session_state,
        )
        return StepOutcome(
            kind="call_tool_executed",
            decision=decision,
            tool_observation=tool_observation,
            tool_execution=execution,
        )

    def _validate_answer(self, observation: ReactObservation, decision: ReactStepDecision) -> StepOutcome:
        """校验 answer 是否满足证据要求。"""
        action = decision.action
        if not action.final_answer or not action.final_answer.strip():
            return self._reject(decision, "选择 answer 时 final_answer 不能为空。")
        if observation.capability.requires_evidence:
            success_observations = [
                item
                for item in observation.executed_tools
                if item.kind == "success" and item.sources_count > 0
            ]
            if not success_observations:
                return self._reject(
                    decision,
                    "当前能力要求 answer 必须基于工具检索到的证据，请先 call_tool 拿到 sources。",
                )
            if not action.evidence_used:
                return self._reject(decision, "answer 必须填 evidence_used，标明引用了哪些 source_id。")
        return StepOutcome(kind="terminal", decision=decision)

    def _validate_ask_user(self, decision: ReactStepDecision) -> StepOutcome:
        """校验 ask_user 消息。"""
        if not decision.action.message:
            return self._reject(decision, "选择 ask_user 时 message 不能为空。")
        return StepOutcome(kind="terminal", decision=decision)

    def _validate_handoff(self, decision: ReactStepDecision) -> StepOutcome:
        """校验 handoff 消息。"""
        if not decision.action.message:
            return self._reject(decision, "选择 handoff 时 message 不能为空。")
        return StepOutcome(kind="terminal", decision=decision)

    @staticmethod
    def _reject(decision: ReactStepDecision, hint: str) -> StepOutcome:
        """构造 invalid_action 结果。"""
        return StepOutcome(kind="invalid_action", decision=decision, invalid_hint=hint)

    @staticmethod
    def _count_same_calls(executed: list[ToolExecutionObservation], action: ReactAction) -> int:
        """统计同一工具名和参数的历史执行次数。"""
        return sum(
            1
            for observation in executed
            if observation.tool_name == action.tool_name and observation.arguments == action.arguments
        )

    def _build_tool_call(
        self,
        action: ReactAction,
        spec: ToolSpec | None,
        observation: ReactObservation,
    ) -> ToolCall:
        """从模型 action 构造受控 ToolCall。"""
        return ToolCall(
            tool_call_id=str(uuid4()),
            tool_name=action.tool_name or "",
            tool_category=spec.category if spec is not None else None,
            arguments=dict(action.arguments),
            risk_level=spec.risk_level if spec is not None else "read_only",
            confirmation_required=spec.requires_confirmation if spec is not None else False,
            reason=f"react_step_{observation.step}",
        )
