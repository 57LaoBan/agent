from __future__ import annotations

from dataclasses import replace

import pytest

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.capabilities.catalog import default_capability_catalog
from xinyidai_agent.protocol import (
    ChatRequest,
    RouteDecision,
    SessionStateSnapshot,
    ToolCall,
    ToolExecutionObservation,
    ToolResult,
    ToolResultEnvelope,
)
from xinyidai_agent.react_observation import ReactObservation
from xinyidai_agent.runtime.react_observation_builder import ReactObservationBuilder
from xinyidai_agent.runtime.react_engine import ReactStepEngine
from xinyidai_agent.tools.base import SlotSpec, ToolExecution, ToolSpec
from xinyidai_agent.tools.registry import ToolRegistry, default_tool_registry


class SequenceModel:
    """按顺序返回固定模型输出。"""

    def __init__(self, outputs: list[str]) -> None:
        """初始化输出队列。"""
        self.outputs = outputs
        self.calls = 0

    def complete(self, messages, response_format=None) -> str:
        """返回下一条模型输出。"""
        self.calls += 1
        index = min(self.calls - 1, len(self.outputs) - 1)
        return self.outputs[index]


class CreateApplicationTool:
    """测试用申请创建工具。"""

    name = "create_application"
    category = "application"
    risk_level = "state_create"
    description = "创建申请草稿。"
    requires_confirmation = True

    def spec(self) -> ToolSpec:
        """返回工具规格。"""
        return ToolSpec(
            name=self.name,
            category=self.category,
            risk_level=self.risk_level,
            description=self.description,
            requires_confirmation=self.requires_confirmation,
            is_read_only=False,
            is_idempotent=False,
            is_concurrency_safe=False,
            input_slots=[SlotSpec("company_name", "string"), SlotSpec("product_name", "string")],
        )

    def execute(self, request: ChatRequest, route: RouteDecision, tool_call: ToolCall) -> ToolExecution:
        """确认前不应直接执行。"""
        result = ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            tool_category=self.category,
            status="success",
            output={"application_id": "APP-001"},
            envelope=ToolResultEnvelope(success=True, status="APPLICATION_DRAFT_CREATED"),
            business_status="APPLICATION_DRAFT_CREATED",
        )
        return ToolExecution(result=result)


class FinalSubmitTool(CreateApplicationTool):
    """测试用最终提交工具。"""

    name = "final_submit"
    risk_level = "final_submit"
    description = "最终提交申请。"


def _json(action: str) -> str:
    """包装模型 JSON 输出。"""
    return f'{{"thought":"t","action":{action},"confidence":0.9}}'


def _capability(capability_id: str) -> CapabilityPolicy:
    """按 ID 读取默认 capability。"""
    return {policy.capability_id: policy for policy in default_capability_catalog()}[capability_id]


def _route(capability: CapabilityPolicy, filled_slots: dict | None = None) -> RouteDecision:
    """构造 route。"""
    return RouteDecision(
        scene=capability.scene,
        intent=capability.standard_intent,
        confidence=0.95,
        filled_slots=filled_slots or {},
        allowed_tools=capability.allowed_tools,
        allowed_tool_categories=capability.allowed_tool_categories,
        risk_level=capability.risk_level,
        route_reason="测试路由。",
        should_call_tool=bool(capability.allowed_tools),
    )


def _session() -> SessionStateSnapshot:
    """构造最小会话状态。"""
    return SessionStateSnapshot(session_id="s-1", updated_at="2026-05-09T00:00:00Z")


def _observation(
    capability: CapabilityPolicy,
    registry: ToolRegistry,
    executed: list[ToolExecutionObservation] | None = None,
) -> ReactObservation:
    """构造 ReactObservation。"""
    return ReactObservationBuilder(registry).build(
        step=1,
        max_steps=capability.max_react_steps,
        request=ChatRequest(user_message="测试问题"),
        route=_route(capability, {"company_name": "杭州示例科技有限公司", "product_name": "小微税贷"}),
        capability=capability,
        session_state=_session(),
        executed_tools=executed or [],
        last_observation=(executed or [None])[-1],
    )


def test_call_tool_executed_when_valid() -> None:
    """合法工具调用会执行并返回 observation。"""
    model = SequenceModel([
        _json('{"type":"call_tool","tool_name":"query_credit_amount","arguments":{"company_name":"杭州示例科技有限公司"}}')
    ])
    engine = ReactStepEngine(model, default_tool_registry())

    outcome = engine.step(_observation(_capability("credit.limit.read"), default_tool_registry()))

    assert outcome.kind == "call_tool_executed"
    assert outcome.tool_observation is not None
    assert outcome.tool_observation.kind == "success"


def test_call_tool_unavailable_returns_call_tool_executed_with_obs() -> None:
    """工具未注册仍返回 call_tool_executed 和 unavailable observation。"""
    model = SequenceModel([
        _json('{"type":"call_tool","tool_name":"query_product_terms","arguments":{"query":"利率"}}')
    ])
    registry = default_tool_registry()
    engine = ReactStepEngine(model, registry)

    outcome = engine.step(_observation(_capability("product.terms.read"), registry))

    assert outcome.kind == "call_tool_executed"
    assert outcome.tool_observation is not None
    assert outcome.tool_observation.kind == "unavailable"


def test_call_tool_blocked_when_not_in_whitelist() -> None:
    """工具不在 capability 白名单时被 engine 阻断。"""
    model = SequenceModel([
        _json('{"type":"call_tool","tool_name":"query_credit_amount","arguments":{"company_name":"杭州示例科技有限公司"}}')
    ])
    engine = ReactStepEngine(model, default_tool_registry())

    outcome = engine.step(_observation(_capability("knowledge.policy.read"), default_tool_registry()))

    assert outcome.kind == "invalid_action"
    assert "白名单" in (outcome.invalid_hint or "")


def test_call_tool_blocked_when_repeat_limit_reached() -> None:
    """同一工具和参数重复两次后硬阻断。"""
    arguments = {"company_name": "杭州示例科技有限公司"}
    executed = [
        ToolExecutionObservation(kind="success", tool_name="query_credit_amount", arguments=arguments),
        ToolExecutionObservation(kind="success", tool_name="query_credit_amount", arguments=arguments),
    ]
    model = SequenceModel([
        _json('{"type":"call_tool","tool_name":"query_credit_amount","arguments":{"company_name":"杭州示例科技有限公司"}}')
    ])
    engine = ReactStepEngine(model, default_tool_registry())

    outcome = engine.step(_observation(_capability("credit.limit.read"), default_tool_registry(), executed))

    assert outcome.kind == "invalid_action"
    assert "已执行 2 次" in (outcome.invalid_hint or "")


def test_answer_requires_evidence_when_capability_requires_it() -> None:
    """requires_evidence=True 时没有成功证据不能 answer。"""
    model = SequenceModel([_json('{"type":"answer","final_answer":"可以办理。","evidence_used":["policy-1"]}')])
    engine = ReactStepEngine(model, default_tool_registry())

    outcome = engine.step(_observation(_capability("knowledge.policy.read"), default_tool_registry()))

    assert outcome.kind == "invalid_action"
    assert "请先 call_tool" in (outcome.invalid_hint or "")


def test_answer_requires_non_empty_final_answer() -> None:
    """answer 必须包含非空 final_answer。"""
    model = SequenceModel([_json('{"type":"answer","final_answer":"  ","evidence_used":["policy-1"]}')])
    engine = ReactStepEngine(model, default_tool_registry())

    outcome = engine.step(_observation(_capability("credit.limit.read"), default_tool_registry()))

    assert outcome.kind == "invalid_action"
    assert "final_answer" in (outcome.invalid_hint or "")


def test_answer_requires_evidence_used_when_required() -> None:
    """requires_evidence=True 时 answer 必须填 evidence_used。"""
    executed = [
        ToolExecutionObservation(kind="success", tool_name="rag_search", sources_count=1, summary="命中证据")
    ]
    model = SequenceModel([_json('{"type":"answer","final_answer":"可以办理。","evidence_used":[]}')])
    engine = ReactStepEngine(model, default_tool_registry())

    outcome = engine.step(_observation(_capability("knowledge.policy.read"), default_tool_registry(), executed))

    assert outcome.kind == "invalid_action"
    assert "evidence_used" in (outcome.invalid_hint or "")


def test_ask_user_requires_non_empty_message() -> None:
    """ask_user 必须包含 message。"""
    model = SequenceModel([_json('{"type":"ask_user","message":""}')])
    engine = ReactStepEngine(model, default_tool_registry())

    outcome = engine.step(_observation(_capability("credit.limit.read"), default_tool_registry()))

    assert outcome.kind == "invalid_action"
    assert "message" in (outcome.invalid_hint or "")


def test_handoff_requires_non_empty_message() -> None:
    """handoff 必须包含 message。"""
    model = SequenceModel([_json('{"type":"handoff","message":""}')])
    engine = ReactStepEngine(model, default_tool_registry())

    outcome = engine.step(_observation(_capability("credit.limit.read"), default_tool_registry()))

    assert outcome.kind == "invalid_action"
    assert "message" in (outcome.invalid_hint or "")


def test_invalid_json_retried_once_then_raises() -> None:
    """非法 JSON 会重试一次，仍失败则抛 RuntimeError。"""
    model = SequenceModel(["not-json", "still-not-json"])
    engine = ReactStepEngine(model, default_tool_registry(), max_json_retries=1)

    with pytest.raises(RuntimeError):
        engine.step(_observation(_capability("credit.limit.read"), default_tool_registry()))

    assert model.calls == 2


def test_call_tool_state_create_sets_pending_action_required() -> None:
    """state_create 工具需要 pending_action 确认，不直接执行。"""
    capability = _capability("application.draft.create")
    registry = ToolRegistry([CreateApplicationTool()])
    model = SequenceModel([
        _json(
            '{"type":"call_tool","tool_name":"create_application",'
            '"arguments":{"company_name":"杭州示例科技有限公司","product_name":"小微税贷"}}'
        )
    ])
    engine = ReactStepEngine(model, registry)

    outcome = engine.step(_observation(capability, registry))

    assert outcome.kind == "terminal"
    assert outcome.pending_action_required
    assert outcome.pending_tool_call is not None


def test_call_tool_final_submit_rejected_to_handoff() -> None:
    """final_submit 工具必须转人工处理。"""
    capability = replace(_capability("application.draft.create"), allowed_tools=["final_submit"])
    registry = ToolRegistry([FinalSubmitTool()])
    model = SequenceModel([
        _json('{"type":"call_tool","tool_name":"final_submit","arguments":{"company_name":"杭州示例科技有限公司"}}')
    ])
    engine = ReactStepEngine(model, registry)

    outcome = engine.step(_observation(capability, registry))

    assert outcome.kind == "invalid_action"
    assert "handoff" in (outcome.invalid_hint or "")
