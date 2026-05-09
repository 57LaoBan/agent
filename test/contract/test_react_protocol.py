from __future__ import annotations

import pytest
from pydantic import ValidationError

from xinyidai_agent.protocol import (
    ReactAction,
    ReactObservationKind,
    ReactStepDecision,
    ReactTerminal,
    ReactTerminalReason,
    ToolExecutionObservation,
    ToolResultEnvelope,
)


def test_react_action_extra_forbid() -> None:
    """多余字段必须被 Pydantic 拒绝。"""
    with pytest.raises(ValidationError):
        ReactAction(type="answer", final_answer="可以办理。", unexpected=True)


def test_react_action_call_tool_minimal() -> None:
    """call_tool 至少包含 tool_name，参数可为空对象。"""
    action = ReactAction(type="call_tool", tool_name="rag_search")

    assert action.tool_name == "rag_search"
    assert action.arguments == {}


def test_react_observation_kind_enumerable() -> None:
    """6 种 observation kind 都可被构造。"""
    kinds: list[ReactObservationKind] = [
        "success",
        "empty",
        "failed",
        "unavailable",
        "blocked",
        "schema_error",
    ]

    observations = [ToolExecutionObservation(kind=kind, tool_name="rag_search") for kind in kinds]

    assert [observation.kind for observation in observations] == kinds


def test_react_terminal_reason_enumerable() -> None:
    """8 种终止原因都可被构造。"""
    reasons: list[ReactTerminalReason] = [
        "completed",
        "answered_no_tool",
        "ask_user",
        "wait_confirmation",
        "handoff",
        "tool_blocked",
        "max_steps",
        "runtime_error",
    ]

    terminals = [ReactTerminal(reason=reason, final_answer="done") for reason in reasons]

    assert [terminal.reason for terminal in terminals] == reasons


def test_business_status_includes_tool_unavailable() -> None:
    """业务状态枚举必须包含工具未注册状态。"""
    envelope = ToolResultEnvelope(success=False, status="TOOL_UNAVAILABLE")

    assert envelope.status == "TOOL_UNAVAILABLE"


def test_react_step_decision_default_confidence() -> None:
    """ReactStepDecision 默认置信度是 0.5。"""
    decision = ReactStepDecision(action=ReactAction(type="answer", final_answer="收到。"))

    assert decision.confidence == 0.5
