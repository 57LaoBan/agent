from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.protocol import ChatRequest, RouteDecision, SessionStateSnapshot, ToolResult  # noqa: E402
from xinyidai_agent.runtime_policy import RuntimeStepContext, RuntimeStepPolicy  # noqa: E402


class ControlledStepPolicyTest(unittest.TestCase):
    def route(self, risk_level: str = "read_only", confirmation_required: bool = False) -> RouteDecision:
        return RouteDecision(
            scene="DATA_QUERY" if risk_level == "read_only" else "LOAN_APPLY",
            intent="CREDIT_LIMIT_QUERY" if risk_level == "read_only" else "CREATE_APPLICATION",
            capability_id="credit.limit.read" if risk_level == "read_only" else "application.draft.create",
            confirmation_required=confirmation_required,
            confidence=0.9,
            filled_slots={"company_name": "杭州示例科技有限公司"},
            allowed_tools=["query_credit_amount"] if risk_level == "read_only" else ["create_application"],
            allowed_tool_categories=["data_query"] if risk_level == "read_only" else ["application"],
            risk_level=risk_level,
            route_reason="测试。",
            should_call_tool=True,
        )

    def context(self, **kwargs) -> RuntimeStepContext:
        return RuntimeStepContext(
            step_index=kwargs.pop("step_index", 1),
            max_steps=kwargs.pop("max_steps", 3),
            request=ChatRequest(user_message="测试", session_id="s1"),
            session_state=SessionStateSnapshot(session_id="s1", updated_at="2026-05-04T00:00:00+00:00"),
            route=kwargs.pop("route", self.route()),
            last_tool_result=kwargs.pop("last_tool_result", None),
            executed_tool_names=kwargs.pop("executed_tool_names", []),
        )

    def test_missing_slots_asks_user(self) -> None:
        route = self.route().model_copy(update={"missing_slots": ["company_name"]})
        decision = RuntimeStepPolicy().decide_next_step(self.context(route=route))

        self.assertEqual(decision.step_type, "ASK_USER")
        self.assertEqual(decision.missing_slots, ["company_name"])

    def test_terminal_tool_result_generates_final_answer(self) -> None:
        result = ToolResult(
            tool_call_id="tool-1",
            tool_name="query_credit_amount",
            status="success",
            terminal=True,
            output={"credit_amount": "50万元"},
        )
        decision = RuntimeStepPolicy().decide_next_step(self.context(step_index=2, last_tool_result=result))

        self.assertEqual(decision.step_type, "GENERATE_FINAL_ANSWER")

    def test_non_terminal_without_next_tool_generates_safe_answer(self) -> None:
        result = ToolResult(
            tool_call_id="tool-1",
            tool_name="query_credit_amount",
            status="success",
            terminal=False,
            allowed_next_tools=[],
        )
        decision = RuntimeStepPolicy().decide_next_step(self.context(step_index=2, last_tool_result=result))

        self.assertEqual(decision.step_type, "GENERATE_FINAL_ANSWER")

    def test_same_tool_repetition_is_stopped(self) -> None:
        decision = RuntimeStepPolicy().decide_next_step(
            self.context(executed_tool_names=["query_credit_amount"])
        )

        self.assertEqual(decision.step_type, "STOP")

    def test_max_steps_stops(self) -> None:
        decision = RuntimeStepPolicy().decide_next_step(self.context(step_index=4, max_steps=3))

        self.assertEqual(decision.step_type, "STOP")

    def test_state_create_becomes_pending_action(self) -> None:
        decision = RuntimeStepPolicy().decide_next_step(
            self.context(route=self.route(risk_level="state_create", confirmation_required=True))
        )

        self.assertEqual(decision.step_type, "PROPOSE_PENDING_ACTION")

    def test_link_create_becomes_pending_action(self) -> None:
        route = self.route(risk_level="link_create", confirmation_required=True).model_copy(
            update={
                "scene": "AUTHORIZATION",
                "intent": "CREATE_AUTHORIZATION_LINK",
                "capability_id": "authorization.link.create",
                "allowed_tools": ["create_authorization_link"],
                "allowed_tool_categories": ["authorization"],
            }
        )
        decision = RuntimeStepPolicy().decide_next_step(self.context(route=route))

        self.assertEqual(decision.step_type, "PROPOSE_PENDING_ACTION")

    def test_final_submit_handoffs(self) -> None:
        decision = RuntimeStepPolicy().decide_next_step(
            self.context(route=self.route(risk_level="final_submit", confirmation_required=True))
        )

        self.assertEqual(decision.step_type, "HANDOFF")


if __name__ == "__main__":
    unittest.main()
