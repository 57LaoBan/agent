from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.protocol import ChatRequest, RouteDecision, ToolCall, ToolResult  # noqa: E402
from xinyidai_agent.runtime import ControlledAgentLoop  # noqa: E402
from xinyidai_agent.tools.base import SlotSpec, ToolExecution, ToolSpec  # noqa: E402
from xinyidai_agent.tools.registry import ToolRegistry  # noqa: E402


class EmptyJsonModel:
    def complete(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | None = None,
    ) -> str:
        return "{}"


class StaticApplicationRouter:
    def route(self, request: ChatRequest) -> RouteDecision:
        return RouteDecision(
            scene="LOAN_APPLY",
            intent="CREATE_APPLICATION",
            raw_intent="create_application_draft",
            capability_id="application.draft.create",
            capability_source="test",
            confirmation_required=True,
            confidence=0.94,
            required_slots=["company_name", "product_name"],
            filled_slots={
                "company_name": "杭州示例科技有限公司",
                "product_name": "小微税贷",
            },
            allowed_tools=["create_application"],
            allowed_tool_categories=["application"],
            risk_level="state_create",
            route_reason="用户申请贷款。",
            route_source="test",
            should_call_tool=True,
        )


class CreateApplicationTool:
    name = "create_application"
    category = "application"
    risk_level = "state_create"
    description = "创建申请草稿。"
    requires_confirmation = True

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            category=self.category,
            risk_level=self.risk_level,
            description=self.description,
            requires_confirmation=self.requires_confirmation,
            input_slots=[SlotSpec("company_name", "string"), SlotSpec("product_name", "string")],
            output_slots=[SlotSpec("application_id", "string")],
        )

    def execute(self, request: ChatRequest, route: RouteDecision, tool_call: ToolCall) -> ToolExecution:
        return ToolExecution(
            result=ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=self.name,
                tool_category=self.category,
                status="success",
                output={"application_id": "APP-001"},
            )
        )


class ControlledMultistepRuntimeTest(unittest.TestCase):
    def test_state_changing_tool_returns_pending_action_snapshot(self) -> None:
        loop = ControlledAgentLoop(
            model=EmptyJsonModel(),
            router=StaticApplicationRouter(),
            tool_registry=ToolRegistry([CreateApplicationTool()]),
        )

        response = loop.answer(ChatRequest(user_message="我要申请小微税贷", session_id="session-1"))

        self.assertEqual(response.stop_reason, "waiting_confirmation")
        self.assertIsNotNone(response.pending_action)
        self.assertEqual(response.pending_action.risk_level, "state_create")
        self.assertEqual(response.pending_action.session_id, "session-1")
        self.assertEqual(response.pending_action.stage, "LOAN_APPLY.confirm_application")
        self.assertEqual(response.pending_action.slot_snapshot["company_name"], "杭州示例科技有限公司")
        self.assertTrue(response.pending_action.precondition_hash)
        self.assertIn("runtime_step_decision", [event.event_type for event in response.events])
        self.assertEqual(response.session_state.confirmation_status, "waiting")


if __name__ == "__main__":
    unittest.main()
