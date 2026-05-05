from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.memory import MemoryManager  # noqa: E402
from xinyidai_agent.protocol import ChatRequest, ConfirmActionRequest, RouteDecision, ToolCall, ToolResult  # noqa: E402
from xinyidai_agent.runtime import ControlledAgentLoop  # noqa: E402
from xinyidai_agent.tools.base import SlotSpec, ToolExecution, ToolSpec  # noqa: E402
from xinyidai_agent.tools.registry import ToolRegistry  # noqa: E402


class EmptyJsonModel:
    def complete(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | None = None,
    ) -> str:
        if response_format:
            return "{}"
        return "申请草稿已创建，申请编号 APP-001。"


class StaticApplicationRouter:
    def route(self, request: ChatRequest) -> RouteDecision:
        return RouteDecision(
            scene="LOAN_APPLY",
            intent="CREATE_APPLICATION",
            capability_id="application.draft.create",
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

    def __init__(self) -> None:
        self.calls = 0

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
        self.calls += 1
        return ToolExecution(
            result=ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=self.name,
                tool_category=self.category,
                status="success",
                business_status="APPLICATION_DRAFT_CREATED",
                output={"application_id": "APP-001"},
                terminal=True,
            )
        )


class ConfirmationFlowTest(unittest.TestCase):
    def build_loop(self):
        tool = CreateApplicationTool()
        memory = MemoryManager()
        loop = ControlledAgentLoop(
            model=EmptyJsonModel(),
            router=StaticApplicationRouter(),
            tool_registry=ToolRegistry([tool]),
            memory_manager=memory,
        )
        return loop, tool

    def test_confirm_executes_original_pending_tool_call(self) -> None:
        loop, tool = self.build_loop()
        first = loop.answer(ChatRequest(user_message="我要申请小微税贷", session_id="session-1"))

        response = loop.confirm(
            ConfirmActionRequest(
                session_id="session-1",
                action_id=first.pending_action.action_id,
                confirmed=True,
            )
        )

        self.assertEqual(tool.calls, 1)
        self.assertEqual(response.stop_reason, "completed")
        self.assertEqual(response.business_status, "APPLICATION_DRAFT_CREATED")
        self.assertEqual(response.session_state.last_application_id, "APP-001")

    def test_cancel_clears_pending_action_and_does_not_execute_tool(self) -> None:
        loop, tool = self.build_loop()
        first = loop.answer(ChatRequest(user_message="我要申请小微税贷", session_id="session-1"))

        response = loop.confirm(
            ConfirmActionRequest(
                session_id="session-1",
                action_id=first.pending_action.action_id,
                confirmed=False,
            )
        )

        self.assertEqual(tool.calls, 0)
        self.assertIsNone(response.session_state.pending_action)
        self.assertEqual(response.session_state.confirmation_status, "cancelled")

    def test_switch_company_invalidates_old_pending_action(self) -> None:
        loop, tool = self.build_loop()
        first = loop.answer(ChatRequest(user_message="我要申请小微税贷", session_id="session-1"))
        state = loop._memory.load("session-1")
        mutated = state.model_copy(
            update={
                "confirmed_slots": {"company_name": "重庆征信", "product_name": "小微税贷"},
            }
        )
        loop._memory.save(mutated)

        response = loop.confirm(
            ConfirmActionRequest(
                session_id="session-1",
                action_id=first.pending_action.action_id,
                confirmed=True,
            )
        )

        self.assertEqual(tool.calls, 0)
        self.assertEqual(response.stop_reason, "confirmation_blocked")
        self.assertIn("SLOT_SNAPSHOT_CONFLICT", response.answer)

    def test_registry_fail_closed_for_state_tool_without_validated_pending_action(self) -> None:
        tool = CreateApplicationTool()
        registry = ToolRegistry([tool])
        route = StaticApplicationRouter().route(ChatRequest(user_message="我要申请", session_id="session-1"))
        tool_call = ToolCall(
            tool_call_id="tool-1",
            tool_name="create_application",
            tool_category="application",
            arguments={"company_name": "杭州示例科技有限公司", "product_name": "小微税贷"},
            risk_level="state_create",
            confirmation_required=True,
        )

        execution = registry.execute(
            ChatRequest(user_message="确认", session_id="session-1"),
            route,
            tool_call,
        )

        self.assertEqual(tool.calls, 0)
        self.assertEqual(execution.result.status, "blocked")
        self.assertIn("pending_action", execution.result.error_message)


if __name__ == "__main__":
    unittest.main()
