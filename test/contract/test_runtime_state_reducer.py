from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.protocol import PendingAction, RouteDecision, SessionStateSnapshot, ToolCall, ToolResult  # noqa: E402
from xinyidai_agent.runtime_state import RuntimeStateReducer  # noqa: E402


class RuntimeStateReducerTest(unittest.TestCase):
    def test_data_query_success_records_credit_amount_and_stage(self) -> None:
        reducer = RuntimeStateReducer()
        state = SessionStateSnapshot(session_id="s1", updated_at="2026-05-04T00:00:00+00:00")
        route = RouteDecision(
            scene="DATA_QUERY",
            intent="CREDIT_LIMIT_QUERY",
            capability_id="credit.limit.read",
            confidence=0.9,
            required_slots=["company_name"],
            filled_slots={"company_name": "杭州示例科技有限公司"},
            route_reason="查询额度。",
        )
        result = ToolResult(
            tool_call_id="tool-1",
            tool_name="query_credit_amount",
            tool_category="data_query",
            status="success",
            business_status="CREDIT_AMOUNT_FOUND",
            output={
                "company_name": "杭州示例科技有限公司",
                "credit_amount": "50万元",
                "data_time": "2026-04-28",
            },
        )

        updated = reducer.apply_tool_result(reducer.apply_route(state, route), route, result)

        self.assertEqual(updated.current_stage, "DATA_QUERY.answered")
        self.assertEqual(updated.last_credit_amount["credit_amount"], "50万元")
        self.assertIn("DATA_QUERY.answered", updated.completed_stages)

    def test_application_creation_records_application_id(self) -> None:
        reducer = RuntimeStateReducer()
        state = SessionStateSnapshot(session_id="s1", updated_at="2026-05-04T00:00:00+00:00")
        route = RouteDecision(
            scene="LOAN_APPLY",
            intent="CREATE_APPLICATION",
            capability_id="application.draft.create",
            confidence=0.9,
            filled_slots={"company_name": "杭州示例科技有限公司", "product_name": "小微税贷"},
            route_reason="创建申请草稿。",
        )
        result = ToolResult(
            tool_call_id="tool-1",
            tool_name="create_application",
            tool_category="application",
            status="success",
            business_status="APPLICATION_DRAFT_CREATED",
            output={"application_id": "APP-001"},
        )

        updated = reducer.apply_tool_result(reducer.apply_route(state, route), route, result)

        self.assertEqual(updated.last_application_id, "APP-001")
        self.assertEqual(updated.current_stage, "LOAN_APPLY.application_draft_created")

    def test_pending_action_and_cancel_updates_confirmation_state(self) -> None:
        reducer = RuntimeStateReducer()
        state = SessionStateSnapshot(session_id="s1", updated_at="2026-05-04T00:00:00+00:00")
        action = PendingAction(
            action_id="action-1",
            tool_call=ToolCall(tool_call_id="tool-1", tool_name="create_authorization_link"),
            title="确认授权",
            summary="生成授权链接。",
            risk_level="link_create",
        )

        waiting = reducer.apply_pending_action(state, action)
        cancelled = reducer.apply_confirmation(waiting, "action-1", confirmed=False)

        self.assertEqual(waiting.confirmation_status, "waiting")
        self.assertEqual(cancelled.confirmation_status, "cancelled")
        self.assertIsNone(cancelled.pending_action)


if __name__ == "__main__":
    unittest.main()
