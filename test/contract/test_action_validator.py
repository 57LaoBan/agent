from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.protocol import RouteDecision, SessionStateSnapshot, ToolCall  # noqa: E402
from xinyidai_agent.router.policy import RoutePolicy  # noqa: E402
from xinyidai_agent.tools.base import SlotSpec, ToolSpec  # noqa: E402


class ActionValidatorTest(unittest.TestCase):
    def test_allows_read_only_tool_with_required_slots(self) -> None:
        policy = RoutePolicy()
        route = RouteDecision(
            scene="DATA_QUERY",
            intent="CREDIT_LIMIT_QUERY",
            capability_id="credit.limit.read",
            confidence=0.9,
            required_slots=["company_name"],
            filled_slots={"company_name": "杭州示例科技有限公司"},
            allowed_tools=["query_credit_amount"],
            allowed_tool_categories=["data_query"],
            route_reason="查询额度。",
        )
        tool_call = ToolCall(
            tool_call_id="tool-1",
            tool_name="query_credit_amount",
            tool_category="data_query",
            arguments={"company_name": "杭州示例科技有限公司", "query": "能贷多少？"},
            risk_level="read_only",
        )
        spec = ToolSpec(
            name="query_credit_amount",
            category="data_query",
            risk_level="read_only",
            description="查询额度。",
            input_slots=[SlotSpec("company_name", "string"), SlotSpec("query", "string")],
        )

        result = policy.validate_tool_action(route, tool_call, [spec])

        self.assertTrue(result.allowed)

    def test_blocks_tool_outside_allowed_names(self) -> None:
        policy = RoutePolicy()
        route = RouteDecision(
            scene="KNOWLEDGE_QA",
            intent="POLICY_OR_PRODUCT_QA",
            confidence=0.9,
            allowed_tools=["rag_search"],
            allowed_tool_categories=["knowledge"],
            route_reason="知识问答。",
        )
        tool_call = ToolCall(
            tool_call_id="tool-1",
            tool_name="query_credit_amount",
            tool_category="data_query",
            arguments={"company_name": "杭州示例科技有限公司"},
        )
        spec = ToolSpec(
            name="query_credit_amount",
            category="data_query",
            risk_level="read_only",
            description="查询额度。",
        )

        result = policy.validate_tool_action(route, tool_call, [spec])

        self.assertFalse(result.allowed)
        self.assertIn("不在当前能力允许列表", result.reason)

    def test_blocks_state_change_without_confirmation(self) -> None:
        policy = RoutePolicy()
        route = RouteDecision(
            scene="LOAN_APPLY",
            intent="CREATE_APPLICATION",
            capability_id="application.draft.create",
            confidence=0.9,
            filled_slots={"company_name": "杭州示例科技有限公司", "product_name": "小微税贷"},
            allowed_tools=["create_application"],
            allowed_tool_categories=["application"],
            route_reason="创建申请。",
        )
        tool_call = ToolCall(
            tool_call_id="tool-1",
            tool_name="create_application",
            tool_category="application",
            arguments={"company_name": "杭州示例科技有限公司", "product_name": "小微税贷"},
            risk_level="state_create",
            confirmation_required=True,
        )
        spec = ToolSpec(
            name="create_application",
            category="application",
            risk_level="state_create",
            description="创建申请草稿。",
            requires_confirmation=True,
            input_slots=[SlotSpec("company_name", "string"), SlotSpec("product_name", "string")],
        )

        result = policy.validate_tool_action(
            route,
            tool_call,
            [spec],
            session_state=SessionStateSnapshot(session_id="s1", updated_at="2026-05-04T00:00:00+00:00"),
        )

        self.assertFalse(result.allowed)
        self.assertEqual(result.code, "PENDING_ACTION_MISSING")
        self.assertIn("pending_action", result.reason)


if __name__ == "__main__":
    unittest.main()
