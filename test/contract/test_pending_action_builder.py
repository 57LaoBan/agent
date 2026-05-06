from __future__ import annotations

from datetime import UTC, datetime
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.pending_actions import PendingActionBuilder  # noqa: E402
from xinyidai_agent.protocol import RouteDecision, SessionStateSnapshot, ToolCall  # noqa: E402
from xinyidai_agent.tools.base import ToolSpec  # noqa: E402


class PendingActionBuilderTest(unittest.TestCase):
    def build_action(self, tool_name: str, risk_level: str, arguments: dict[str, object]):
        route = RouteDecision(
            scene="LOAN_APPLY" if tool_name != "create_authorization_link" else "AUTHORIZATION",
            intent="CREATE_APPLICATION",
            capability_id="application.draft.create",
            confidence=0.9,
            filled_slots=arguments,
            allowed_tools=[tool_name],
            allowed_tool_categories=["application" if tool_name != "create_authorization_link" else "authorization"],
            risk_level=risk_level,
            route_reason="测试。",
        )
        state = SessionStateSnapshot(
            session_id="session-1",
            current_stage="LOAN_APPLY.confirm_application",
            confirmed_slots=arguments,
            updated_at="2026-05-04T00:00:00+00:00",
        )
        tool_call = ToolCall(
            tool_call_id="tool-1",
            tool_name=tool_name,
            arguments=arguments,
            risk_level=risk_level,
            confirmation_required=True,
        )
        spec = ToolSpec(
            name=tool_name,
            category="application" if tool_name != "create_authorization_link" else "authorization",
            risk_level=risk_level,
            description="测试工具。",
            requires_confirmation=True,
        )
        return PendingActionBuilder().build(
            request_session_id="session-1",
            route=route,
            state=state,
            tool_call=tool_call,
            spec=spec,
            now=datetime(2026, 5, 4, tzinfo=UTC),
        )

    def test_create_application_copy_is_specific(self) -> None:
        action = self.build_action(
            "create_application",
            "state_create",
            {"company_name": "杭州示例科技有限公司", "product_name": "小微税贷"},
        )

        self.assertEqual(action.title, "确认创建贷款申请草稿")
        self.assertIn("杭州示例科技有限公司", action.summary)
        self.assertIn("尚不会最终提交", action.summary)

    def test_create_authorization_link_copy_is_specific(self) -> None:
        action = self.build_action(
            "create_authorization_link",
            "link_create",
            {"company_name": "杭州示例科技有限公司"},
        )

        self.assertEqual(action.title, "确认生成企业授权链接")
        self.assertIn("授权流程", action.summary)

    def test_state_update_copy_is_cautious(self) -> None:
        action = self.build_action(
            "update_application",
            "state_update",
            {"application_id": "APP-001"},
        )

        self.assertEqual(action.title, "确认修改申请信息")
        self.assertIn("APP-001", action.summary)

    def test_final_submit_copy_requires_strong_warning(self) -> None:
        action = self.build_action("final_submit", "final_submit", {"application_id": "APP-001"})

        self.assertEqual(action.title, "确认最终提交")
        self.assertIn("正式业务提交", action.summary)

    def test_unknown_state_changing_tool_uses_safe_fallback(self) -> None:
        action = self.build_action("unknown_state_tool", "state_create", {"company_name": "杭州示例科技有限公司"})

        self.assertEqual(action.title, "确认执行业务动作")
        self.assertIn("请确认", action.summary)


if __name__ == "__main__":
    unittest.main()
