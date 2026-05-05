from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.protocol import PendingAction, SessionStateSnapshot, ToolCall  # noqa: E402
from xinyidai_agent.runtime_state import RuntimeStateReducer  # noqa: E402


class PendingActionInvalidationTest(unittest.TestCase):
    def build_state(self) -> SessionStateSnapshot:
        action = PendingAction(
            action_id="action-1",
            tool_call=ToolCall(
                tool_call_id="tool-1",
                tool_name="create_application",
                arguments={"company_name": "杭州示例科技有限公司", "product_name": "小微税贷"},
                risk_level="state_create",
                confirmation_required=True,
            ),
            title="确认创建贷款申请草稿",
            summary="创建申请草稿。",
            risk_level="state_create",
            slot_snapshot={"company_name": "杭州示例科技有限公司", "product_name": "小微税贷"},
        )
        return SessionStateSnapshot(
            session_id="session-1",
            confirmed_slots={"company_name": "杭州示例科技有限公司", "product_name": "小微税贷"},
            selected_company_name="杭州示例科技有限公司",
            selected_product_name="小微税贷",
            last_credit_amount={"company_name": "杭州示例科技有限公司", "credit_amount": "50万元"},
            last_application_id="APP-001",
            authorization_status="link_created",
            pending_action=action,
            confirmation_status="waiting",
            updated_at="2026-05-04T00:00:00+00:00",
        )

    def test_company_change_invalidates_pending_and_business_results(self) -> None:
        state = self.build_state()
        updated = RuntimeStateReducer().apply_slot_change(state, "company_name", "重庆征信")

        self.assertIsNone(updated.pending_action)
        self.assertEqual(updated.confirmation_status, "none")
        self.assertIsNone(updated.last_credit_amount)
        self.assertIsNone(updated.last_application_id)
        self.assertIsNone(updated.authorization_status)
        self.assertEqual(updated.confirmed_slots["company_name"], "重庆征信")

    def test_product_change_invalidates_product_bound_pending(self) -> None:
        state = self.build_state()
        updated = RuntimeStateReducer().apply_slot_change(state, "product_name", "经营贷")

        self.assertIsNone(updated.pending_action)
        self.assertEqual(updated.confirmation_status, "none")
        self.assertEqual(updated.selected_product_name, "经营贷")


if __name__ == "__main__":
    unittest.main()
