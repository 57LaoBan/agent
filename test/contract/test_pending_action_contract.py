from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.protocol import PendingAction, ToolCall  # noqa: E402
from xinyidai_agent.runtime_state import build_pending_action_hash  # noqa: E402


class PendingActionContractTest(unittest.TestCase):
    def test_pending_action_contains_snapshot_fields(self) -> None:
        tool_call = ToolCall(
            tool_call_id="tool-1",
            tool_name="create_application",
            tool_category="application",
            arguments={"company_name": "杭州示例科技有限公司", "product_name": "小微税贷"},
            risk_level="state_create",
            confirmation_required=True,
        )
        slot_snapshot = dict(tool_call.arguments)
        precondition_hash = build_pending_action_hash(
            session_id="session-1",
            scene="LOAN_APPLY",
            capability_id="application.draft.create",
            stage="LOAN_APPLY.confirm_application",
            slot_snapshot=slot_snapshot,
            tool_name=tool_call.tool_name,
        )

        action = PendingAction(
            action_id="action-1",
            tool_call=tool_call,
            title="确认创建贷款申请",
            summary="创建申请草稿。",
            risk_level="state_create",
            session_id="session-1",
            scene="LOAN_APPLY",
            capability_id="application.draft.create",
            stage="LOAN_APPLY.confirm_application",
            slot_snapshot=slot_snapshot,
            precondition_hash=precondition_hash,
            created_at="2026-05-04T00:00:00+00:00",
            expires_at="2026-05-04T00:10:00+00:00",
        )

        payload = action.model_dump()

        self.assertEqual(payload["risk_level"], "state_create")
        self.assertEqual(payload["session_id"], "session-1")
        self.assertEqual(payload["slot_snapshot"]["company_name"], "杭州示例科技有限公司")
        self.assertEqual(payload["precondition_hash"], precondition_hash)


if __name__ == "__main__":
    unittest.main()
