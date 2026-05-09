from __future__ import annotations

from datetime import UTC, datetime
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.pending_actions import PendingActionBuilder  # noqa: E402
from xinyidai_agent.protocol import RouteDecision, SessionStateSnapshot, ToolCall  # noqa: E402
from xinyidai_agent.router.action_validator import PendingActionValidator  # noqa: E402
from xinyidai_agent.tools.base import SlotSpec, ToolSpec  # noqa: E402


class PendingActionValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.route = RouteDecision(
            scene="LOAN_APPLY",
            intent="CREATE_APPLICATION",
            capability_id="application.draft.create",
            confirmation_required=True,
            confidence=0.93,
            filled_slots={"company_name": "杭州示例科技有限公司", "product_name": "小微税贷"},
            allowed_tools=["create_application"],
            allowed_tool_categories=["application"],
            risk_level="state_create",
            route_reason="创建申请草稿。",
        )
        self.tool_call = ToolCall(
            tool_call_id="tool-1",
            tool_name="create_application",
            tool_category="application",
            arguments={"company_name": "杭州示例科技有限公司", "product_name": "小微税贷"},
            risk_level="state_create",
            confirmation_required=True,
        )
        self.spec = ToolSpec(
            name="create_application",
            category="application",
            risk_level="state_create",
            description="创建申请草稿。",
            requires_confirmation=True,
            input_slots=[SlotSpec("company_name", "string"), SlotSpec("product_name", "string")],
        )
        base_state = SessionStateSnapshot(
            session_id="session-1",
            active_scene="LOAN_APPLY",
            active_capability_id="application.draft.create",
            active_flow="LOAN_APPLY",
            current_stage="LOAN_APPLY.confirm_application",
            confirmed_slots={"company_name": "杭州示例科技有限公司", "product_name": "小微税贷"},
            updated_at="2026-05-04T00:00:00+00:00",
        )
        self.action = PendingActionBuilder().build(
            request_session_id="session-1",
            route=self.route,
            state=base_state,
            tool_call=self.tool_call,
            spec=self.spec,
            now=datetime(2026, 5, 4, tzinfo=UTC),
        )
        self.state = base_state.model_copy(
            update={"pending_action": self.action, "confirmation_status": "confirmed"}
        )
        self.validator = PendingActionValidator()

    def assertBlocked(self, result, code: str) -> None:
        self.assertFalse(result.allowed)
        self.assertEqual(result.code, code)

    def test_valid_snapshot_is_allowed(self) -> None:
        result = self.validator.validate_pending_action_snapshot(
            state=self.state,
            route=self.route,
            action=self.action,
            tool_call=self.tool_call,
            spec=self.spec,
            now=datetime(2026, 5, 4, 0, 1, tzinfo=UTC),
        )

        self.assertTrue(result.allowed)

    def test_wrong_action_id_is_blocked(self) -> None:
        action = self.action.model_copy(update={"action_id": "other-action"})
        result = self.validator.validate_pending_action_snapshot(
            state=self.state,
            route=self.route,
            action=action,
            tool_call=self.tool_call,
            spec=self.spec,
        )

        self.assertBlocked(result, "ACTION_ID_MISMATCH")

    def test_wrong_session_id_is_blocked(self) -> None:
        action = self.action.model_copy(update={"session_id": "other-session"})
        state = self.state.model_copy(update={"pending_action": action})

        result = self.validator.validate_pending_action_snapshot(
            state=state,
            route=self.route,
            action=action,
            tool_call=self.tool_call,
            spec=self.spec,
        )

        self.assertBlocked(result, "SESSION_MISMATCH")

    def test_expired_action_is_blocked(self) -> None:
        result = self.validator.validate_pending_action_snapshot(
            state=self.state,
            route=self.route,
            action=self.action,
            tool_call=self.tool_call,
            spec=self.spec,
            now=datetime(2026, 5, 4, 0, 11, tzinfo=UTC),
        )

        self.assertBlocked(result, "ACTION_EXPIRED")

    def test_changed_company_name_is_blocked(self) -> None:
        state = self.state.model_copy(
            update={"confirmed_slots": {"company_name": "重庆征信", "product_name": "小微税贷"}}
        )
        result = self.validator.validate_pending_action_snapshot(
            state=state,
            route=self.route,
            action=self.action,
            tool_call=self.tool_call,
            spec=self.spec,
        )

        self.assertBlocked(result, "SLOT_SNAPSHOT_CONFLICT")

    def test_changed_product_name_is_blocked(self) -> None:
        state = self.state.model_copy(
            update={"confirmed_slots": {"company_name": "杭州示例科技有限公司", "product_name": "经营贷"}}
        )
        result = self.validator.validate_pending_action_snapshot(
            state=state,
            route=self.route,
            action=self.action,
            tool_call=self.tool_call,
            spec=self.spec,
        )

        self.assertBlocked(result, "SLOT_SNAPSHOT_CONFLICT")

    def test_changed_tool_arguments_are_blocked(self) -> None:
        tool_call = self.tool_call.model_copy(
            update={"arguments": {"company_name": "杭州示例科技有限公司", "product_name": "经营贷"}}
        )
        result = self.validator.validate_pending_action_snapshot(
            state=self.state,
            route=self.route,
            action=self.action,
            tool_call=tool_call,
            spec=self.spec,
        )

        self.assertBlocked(result, "TOOL_ARGUMENTS_MISMATCH")

    def test_changed_capability_id_is_blocked(self) -> None:
        route = self.route.model_copy(update={"capability_id": "other.capability"})
        result = self.validator.validate_pending_action_snapshot(
            state=self.state,
            route=route,
            action=self.action,
            tool_call=self.tool_call,
            spec=self.spec,
        )

        self.assertBlocked(result, "CAPABILITY_MISMATCH")

    def test_changed_risk_level_is_blocked(self) -> None:
        spec = self.spec.__class__(
            name=self.spec.name,
            category=self.spec.category,
            risk_level="read_only",
            description=self.spec.description,
            input_slots=self.spec.input_slots,
        )
        result = self.validator.validate_pending_action_snapshot(
            state=self.state,
            route=self.route,
            action=self.action,
            tool_call=self.tool_call,
            spec=spec,
        )

        self.assertBlocked(result, "RISK_LEVEL_MISMATCH")

    def test_missing_pending_action_is_blocked(self) -> None:
        state = self.state.model_copy(update={"pending_action": None})
        result = self.validator.validate_pending_action_snapshot(
            state=state,
            route=self.route,
            action=self.action,
            tool_call=self.tool_call,
            spec=self.spec,
        )

        self.assertBlocked(result, "PENDING_ACTION_MISSING")

    def test_confirmed_action_a_cannot_execute_tool_b(self) -> None:
        tool_call = self.tool_call.model_copy(update={"tool_name": "create_authorization_link"})
        result = self.validator.validate_pending_action_snapshot(
            state=self.state,
            route=self.route,
            action=self.action,
            tool_call=tool_call,
            spec=self.spec,
        )

        self.assertBlocked(result, "TOOL_NAME_MISMATCH")


if __name__ == "__main__":
    unittest.main()
