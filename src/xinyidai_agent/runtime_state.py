from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from xinyidai_agent.protocol import PendingAction, RouteDecision, SessionStateSnapshot, ToolResult


class RuntimeStateReducer:
    """集中处理 runtime 产生的短期业务状态变化。"""

    def apply_route(
        self,
        state: SessionStateSnapshot,
        route: RouteDecision,
    ) -> SessionStateSnapshot:
        confirmed_slots = dict(state.confirmed_slots)
        for slot, value in route.filled_slots.items():
            if value:
                confirmed_slots[slot] = value

        current_stage = _stage_for_route(route, confirmed_slots)
        return state.model_copy(
            update={
                "active_scene": route.scene,
                "active_capability_id": route.capability_id,
                "active_flow": route.scene,
                "current_stage": current_stage,
                "confirmed_slots": confirmed_slots,
                "pending_slots": dict(route.filled_slots),
                "awaiting_slots": list(route.missing_slots),
                "last_route": route,
                "selected_company_name": confirmed_slots.get("company_name") or state.selected_company_name,
                "selected_product_name": confirmed_slots.get("product_name") or state.selected_product_name,
                "stage_history": _append_stage_history(
                    state.stage_history,
                    current_stage,
                    reason="route_applied",
                ),
                "updated_at": _now(),
            }
        )

    def apply_tool_result(
        self,
        state: SessionStateSnapshot,
        route: RouteDecision,
        result: ToolResult,
    ) -> SessionStateSnapshot:
        confirmed_slots = dict(state.confirmed_slots)
        output = result.output
        selected_company_name = state.selected_company_name
        selected_product_name = state.selected_product_name
        last_credit_amount = state.last_credit_amount
        last_application_id = state.last_application_id
        authorization_status = state.authorization_status
        completed_stages = list(state.completed_stages)

        if output.get("company_name"):
            confirmed_slots["company_name"] = output["company_name"]
            selected_company_name = str(output["company_name"])
        if output.get("product_name"):
            confirmed_slots["product_name"] = output["product_name"]
            selected_product_name = str(output["product_name"])

        if result.tool_name == "query_credit_amount" and output.get("credit_amount"):
            last_credit_amount = {
                "company_name": output.get("company_name") or confirmed_slots.get("company_name"),
                "credit_amount": output.get("credit_amount"),
                "data_time": output.get("data_time"),
                "business_status": result.business_status,
            }
            current_stage = "DATA_QUERY.answered"
            completed_stages = _add_unique(completed_stages, current_stage)
        elif result.tool_name == "create_application":
            application_id = _pick_output_value(output, "application_id", "application_no", "draft_id")
            if application_id:
                last_application_id = str(application_id)
                confirmed_slots["application_id"] = str(application_id)
            current_stage = "LOAN_APPLY.application_draft_created"
            completed_stages = _add_unique(completed_stages, current_stage)
        elif result.tool_name == "create_authorization_link":
            authorization_status = "link_created"
            current_stage = "AUTHORIZATION.link_created"
            completed_stages = _add_unique(completed_stages, current_stage)
        elif result.tool_name == "query_application_status":
            current_stage = "APPLICATION_STATUS.answered"
            completed_stages = _add_unique(completed_stages, current_stage)
        else:
            current_stage = state.current_stage or _stage_for_route(route, confirmed_slots)

        return state.model_copy(
            update={
                "active_scene": route.scene,
                "active_capability_id": route.capability_id,
                "active_flow": route.scene,
                "current_stage": current_stage,
                "confirmed_slots": confirmed_slots,
                "selected_company_name": selected_company_name,
                "selected_product_name": selected_product_name,
                "last_credit_amount": last_credit_amount,
                "last_application_id": last_application_id,
                "authorization_status": authorization_status,
                "completed_stages": completed_stages,
                "stage_history": _append_stage_history(
                    state.stage_history,
                    current_stage,
                    reason=f"tool_result:{result.tool_name}",
                ),
                "updated_at": _now(),
            }
        )

    def apply_pending_action(
        self,
        state: SessionStateSnapshot,
        action: PendingAction,
    ) -> SessionStateSnapshot:
        stage = action.stage or state.current_stage
        return state.model_copy(
            update={
                "pending_action": action,
                "confirmation_status": "waiting",
                "current_stage": stage,
                "stage_history": _append_stage_history(
                    state.stage_history,
                    stage,
                    reason=f"pending_action:{action.tool_call.tool_name}",
                ),
                "updated_at": _now(),
            }
        )

    def apply_confirmation(
        self,
        state: SessionStateSnapshot,
        action_id: str,
        confirmed: bool,
    ) -> SessionStateSnapshot:
        if state.pending_action is None or state.pending_action.action_id != action_id:
            return state

        status = "confirmed" if confirmed else "cancelled"
        return state.model_copy(
            update={
                "pending_action": state.pending_action if confirmed else None,
                "confirmation_status": status,
                "stage_history": _append_stage_history(
                    state.stage_history,
                    state.current_stage,
                    reason=f"confirmation:{status}",
                ),
                "updated_at": _now(),
            }
        )

    def apply_slot_change(
        self,
        state: SessionStateSnapshot,
        slot: str,
        value: Any,
    ) -> SessionStateSnapshot:
        confirmed_slots = dict(state.confirmed_slots)
        old_value = confirmed_slots.get(slot)
        confirmed_slots[slot] = value
        update: dict[str, Any] = {
            "confirmed_slots": confirmed_slots,
            "updated_at": _now(),
        }

        if slot == "company_name":
            update["selected_company_name"] = str(value) if value else None
            if old_value and old_value != value:
                update.update(
                    {
                        "last_credit_amount": None,
                        "last_application_id": None,
                        "authorization_status": None,
                        "pending_action": None,
                        "confirmation_status": "none",
                        "stage_history": _append_stage_history(
                            state.stage_history,
                            state.current_stage,
                            reason="pending_action_invalidated:company_changed",
                        ),
                    }
                )

        if slot == "product_name":
            update["selected_product_name"] = str(value) if value else None
            if old_value and old_value != value:
                update.update(
                    {
                        "pending_action": None,
                        "confirmation_status": "none",
                        "stage_history": _append_stage_history(
                            state.stage_history,
                            state.current_stage,
                            reason="pending_action_invalidated:product_changed",
                        ),
                    }
                )

        return state.model_copy(update=update)


def build_pending_action_hash(
    *,
    session_id: str | None,
    scene: str | None,
    capability_id: str | None,
    stage: str | None,
    slot_snapshot: dict[str, Any],
    tool_name: str,
) -> str:
    payload = {
        "session_id": session_id,
        "scene": scene,
        "capability_id": capability_id,
        "stage": stage,
        "slot_snapshot": slot_snapshot,
        "tool_name": tool_name,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stage_for_route(route: RouteDecision, slots: dict[str, Any]) -> str:
    if route.missing_slots:
        first_missing = route.missing_slots[0]
        if route.scene == "DATA_QUERY" and first_missing == "company_name":
            return "DATA_QUERY.collect_company"
        if route.scene == "LOAN_APPLY" and first_missing == "company_name":
            return "LOAN_APPLY.collect_company"
        if route.scene == "LOAN_APPLY" and first_missing == "product_name":
            return "LOAN_APPLY.select_product"
        if route.scene == "AUTHORIZATION" and first_missing == "company_name":
            return "AUTHORIZATION.collect_company"
        return f"{route.scene}.collect_{first_missing}"

    if route.scene == "DATA_QUERY":
        return "DATA_QUERY.querying"
    if route.scene == "LOAN_APPLY":
        if slots.get("company_name") and slots.get("product_name"):
            return "LOAN_APPLY.confirm_application"
        if slots.get("company_name"):
            return "LOAN_APPLY.select_product"
        return "LOAN_APPLY.collect_company"
    if route.scene == "AUTHORIZATION":
        return "AUTHORIZATION.confirm_link_create"
    if route.scene == "APPLICATION_STATUS":
        return "APPLICATION_STATUS.querying"
    return f"{route.scene}.active"


def _append_stage_history(
    history: list[dict[str, Any]],
    stage: str | None,
    reason: str,
) -> list[dict[str, Any]]:
    if not stage:
        return history
    item = {"stage": stage, "reason": reason, "created_at": _now()}
    return [*history, item][-20:]


def _add_unique(items: list[str], item: str) -> list[str]:
    if item in items:
        return items
    return [*items, item]


def _pick_output_value(output: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = output.get(key)
        if value:
            return value
    return None


def _now() -> str:
    return datetime.now(UTC).isoformat()
