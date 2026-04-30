from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from xinyidai_agent.protocol import (
    ChatRequest,
    PendingAction,
    RouteDecision,
    SessionStateSnapshot,
    SessionToolResult,
    ToolResult,
)


class SessionStore(Protocol):
    def get(self, session_id: str) -> SessionStateSnapshot | None:
        ...

    def save(self, state: SessionStateSnapshot) -> None:
        ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self._states: dict[str, SessionStateSnapshot] = {}

    def get(self, session_id: str) -> SessionStateSnapshot | None:
        return self._states.get(session_id)

    def save(self, state: SessionStateSnapshot) -> None:
        self._states[state.session_id] = state


class MemoryManager:
    """Structured short-term business memory for one chat session."""

    def __init__(self, store: SessionStore | None = None, max_tool_results: int = 5) -> None:
        self._store = store or InMemorySessionStore()
        self._max_tool_results = max_tool_results

    def load(self, session_id: str | None) -> SessionStateSnapshot:
        resolved_session_id = session_id or str(uuid4())
        existing = self._store.get(resolved_session_id)
        if existing is not None:
            return existing
        return SessionStateSnapshot(
            session_id=resolved_session_id,
            updated_at=_now(),
        )

    def save(self, state: SessionStateSnapshot) -> None:
        self._store.save(state.model_copy(update={"updated_at": _now()}))

    def enrich_request(self, request: ChatRequest, state: SessionStateSnapshot) -> ChatRequest:
        metadata = dict(request.metadata)
        metadata["session_id"] = state.session_id

        for key, value in state.confirmed_slots.items():
            if value and key not in metadata:
                metadata[key] = value

        if state.selected_company_name and "company_name" not in metadata:
            metadata["company_name"] = state.selected_company_name
        if state.selected_product_name and "product_name" not in metadata:
            metadata["product_name"] = state.selected_product_name

        if state.awaiting_slots:
            metadata["_session_awaiting_slots"] = list(state.awaiting_slots)
        if state.last_route and self._can_resume_route(state, metadata):
            metadata["_session_resume_route"] = state.last_route.model_dump()
        if state.short_summary:
            metadata["_session_summary"] = state.short_summary

        return request.model_copy(update={"session_id": state.session_id, "metadata": metadata})

    def update(
        self,
        state: SessionStateSnapshot,
        request: ChatRequest,
        route: RouteDecision | None = None,
        tool_result: ToolResult | None = None,
        pending_action: PendingAction | None = None,
        stop_reason: str | None = None,
    ) -> SessionStateSnapshot:
        confirmed_slots = dict(state.confirmed_slots)
        pending_slots = dict(state.pending_slots)
        awaiting_slots = list(state.awaiting_slots)
        last_route = state.last_route
        active_scene = state.active_scene
        active_capability_id = state.active_capability_id
        selected_company_name = state.selected_company_name
        selected_product_name = state.selected_product_name
        last_credit_amount = state.last_credit_amount
        last_tool_results = list(state.last_tool_results)
        confirmation_status = state.confirmation_status
        next_pending_action = pending_action if pending_action is not None else state.pending_action

        if route is not None:
            last_route = route
            active_scene = route.scene
            active_capability_id = route.capability_id
            for slot, value in route.filled_slots.items():
                if value:
                    confirmed_slots[slot] = value
            pending_slots = dict(route.filled_slots)
            awaiting_slots = list(route.missing_slots)
            if not awaiting_slots and stop_reason != "waiting_confirmation":
                pending_slots = {}

        selected_company_name = confirmed_slots.get("company_name") or selected_company_name
        selected_product_name = confirmed_slots.get("product_name") or selected_product_name

        if pending_action is not None:
            confirmation_status = "waiting"
        elif stop_reason != "waiting_confirmation":
            confirmation_status = "none"
            next_pending_action = None

        if tool_result is not None:
            last_tool_results.append(
                SessionToolResult(
                    tool_name=tool_result.tool_name,
                    business_status=tool_result.business_status,
                    output=tool_result.output,
                    created_at=_now(),
                )
            )
            last_tool_results = last_tool_results[-self._max_tool_results :]
            output = tool_result.output
            if output.get("company_name"):
                confirmed_slots["company_name"] = output["company_name"]
                selected_company_name = str(output["company_name"])
            if tool_result.tool_name == "query_credit_amount" and output.get("credit_amount"):
                last_credit_amount = {
                    "company_name": output.get("company_name"),
                    "credit_amount": output.get("credit_amount"),
                    "data_time": output.get("data_time"),
                    "business_status": tool_result.business_status,
                }

        updated = state.model_copy(
            update={
                "active_scene": active_scene,
                "active_capability_id": active_capability_id,
                "confirmed_slots": confirmed_slots,
                "pending_slots": pending_slots,
                "awaiting_slots": awaiting_slots,
                "last_route": last_route,
                "last_tool_results": last_tool_results,
                "pending_action": next_pending_action,
                "confirmation_status": confirmation_status,
                "selected_company_name": selected_company_name,
                "selected_product_name": selected_product_name,
                "last_credit_amount": last_credit_amount,
                "short_summary": _build_summary(
                    active_scene=active_scene,
                    active_capability_id=active_capability_id,
                    selected_company_name=selected_company_name,
                    selected_product_name=selected_product_name,
                    awaiting_slots=awaiting_slots,
                    last_credit_amount=last_credit_amount,
                    confirmation_status=confirmation_status,
                ),
                "turn_count": state.turn_count + 1,
                "updated_at": _now(),
            }
        )
        self._store.save(updated)
        return updated

    def _can_resume_route(self, state: SessionStateSnapshot, metadata: dict[str, Any]) -> bool:
        if state.last_route is None or not state.awaiting_slots:
            return False
        return any(metadata.get(slot) for slot in state.awaiting_slots)


def _build_summary(
    active_scene: str | None,
    active_capability_id: str | None,
    selected_company_name: str | None,
    selected_product_name: str | None,
    awaiting_slots: list[str],
    last_credit_amount: dict[str, Any] | None,
    confirmation_status: str,
) -> str:
    parts: list[str] = []
    if active_scene:
        parts.append(f"active_scene={active_scene}")
    if active_capability_id:
        parts.append(f"capability={active_capability_id}")
    if selected_company_name:
        parts.append(f"company={selected_company_name}")
    if selected_product_name:
        parts.append(f"product={selected_product_name}")
    if last_credit_amount:
        parts.append(f"last_credit_amount={last_credit_amount.get('credit_amount')}")
    if awaiting_slots:
        parts.append(f"awaiting={','.join(awaiting_slots)}")
    if confirmation_status != "none":
        parts.append(f"confirmation={confirmation_status}")
    return "; ".join(parts)


def _now() -> str:
    return datetime.now(UTC).isoformat()
