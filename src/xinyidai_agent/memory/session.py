from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from xinyidai_agent.protocol import (
    AgentEvent,
    ChatRequest,
    PendingAction,
    RouteDecision,
    SessionStateSnapshot,
    SessionToolResult,
    ToolResult,
)
from xinyidai_agent.runtime_state import RuntimeStateReducer
from xinyidai_agent.memory.store import JsonlTranscriptStore, TranscriptStore


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

    def __init__(
        self,
        store: SessionStore | None = None,
        transcript_store: TranscriptStore | None = None,
        max_tool_results: int = 5,
        max_recent_turns: int = 10,
    ) -> None:
        self._store = store or InMemorySessionStore()
        self._transcripts = transcript_store or JsonlTranscriptStore()
        self._max_tool_results = max_tool_results
        self._max_recent_turns = max_recent_turns
        self._reducer = RuntimeStateReducer()

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

    def append_event(self, session_id: str, event: AgentEvent) -> None:
        self._transcripts.append_event(session_id, event.turn_id, event.sequence, event)

    def append_turn_summary(self, session_id: str, turn_id: str, summary: dict[str, Any]) -> None:
        self._transcripts.append_turn_summary(session_id, turn_id, summary)

    def get_transcript_events(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        return self._transcripts.list_events(session_id, limit=limit)

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
        if state.current_stage:
            metadata["_session_current_stage"] = state.current_stage
        if state.recent_turns:
            metadata["_session_recent_turns"] = list(state.recent_turns[-6:])

        return request.model_copy(update={"session_id": state.session_id, "metadata": metadata})

    def update(
        self,
        state: SessionStateSnapshot,
        request: ChatRequest,
        route: RouteDecision | None = None,
        tool_result: ToolResult | None = None,
        pending_action: PendingAction | None = None,
        stop_reason: str | None = None,
        final_answer: str | None = None,
    ) -> SessionStateSnapshot:
        reduced = state
        if route is not None:
            reduced = self._reducer.apply_route(reduced, route)
        if tool_result is not None and route is not None:
            reduced = self._reducer.apply_tool_result(reduced, route, tool_result)
        if pending_action is not None:
            reduced = self._reducer.apply_pending_action(reduced, pending_action)

        confirmed_slots = dict(state.confirmed_slots)
        pending_slots = dict(state.pending_slots)
        awaiting_slots = list(state.awaiting_slots)
        last_route = state.last_route
        active_scene = state.active_scene
        active_capability_id = state.active_capability_id
        active_flow = reduced.active_flow
        current_stage = reduced.current_stage
        selected_company_name = state.selected_company_name
        selected_product_name = state.selected_product_name
        last_credit_amount = state.last_credit_amount
        last_application_id = reduced.last_application_id
        authorization_status = reduced.authorization_status
        last_tool_results = list(state.last_tool_results)
        confirmation_status = state.confirmation_status
        next_pending_action = pending_action if pending_action is not None else state.pending_action
        completed_stages = list(reduced.completed_stages)
        stage_history = list(reduced.stage_history)
        recent_turns = list(state.recent_turns)

        if route is not None:
            last_route = route
            active_scene = route.scene
            active_capability_id = route.capability_id
            active_flow = reduced.active_flow
            current_stage = reduced.current_stage
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
            if tool_result.tool_name == "create_application":
                application_id = (
                    output.get("application_id")
                    or output.get("application_no")
                    or output.get("draft_id")
                )
                if application_id:
                    last_application_id = str(application_id)
            if tool_result.tool_name == "create_authorization_link":
                authorization_status = "link_created"

        recent_turns, folded_summary = _append_recent_turn(
            recent_turns,
            request=request,
            route=route,
            tool_result=tool_result,
            pending_action=pending_action,
            final_answer=final_answer,
            stop_reason=stop_reason,
            limit=self._max_recent_turns,
        )

        updated = state.model_copy(
            update={
                "active_scene": active_scene,
                "active_capability_id": active_capability_id,
                "active_flow": active_flow,
                "current_stage": current_stage,
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
                "last_application_id": last_application_id,
                "authorization_status": authorization_status,
                "recent_turns": recent_turns,
                "completed_stages": completed_stages,
                "stage_history": stage_history,
                "short_summary": _build_summary(
                    active_scene=active_scene,
                    active_capability_id=active_capability_id,
                    active_flow=active_flow,
                    current_stage=current_stage,
                    selected_company_name=selected_company_name,
                    selected_product_name=selected_product_name,
                    awaiting_slots=awaiting_slots,
                    last_credit_amount=last_credit_amount,
                    last_application_id=last_application_id,
                    authorization_status=authorization_status,
                    confirmation_status=confirmation_status,
                    folded_summary=folded_summary or state.short_summary,
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
    active_flow: str | None,
    current_stage: str | None,
    selected_company_name: str | None,
    selected_product_name: str | None,
    awaiting_slots: list[str],
    last_credit_amount: dict[str, Any] | None,
    last_application_id: str | None,
    authorization_status: str | None,
    confirmation_status: str,
    folded_summary: str = "",
) -> str:
    parts: list[str] = []
    if folded_summary:
        parts.append(f"历史业务摘要={folded_summary}")
    if active_scene:
        parts.append(f"active_scene={active_scene}")
    if active_capability_id:
        parts.append(f"capability={active_capability_id}")
    if active_flow:
        parts.append(f"flow={active_flow}")
    if current_stage:
        parts.append(f"stage={current_stage}")
    if selected_company_name:
        parts.append(f"company={selected_company_name}")
    if selected_product_name:
        parts.append(f"product={selected_product_name}")
    if last_credit_amount:
        parts.append(f"last_credit_amount={last_credit_amount.get('credit_amount')}")
    if last_application_id:
        parts.append(f"last_application_id={last_application_id}")
    if authorization_status:
        parts.append(f"authorization={authorization_status}")
    if awaiting_slots:
        parts.append(f"awaiting={','.join(awaiting_slots)}")
    if confirmation_status != "none":
        parts.append(f"confirmation={confirmation_status}")
    return "; ".join(parts)


def _append_recent_turn(
    recent_turns: list[dict[str, Any]],
    request: ChatRequest,
    route: RouteDecision | None,
    tool_result: ToolResult | None,
    pending_action: PendingAction | None,
    final_answer: str | None,
    stop_reason: str | None,
    limit: int,
) -> tuple[list[dict[str, Any]], str]:
    item: dict[str, Any] = {
        "user_message": request.user_message,
        "stop_reason": stop_reason,
        "created_at": _now(),
    }
    if route is not None:
        item["scene"] = route.scene
        item["capability_id"] = route.capability_id
        item["intent"] = route.intent
    if tool_result is not None:
        item["tool_name"] = tool_result.tool_name
        item["business_status"] = tool_result.business_status
    if pending_action is not None:
        item["pending_action_id"] = pending_action.action_id
        item["pending_tool_name"] = pending_action.tool_call.tool_name
    if final_answer:
        item["assistant_answer"] = final_answer

    combined = [*recent_turns, item]
    overflow = combined[:-limit] if len(combined) > limit else []
    window = combined[-limit:]
    folded = _fold_turns(overflow)
    return window, folded


def _fold_turns(turns: list[dict[str, Any]]) -> str:
    business_facts: list[str] = []
    for turn in turns:
        scene = turn.get("scene")
        tool_name = turn.get("tool_name")
        business_status = turn.get("business_status")
        if scene or tool_name or business_status:
            business_facts.append(
                ",".join(str(value) for value in [scene, tool_name, business_status] if value)
            )
    return " | ".join(business_facts[-5:])


def _now() -> str:
    return datetime.now(UTC).isoformat()
