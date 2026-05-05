from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from xinyidai_agent.protocol import ChatRequest, PendingAction, RouteDecision, SessionStateSnapshot, ToolCall
from xinyidai_agent.runtime_state import build_pending_action_hash
from xinyidai_agent.tools.base import ToolSpec


class PendingActionBuilder:
    """为需要确认的业务动作生成可校验的 action snapshot。"""

    def build(
        self,
        *,
        route: RouteDecision,
        state: SessionStateSnapshot,
        tool_call: ToolCall,
        spec: ToolSpec,
        request: ChatRequest | None = None,
        request_session_id: str | None = None,
        now: datetime | None = None,
    ) -> PendingAction:
        created_at = now or datetime.now(UTC)
        session_id = request.session_id if request is not None else request_session_id
        slot_snapshot = self._slot_snapshot(route, state, tool_call)
        title, summary = self._copy(tool_call, spec)
        precondition_hash = build_pending_action_hash(
            session_id=session_id,
            scene=route.scene,
            capability_id=route.capability_id,
            stage=state.current_stage,
            slot_snapshot=slot_snapshot,
            tool_name=tool_call.tool_name,
        )

        return PendingAction(
            action_id=str(uuid4()),
            tool_call=tool_call,
            title=title,
            summary=summary,
            risk_level=spec.risk_level,
            confirm_label="确认",
            cancel_label="取消",
            details=self._details(tool_call),
            session_id=session_id,
            scene=route.scene,
            capability_id=route.capability_id,
            stage=state.current_stage,
            slot_snapshot=slot_snapshot,
            precondition_hash=precondition_hash,
            created_at=created_at.isoformat(),
            expires_at=(created_at + timedelta(minutes=10)).isoformat(),
        )

    def _slot_snapshot(
        self,
        route: RouteDecision,
        state: SessionStateSnapshot,
        tool_call: ToolCall,
    ) -> dict[str, Any]:
        return {
            **state.confirmed_slots,
            **route.filled_slots,
            **tool_call.arguments,
        }

    def _copy(self, tool_call: ToolCall, spec: ToolSpec) -> tuple[str, str]:
        args = tool_call.arguments
        company = str(args.get("company_name") or "该企业")
        product = str(args.get("product_name") or "该产品")
        application_id = str(args.get("application_id") or "当前申请")

        if tool_call.tool_name == "create_application":
            return (
                "确认创建贷款申请草稿",
                f"将为 {company} 创建 {product} 申请草稿，尚不会最终提交。",
            )
        if tool_call.tool_name == "create_authorization_link":
            return (
                "确认生成企业授权链接",
                f"将为 {company} 生成授权链接，客户点击后进入授权流程。",
            )
        if tool_call.tool_name == "update_application":
            return (
                "确认修改申请信息",
                f"将修改申请 {application_id} 的业务信息。",
            )
        if tool_call.tool_name == "final_submit" or spec.risk_level == "final_submit":
            return (
                "确认最终提交",
                "该操作可能产生正式业务提交，建议二次确认或转人工。",
            )
        return (
            "确认执行业务动作",
            f"该操作风险等级为 {spec.risk_level}，请确认业务主体和参数无误后再执行。",
        )

    def _details(self, tool_call: ToolCall) -> list[dict[str, str]]:
        labels = {
            "company_name": "企业",
            "product_name": "产品",
            "application_id": "申请编号",
        }
        details = [
            {"label": labels.get(key, key), "value": str(value)}
            for key, value in tool_call.arguments.items()
            if value is not None
        ]
        details.append({"label": "动作", "value": tool_call.tool_name})
        return details
