from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from xinyidai_agent.llm import ChatModel
from xinyidai_agent.protocol import ChatRequest, SessionStateSnapshot


MemoryOperationType = Literal["set_slot", "clear_slot", "clear_result"]
MemoryOperationSource = Literal["user_explicit", "tool_result", "route_result", "system"]


class MemoryOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: MemoryOperationType
    slot: str | None = None
    target: str | None = None
    value: Any = None
    source: MemoryOperationSource = "user_explicit"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class UpdateSessionStateArguments(BaseModel):
    model_config = ConfigDict(frozen=True)

    operations: list[MemoryOperation] = Field(default_factory=list)


@dataclass(frozen=True)
class SessionMemoryToolExecution:
    tool_name: str
    arguments: dict[str, Any]
    status: Literal["success", "noop", "failed"]
    state: SessionStateSnapshot
    applied_operations: list[dict[str, Any]]
    rejected_operations: list[dict[str, Any]]
    error: str | None = None


class SessionMemorySystemTool:
    name = "update_session_state"
    visibility = "model_visible"

    _slot_types: dict[str, type] = {
        "company_name": str,
        "product_name": str,
        "application_id": str,
    }
    _clear_targets = {"last_credit_amount", "pending_action", "selected_product_name"}

    def plan_and_execute(
        self,
        model: ChatModel,
        request: ChatRequest,
        state: SessionStateSnapshot,
    ) -> SessionMemoryToolExecution:
        raw = model.complete(self._build_messages(request, state))
        try:
            arguments = self._parse_arguments(raw)
        except (ValueError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            return SessionMemoryToolExecution(
                tool_name=self.name,
                arguments={},
                status="noop",
                state=state,
                applied_operations=[],
                rejected_operations=[],
                error=f"模型未调用 {self.name} 或参数不可解析：{exc}",
            )

        return self.execute(state, arguments)

    def execute(
        self,
        state: SessionStateSnapshot,
        arguments: UpdateSessionStateArguments,
    ) -> SessionMemoryToolExecution:
        confirmed_slots = dict(state.confirmed_slots)
        pending_slots = dict(state.pending_slots)
        awaiting_slots = list(state.awaiting_slots)
        selected_company_name = state.selected_company_name
        selected_product_name = state.selected_product_name
        last_credit_amount = state.last_credit_amount
        pending_action = state.pending_action

        applied: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for operation in arguments.operations:
            error = self._validate_operation(operation)
            if error:
                rejected.append({"operation": operation.model_dump(), "error": error})
                continue

            if operation.op == "set_slot":
                slot = operation.slot or ""
                old_value = confirmed_slots.get(slot)
                confirmed_slots[slot] = operation.value
                pending_slots[slot] = operation.value
                awaiting_slots = [item for item in awaiting_slots if item != slot]
                if slot == "company_name":
                    selected_company_name = str(operation.value)
                    if old_value and old_value != operation.value:
                        last_credit_amount = None
                if slot == "product_name":
                    selected_product_name = str(operation.value)
                applied.append(operation.model_dump())
                continue

            if operation.op == "clear_slot":
                slot = operation.slot or ""
                confirmed_slots.pop(slot, None)
                pending_slots.pop(slot, None)
                if slot == "company_name":
                    selected_company_name = None
                    last_credit_amount = None
                if slot == "product_name":
                    selected_product_name = None
                applied.append(operation.model_dump())
                continue

            if operation.op == "clear_result":
                if operation.target == "last_credit_amount":
                    last_credit_amount = None
                if operation.target == "pending_action":
                    pending_action = None
                if operation.target == "selected_product_name":
                    selected_product_name = None
                    confirmed_slots.pop("product_name", None)
                    pending_slots.pop("product_name", None)
                applied.append(operation.model_dump())

        updated = state.model_copy(
            update={
                "confirmed_slots": confirmed_slots,
                "pending_slots": pending_slots,
                "awaiting_slots": awaiting_slots,
                "pending_action": pending_action,
                "selected_company_name": selected_company_name,
                "selected_product_name": selected_product_name,
                "last_credit_amount": last_credit_amount,
                "short_summary": self._build_summary(
                    state=state,
                    confirmed_slots=confirmed_slots,
                    awaiting_slots=awaiting_slots,
                    selected_company_name=selected_company_name,
                    selected_product_name=selected_product_name,
                    last_credit_amount=last_credit_amount,
                ),
            }
        )
        status = "success" if applied else "noop"
        return SessionMemoryToolExecution(
            tool_name=self.name,
            arguments=arguments.model_dump(),
            status=status,
            state=updated,
            applied_operations=applied,
            rejected_operations=rejected,
        )

    def _validate_operation(self, operation: MemoryOperation) -> str | None:
        if operation.confidence < 0.55:
            return "confidence 低于 0.55，不写入会话状态"

        if operation.op in {"set_slot", "clear_slot"}:
            if not operation.slot:
                return "slot 不能为空"
            if operation.slot not in self._slot_types:
                return f"不允许写入槽位 {operation.slot}"

        if operation.op == "set_slot":
            expected_type = self._slot_types[operation.slot or ""]
            if not isinstance(operation.value, expected_type):
                return f"{operation.slot} 类型应为 {expected_type.__name__}"
            if isinstance(operation.value, str) and not operation.value.strip():
                return f"{operation.slot} 不能为空字符串"

        if operation.op == "clear_result":
            if operation.target not in self._clear_targets:
                return f"不允许清理目标 {operation.target}"

        return None

    def _build_messages(
        self,
        request: ChatRequest,
        state: SessionStateSnapshot,
    ) -> list[dict[str, str]]:
        schema = {
            "tool_name": self.name,
            "arguments": {
                "operations": [
                    {
                        "op": "set_slot | clear_slot | clear_result",
                        "slot": "company_name | product_name | application_id",
                        "target": "last_credit_amount | pending_action | selected_product_name",
                        "value": "写入值",
                        "source": "user_explicit | tool_result | route_result | system",
                        "confidence": 0.0,
                        "reason": "为什么更新",
                    }
                ]
            },
        }
        return [
            {
                "role": "system",
                "content": (
                    "你是信易贷 Agent 的会话状态维护器。"
                    "你可以调用系统工具 update_session_state 更新短期业务记忆。"
                    "是否把用户输入中的信息写入 session，由你根据语义判断；不要依赖固定关键词。"
                    "只输出 JSON，不要输出解释文字。"
                    "如果用户明确切换企业主体，应 set_slot company_name，并在必要时 clear_result last_credit_amount。"
                    "如果用户只是闲聊或没有新增业务状态，operations 返回空数组。"
                    f"工具调用格式：{json.dumps(schema, ensure_ascii=False)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"当前 session_state：{state.model_dump_json()}\n"
                    f"用户本轮输入：{request.user_message}\n"
                    f"外部 metadata：{json.dumps(request.metadata, ensure_ascii=False)}"
                ),
            },
        ]

    def _parse_arguments(self, raw: str) -> UpdateSessionStateArguments:
        payload = self._parse_json(raw)
        if payload.get("tool_name") not in {self.name, None}:
            raise ValueError(f"模型调用了未知系统工具 {payload.get('tool_name')}")

        arguments = payload.get("arguments", payload)
        return UpdateSessionStateArguments.model_validate(arguments)

    def _parse_json(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("模型未返回 JSON 对象")

        return json.loads(text[start : end + 1])

    def _build_summary(
        self,
        state: SessionStateSnapshot,
        confirmed_slots: dict[str, Any],
        awaiting_slots: list[str],
        selected_company_name: str | None,
        selected_product_name: str | None,
        last_credit_amount: dict[str, Any] | None,
    ) -> str:
        parts: list[str] = []
        if state.active_scene:
            parts.append(f"active_scene={state.active_scene}")
        if state.active_capability_id:
            parts.append(f"capability={state.active_capability_id}")
        if selected_company_name:
            parts.append(f"company={selected_company_name}")
        if selected_product_name:
            parts.append(f"product={selected_product_name}")
        if last_credit_amount:
            parts.append(f"last_credit_amount={last_credit_amount.get('credit_amount')}")
        if awaiting_slots:
            parts.append(f"awaiting={','.join(awaiting_slots)}")
        if state.confirmation_status != "none":
            parts.append(f"confirmation={state.confirmation_status}")
        return "; ".join(parts)
