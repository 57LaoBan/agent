"""MCP 到 AgentTool 的适配层。"""

from __future__ import annotations

import asyncio
from typing import Any

from xinyidai_agent.mcp.manager import McpManager
from xinyidai_agent.mcp.protocol import McpToolCallResult, McpToolDescriptor
from xinyidai_agent.protocol import (
    BusinessStatus,
    ChatRequest,
    RouteDecision,
    ToolCall,
    ToolCategory,
    ToolResult,
    ToolResultEnvelope,
)
from xinyidai_agent.tools.base import SlotSpec, ToolExecution, ToolSpec


_MCP_CATEGORY: ToolCategory = "utility"


class McpToolAdapter:
    """单个 MCP 工具的本地 AgentTool 适配器。"""

    def __init__(self, descriptor: McpToolDescriptor, manager: McpManager) -> None:
        """绑定工具描述符与 manager。"""
        self._descriptor = descriptor
        self._manager = manager
        self.name: str = descriptor.qualified_name
        self.category: ToolCategory = _MCP_CATEGORY
        self.risk_level = descriptor.risk_level
        self.description = descriptor.description
        self.requires_confirmation = descriptor.risk_level != "read_only"

    def spec(self) -> ToolSpec:
        """根据 MCP inputSchema 派生 ToolSpec。"""
        is_read_only = self.risk_level == "read_only"
        return ToolSpec(
            name=self.name,
            category=self.category,
            risk_level=self.risk_level,
            description=self.description,
            requires_confirmation=self.requires_confirmation,
            input_slots=_derive_slot_specs(self._descriptor.input_schema),
            output_slots=[
                SlotSpec(
                    "text_content",
                    "string",
                    required=False,
                    description="MCP Server 返回的文本内容",
                    allow_empty=True,
                ),
                SlotSpec(
                    "structured",
                    "object",
                    required=False,
                    description="MCP Server 返回的结构化结果",
                    allow_empty=True,
                ),
            ],
            input_schema=dict(self._descriptor.input_schema),
            is_read_only=is_read_only,
            is_idempotent=is_read_only,
            is_concurrency_safe=is_read_only,
            cost_class="medium",
            max_duration_ms=30_000,
        )

    def execute(
        self,
        request: ChatRequest,
        route: RouteDecision,
        tool_call: ToolCall,
    ) -> ToolExecution:
        """同步执行 MCP 工具并产出标准 ToolResult。"""
        try:
            call_result = self._manager.call_tool_sync(self.name, dict(tool_call.arguments or {}))
        except (TimeoutError, asyncio.TimeoutError):
            return self._build_failed_execution(
                tool_call,
                business_status="TOOL_TIMEOUT",
                message=f"MCP 工具 {self.name} 调用超时。",
                model_hint="工具调用超时，请换工具或要求用户提供更精简的输入。",
            )
        except KeyError as exc:
            return self._build_failed_execution(
                tool_call,
                business_status="TOOL_UNAVAILABLE",
                message=f"MCP 工具不可用：{exc}",
                model_hint="该工具暂不可用，请考虑改用其它工具或转人工。",
            )
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            return self._build_failed_execution(
                tool_call,
                business_status="TOOL_FAILED",
                message=f"MCP 工具异常：{exc}",
                model_hint="工具内部错误，不要重复同样的调用，换工具或解释失败。",
            )
        return self._build_success_execution(tool_call, call_result)

    def _build_success_execution(
        self,
        tool_call: ToolCall,
        call_result: McpToolCallResult,
    ) -> ToolExecution:
        """根据 MCP 返回构造成败分支。"""
        if call_result.is_error:
            return self._build_failed_execution(
                tool_call,
                business_status="TOOL_FAILED",
                message=call_result.text_content or "MCP 工具返回 isError=true。",
                model_hint="工具明确报错，请不要继续用相同参数调用。",
            )

        output: dict[str, Any] = {
            "text_content": call_result.text_content,
            "structured": call_result.structured,
        }
        has_payload = bool(call_result.text_content) or bool(call_result.structured)
        business_status: BusinessStatus = "OK" if has_payload else "PARTIAL_DATA"
        message = f"MCP 工具 {self.name} 执行完成。" if has_payload else f"MCP 工具 {self.name} 未返回有效内容。"
        result = ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            tool_category=self.category,
            status="success",
            output=output,
            envelope=ToolResultEnvelope(
                success=True,
                status=business_status,
                code=0,
                message=message,
                data=output,
            ),
            business_status=business_status,
            code=0,
            message=message,
            terminal=True,
            model_observation=(
                "MCP 工具执行完成，可基于 text_content / structured 组织答复。"
                if has_payload
                else "MCP 工具未返回内容，请考虑换工具或转人工。"
            ),
            user_visible_message=None,
        )
        return ToolExecution(result=result)

    def _build_failed_execution(
        self,
        tool_call: ToolCall,
        *,
        business_status: BusinessStatus,
        message: str,
        model_hint: str,
    ) -> ToolExecution:
        """统一构造失败分支，业务状态必须来自 BusinessStatus。"""
        result = ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            tool_category=self.category,
            status="failed",
            output={},
            envelope=ToolResultEnvelope(
                success=False,
                status=business_status,
                code=-1,
                message=message,
                data={},
            ),
            business_status=business_status,
            code=-1,
            message=message,
            terminal=True,
            model_observation=model_hint,
            user_visible_message=None,
        )
        return ToolExecution(result=result)


def _derive_slot_specs(schema: dict[str, Any]) -> list[SlotSpec]:
    """从 JSON Schema 顶层 properties 派生 SlotSpec。"""
    if not schema:
        return []
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    slots: list[SlotSpec] = []
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            continue
        slots.append(
            SlotSpec(
                name=str(prop_name),
                value_type=_map_json_type(prop_schema.get("type")),
                required=prop_name in required,
                description=str(prop_schema.get("description") or ""),
                allow_empty=False,
            )
        )
    return slots


def _map_json_type(json_type: Any) -> str:
    """把 JSON Schema type 映射到现有 SlotSpec.value_type 词表。"""
    if isinstance(json_type, list):
        for candidate in json_type:
            if candidate and candidate != "null":
                return str(candidate)
        return "string"
    if isinstance(json_type, str):
        return json_type
    return "string"
