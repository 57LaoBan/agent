"""MCP 适配层契约测试：JSON Schema 转槽位、执行结果转 ToolResult。"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from xinyidai_agent.mcp.protocol import McpToolCallResult, McpToolDescriptor
from xinyidai_agent.mcp.tool_adapter import McpToolAdapter
from xinyidai_agent.protocol import ChatRequest, RouteDecision, ToolCall


def _descriptor(input_schema: dict[str, Any] | None = None) -> McpToolDescriptor:
    """生成只读 MCP 工具描述。"""
    return McpToolDescriptor(
        server_name="template",
        tool_name="generate_markdown_template",
        qualified_name="mcp__template__generate_markdown_template",
        description="测试模板工具",
        input_schema=input_schema or {},
        risk_level="read_only",
    )


def _tool_call(arguments: dict[str, Any] | None = None) -> ToolCall:
    """生成标准工具调用。"""
    return ToolCall(
        tool_call_id="t1",
        tool_name="mcp__template__generate_markdown_template",
        tool_category="utility",
        arguments=arguments or {"scene": "loan_apply"},
        risk_level="read_only",
    )


def _route() -> RouteDecision:
    """生成允许 utility 工具执行的路由结果。"""
    return RouteDecision(
        scene="UTILITY",
        intent="UTILITY",
        confidence=1.0,
        allowed_tools=["mcp__template__generate_markdown_template"],
        allowed_tool_categories=["utility"],
        risk_level="read_only",
        route_reason="test",
    )


def _request() -> ChatRequest:
    """生成最小聊天请求。"""
    return ChatRequest(user_message="生成模板", top_k=3)


class TestSpecDerivation:
    """MCP inputSchema 到 ToolSpec 的派生契约。"""

    def test_required_slot_from_schema(self) -> None:
        """required 字段必须落入 input_slots。"""
        adapter = McpToolAdapter(
            _descriptor(
                {
                    "type": "object",
                    "properties": {
                        "scene": {"type": "string", "description": "场景"},
                        "count": {"type": "integer", "description": "数量"},
                    },
                    "required": ["scene"],
                }
            ),
            MagicMock(),
        )
        spec = adapter.spec()
        slots = {slot.name: slot for slot in spec.input_slots}
        assert slots["scene"].required is True
        assert slots["scene"].value_type == "string"
        assert slots["count"].required is False
        assert slots["count"].value_type == "integer"
        assert spec.is_read_only is True
        assert spec.is_concurrency_safe is True


class TestExecutionMapping:
    """MCP 调用结果到标准工具结果的映射契约。"""

    def test_success_result(self) -> None:
        """MCP 文本结果必须映射为 OK。"""
        manager = MagicMock()
        manager.call_tool_sync.return_value = McpToolCallResult(
            is_error=False,
            text_content="已生成 Markdown 模板：.data/template.md",
            structured={},
            raw_content=[],
        )
        adapter = McpToolAdapter(_descriptor(), manager)

        execution = adapter.execute(_request(), _route(), _tool_call())

        assert execution.result.status == "success"
        assert execution.result.business_status == "OK"
        assert execution.result.output["text_content"].startswith("已生成 Markdown")
        manager.call_tool_sync.assert_called_once_with(
            "mcp__template__generate_markdown_template",
            {"scene": "loan_apply"},
        )

    def test_empty_success_maps_to_partial_data(self) -> None:
        """MCP 空返回不能宣称 OK，必须降级为 PARTIAL_DATA。"""
        manager = MagicMock()
        manager.call_tool_sync.return_value = McpToolCallResult(is_error=False)
        adapter = McpToolAdapter(_descriptor(), manager)

        execution = adapter.execute(_request(), _route(), _tool_call())

        assert execution.result.status == "success"
        assert execution.result.business_status == "PARTIAL_DATA"

    def test_is_error_maps_to_tool_failed(self) -> None:
        """Server 显式 isError 必须映射为 TOOL_FAILED。"""
        manager = MagicMock()
        manager.call_tool_sync.return_value = McpToolCallResult(
            is_error=True,
            text_content="业务参数错误",
        )
        adapter = McpToolAdapter(_descriptor(), manager)

        execution = adapter.execute(_request(), _route(), _tool_call())

        assert execution.result.status == "failed"
        assert execution.result.business_status == "TOOL_FAILED"

    def test_timeout_maps_to_tool_timeout(self) -> None:
        """超时必须保留为独立业务状态，便于运行时兜底。"""
        manager = MagicMock()
        manager.call_tool_sync.side_effect = asyncio.TimeoutError()
        adapter = McpToolAdapter(_descriptor(), manager)

        execution = adapter.execute(_request(), _route(), _tool_call())

        assert execution.result.status == "failed"
        assert execution.result.business_status == "TOOL_TIMEOUT"

    def test_unknown_tool_maps_to_unavailable(self) -> None:
        """工具未注册必须映射为 TOOL_UNAVAILABLE。"""
        manager = MagicMock()
        manager.call_tool_sync.side_effect = KeyError("missing")
        adapter = McpToolAdapter(_descriptor(), manager)

        execution = adapter.execute(_request(), _route(), _tool_call())

        assert execution.result.status == "failed"
        assert execution.result.business_status == "TOOL_UNAVAILABLE"
