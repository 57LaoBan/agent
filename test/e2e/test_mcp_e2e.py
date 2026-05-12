"""MCP 端到端测试：真实 MCP Server 通过 ToolRegistry 执行。"""

from __future__ import annotations

from pathlib import Path

from xinyidai_agent.mcp import McpManager
from xinyidai_agent.mcp.protocol import McpServerConfig
from xinyidai_agent.mcp.tool_adapter import McpToolAdapter
from xinyidai_agent.protocol import ChatRequest, RouteDecision, ToolCall
from xinyidai_agent.rag import AsyncRuntime
from xinyidai_agent.tools import build_tool_registry


def test_template_tool_executes_through_registry(tmp_path: Path) -> None:
    """MCP 工具必须能被注册表以标准 ToolCall 调用并返回标准 ToolResult。"""
    runtime = AsyncRuntime(thread_name="test-mcp-e2e-runtime")
    manager = McpManager(
        [
            McpServerConfig(
                name="template",
                command="python",
                args=["-m", "xinyidai_mcp_servers.template_server.server"],
                env={"TEMPLATE_OUTPUT_DIR": str(tmp_path)},
                allowed_tools=["generate_markdown_template"],
                risk_level="read_only",
                description="测试模板 MCP Server",
            )
        ],
        runtime,
    )
    try:
        manager.start()
        adapters = [
            McpToolAdapter(descriptor, manager)
            for descriptor in manager.iter_descriptors()
        ]
        registry = build_tool_registry(mcp_adapters=adapters)
        request = ChatRequest(user_message="生成贷款申请模板")
        route = RouteDecision(
            scene="UTILITY",
            intent="UTILITY",
            confidence=1.0,
            allowed_tool_categories=["utility"],
            allowed_tools=["mcp__template__generate_markdown_template"],
            route_reason="mcp e2e",
        )
        tool_call = ToolCall(
            tool_call_id="mcp-e2e-1",
            tool_name="mcp__template__generate_markdown_template",
            tool_category="utility",
            arguments={"scene": "loan_apply", "title": "贷款申请端到端模板"},
        )

        execution = registry.execute(request, route, tool_call)

        assert execution.result.status == "success"
        assert execution.result.business_status == "OK"
        assert "Markdown" in execution.result.output["text_content"]
        assert len(list(tmp_path.glob("loan_apply_*.md"))) == 1
    finally:
        manager.shutdown()
        runtime.close()
