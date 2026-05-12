"""MCP 连通性冒烟测试：真实拉起本项目 template server。"""

from __future__ import annotations

from pathlib import Path

from xinyidai_agent.mcp import McpManager
from xinyidai_agent.mcp.protocol import McpServerConfig
from xinyidai_agent.rag import AsyncRuntime


def test_template_server_lists_and_calls_tools(tmp_path: Path) -> None:
    """template server 必须能完成 list_tools 与 tools/call。"""
    runtime = AsyncRuntime(thread_name="test-mcp-template-runtime")
    manager = McpManager(
        [
            McpServerConfig(
                name="template",
                command="python",
                args=["-m", "xinyidai_mcp_servers.template_server.server"],
                env={"TEMPLATE_OUTPUT_DIR": str(tmp_path)},
                allowed_tools=[
                    "generate_markdown_template",
                    "generate_excel_template",
                    "list_generated_templates",
                ],
                risk_level="read_only",
                description="测试模板 MCP Server",
            )
        ],
        runtime,
    )
    try:
        manager.start()
        tool_names = {descriptor.qualified_name for descriptor in manager.iter_descriptors()}
        assert tool_names == {
            "mcp__template__generate_markdown_template",
            "mcp__template__generate_excel_template",
            "mcp__template__list_generated_templates",
        }

        result = manager.call_tool_sync(
            "mcp__template__generate_markdown_template",
            {"scene": "loan_apply", "title": "贷款申请测试模板"},
        )

        assert result.is_error is False
        assert "Markdown" in result.text_content
        assert len(list(tmp_path.glob("loan_apply_*.md"))) == 1
    finally:
        manager.shutdown()
        runtime.close()
