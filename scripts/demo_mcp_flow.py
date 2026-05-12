"""演示 MCP 工具接入 Agent ToolRegistry 的完整链路。"""

from __future__ import annotations

import os
from pathlib import Path

from xinyidai_agent.mcp import McpManager, default_config_path, load_mcp_servers_config
from xinyidai_agent.mcp.tool_adapter import McpToolAdapter
from xinyidai_agent.protocol import ChatRequest, RouteDecision, ToolCall
from xinyidai_agent.rag import AsyncRuntime
from xinyidai_agent.tools import build_tool_registry


def main() -> None:
    """启动默认 MCP 配置，调用 template 与 filesystem 两类工具。"""
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    runtime = AsyncRuntime(thread_name="demo-mcp-runtime")
    manager = McpManager(load_mcp_servers_config(default_config_path()).servers, runtime)
    try:
        manager.start()
        descriptors = manager.iter_descriptors()
        print("已注册 MCP 工具：")
        for descriptor in descriptors:
            print(f"- {descriptor.qualified_name}")

        adapters = [McpToolAdapter(descriptor, manager) for descriptor in descriptors]
        registry = build_tool_registry(mcp_adapters=adapters)
        execution = registry.execute(
            ChatRequest(user_message="生成贷款申请模板"),
            RouteDecision(
                scene="UTILITY",
                intent="UTILITY",
                confidence=1.0,
                allowed_tool_categories=["utility"],
                allowed_tools=["mcp__template__generate_markdown_template"],
                route_reason="demo mcp template",
            ),
            ToolCall(
                tool_call_id="demo-template-1",
                tool_name="mcp__template__generate_markdown_template",
                tool_category="utility",
                arguments={"scene": "loan_apply", "title": "贷款申请演示模板"},
            ),
        )
        print("\ntemplate 执行状态：", execution.result.business_status)
        print(execution.result.output.get("text_content", ""))

        sample_path = repo_root / "workspace" / "sample.csv"
        fs_result = manager.call_tool_sync(
            "mcp__filesystem__read_file",
            {"path": str(sample_path)},
        )
        print("\nfilesystem 读取 sample.csv：")
        print(fs_result.text_content)
    finally:
        manager.shutdown()
        runtime.close()


if __name__ == "__main__":
    main()
