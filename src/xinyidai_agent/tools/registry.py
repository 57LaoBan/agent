from __future__ import annotations

from collections.abc import Iterable

from xinyidai_agent.protocol import ChatRequest, RouteDecision, ToolCall, ToolCategory, ToolResult
from xinyidai_agent.rag import Retriever
from xinyidai_agent.tools.base import AgentTool, ToolExecution, ToolSpec
from xinyidai_agent.tools.mock_credit import MockCreditAmountTool
from xinyidai_agent.tools.rag_search import RagSearchTool


class ToolRegistry:
    def __init__(self, tools: Iterable[AgentTool]) -> None:
        tool_list = list(tools)
        self._tools = {tool.name: tool for tool in tool_list}
        self._tools_by_category: dict[ToolCategory, list[AgentTool]] = {}
        for tool in tool_list:
            self._tools_by_category.setdefault(tool.category, []).append(tool)

    def names_for_categories(self, categories: Iterable[ToolCategory]) -> list[str]:
        names: list[str] = []
        for category in categories:
            names.extend(tool.name for tool in self._tools_by_category.get(category, []))
        return names

    def specs_for_categories(self, categories: Iterable[ToolCategory]) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for category in categories:
            specs.extend(tool.spec() for tool in self._tools_by_category.get(category, []))
        return specs

    def execute(
        self,
        request: ChatRequest,
        route: RouteDecision,
        tool_call: ToolCall,
    ) -> ToolExecution:
        tool = self._tools.get(tool_call.tool_name)
        if tool is None:
            result = ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                tool_category=tool_call.tool_category,
                status="failed",
                error_message=f"未注册工具：{tool_call.tool_name}",
                terminal=True,
                user_visible_message="当前工具尚未接入，请稍后再试。",
            )
            return ToolExecution(result=result)

        if not self._is_allowed(route, tool):
            result = ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                tool_category=tool.category,
                status="blocked",
                error_message=f"当前路由不允许调用工具：{tool_call.tool_name}",
                terminal=True,
                user_visible_message="当前请求不能执行这个工具动作。",
                model_observation="工具被注册表拦截：该工具不在当前路由允许的工具名或工具分类内。",
            )
            return ToolExecution(result=result)

        return tool.execute(request, route, tool_call)

    def _is_allowed(self, route: RouteDecision, tool: AgentTool) -> bool:
        if not route.allowed_tools and not route.allowed_tool_categories:
            return True

        return tool.name in route.allowed_tools or tool.category in route.allowed_tool_categories


def default_tool_registry(retriever: Retriever | None = None) -> ToolRegistry:
    return ToolRegistry(
        [
            MockCreditAmountTool(),
            RagSearchTool(retriever=retriever),
        ]
    )
