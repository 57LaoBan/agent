from __future__ import annotations

from collections.abc import Iterable

from xinyidai_agent.protocol import (
    ChatRequest,
    RouteDecision,
    ToolCall,
    ToolCategory,
    ToolResult,
    ToolResultEnvelope,
)
from xinyidai_agent.rag import Retriever
from xinyidai_agent.tools.base import AgentTool, SlotSpec, ToolExecution, ToolSpec
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
                envelope=ToolResultEnvelope(
                    success=False,
                    status="ERROR",
                    code=404,
                    message=f"未注册工具：{tool_call.tool_name}",
                ),
                error_message=f"未注册工具：{tool_call.tool_name}",
                business_status="ERROR",
                code=404,
                message=f"未注册工具：{tool_call.tool_name}",
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
                envelope=ToolResultEnvelope(
                    success=False,
                    status="TOOL_BLOCKED",
                    code=403,
                    message=f"当前路由不允许调用工具：{tool_call.tool_name}",
                ),
                error_message=f"当前路由不允许调用工具：{tool_call.tool_name}",
                business_status="TOOL_BLOCKED",
                code=403,
                message=f"当前路由不允许调用工具：{tool_call.tool_name}",
                terminal=True,
                user_visible_message="当前请求不能执行这个工具动作。",
                model_observation="工具被注册表拦截：该工具不在当前路由允许的工具名或工具分类内。",
            )
            return ToolExecution(result=result)

        spec = tool.spec()
        if tool_call.tool_category and tool_call.tool_category != tool.category:
            result = self._blocked_result(
                tool_call,
                tool.category,
                f"工具分类不匹配：调用方={tool_call.tool_category}，注册表={tool.category}",
                "工具分类和注册表声明不一致，已拦截执行。",
            )
            return ToolExecution(result=result)

        input_errors = self._validate_slots(tool_call.arguments, spec.input_slots)
        if input_errors:
            result = self._blocked_result(
                tool_call,
                tool.category,
                f"工具输入槽位校验失败：{'; '.join(input_errors)}",
                "工具输入槽位校验失败，未执行工具。",
            )
            return ToolExecution(result=result)

        execution = tool.execute(request, route, tool_call)
        output_errors = self._validate_slots(execution.result.output, spec.output_slots)
        if output_errors:
            result = ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                tool_category=tool.category,
                status="failed",
                envelope=ToolResultEnvelope(
                    success=False,
                    status="TOOL_SCHEMA_ERROR",
                    code=422,
                    message=f"工具输出槽位校验失败：{'; '.join(output_errors)}",
                    data=execution.result.output,
                ),
                error_message=f"工具输出槽位校验失败：{'; '.join(output_errors)}",
                business_status="TOOL_SCHEMA_ERROR",
                code=422,
                message=f"工具输出槽位校验失败：{'; '.join(output_errors)}",
                terminal=True,
                user_visible_message="工具返回结果不完整，请稍后再试。",
                model_observation="工具输出槽位校验失败，不能基于不完整结果回答用户。",
            )
            return ToolExecution(result=result, sources=execution.sources, retrieval_trace=execution.retrieval_trace)

        return execution

    def _is_allowed(self, route: RouteDecision, tool: AgentTool) -> bool:
        if not route.allowed_tools and not route.allowed_tool_categories:
            return True

        return tool.name in route.allowed_tools or tool.category in route.allowed_tool_categories

    def _validate_slots(self, values: dict[str, object], slots: list[SlotSpec]) -> list[str]:
        errors: list[str] = []
        for slot in slots:
            value = values.get(slot.name)
            if slot.required and not self._has_value(value, allow_empty=slot.allow_empty):
                errors.append(f"{slot.name} 缺失")
                continue
            if not self._has_value(value, allow_empty=slot.allow_empty):
                continue
            if not self._matches_type(value, slot.value_type):
                errors.append(f"{slot.name} 类型应为 {slot.value_type}，实际为 {type(value).__name__}")
        return errors

    def _has_value(self, value: object, allow_empty: bool = False) -> bool:
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        if isinstance(value, list | dict | tuple | set) and not value:
            return allow_empty
        return True

    def _matches_type(self, value: object, value_type: str) -> bool:
        if value_type == "string":
            return isinstance(value, str)
        if value_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if value_type == "number":
            return isinstance(value, int | float) and not isinstance(value, bool)
        if value_type == "boolean":
            return isinstance(value, bool)
        if value_type == "object":
            return isinstance(value, dict)
        if value_type == "array":
            return isinstance(value, list)
        return True

    def _blocked_result(
        self,
        tool_call: ToolCall,
        tool_category: ToolCategory,
        error_message: str,
        model_observation: str,
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            tool_category=tool_category,
            status="blocked",
            envelope=ToolResultEnvelope(
                success=False,
                status="TOOL_BLOCKED",
                code=422,
                message=error_message,
            ),
            error_message=error_message,
            business_status="TOOL_BLOCKED",
            code=422,
            message=error_message,
            terminal=True,
            user_visible_message="当前请求不能执行这个工具动作。",
            model_observation=model_observation,
        )


def default_tool_registry(retriever: Retriever | None = None) -> ToolRegistry:
    return ToolRegistry(
        [
            MockCreditAmountTool(),
            RagSearchTool(retriever=retriever),
        ]
    )
