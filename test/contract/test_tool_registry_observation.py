from __future__ import annotations

from xinyidai_agent.capabilities.catalog import default_capability_catalog
from xinyidai_agent.protocol import ChatRequest, RouteDecision, ToolCall, ToolResult, ToolResultEnvelope
from xinyidai_agent.tools.base import SlotSpec, ToolExecution, ToolSpec
from xinyidai_agent.tools.mock_credit import MockCreditAmountTool
from xinyidai_agent.tools.rag_search import RagSearchTool
from xinyidai_agent.tools.registry import ToolRegistry, default_tool_registry


class ExplodingTool:
    """测试用失败工具，模拟后端运行时异常。"""

    name = "explode"
    category = "utility"
    risk_level = "read_only"
    description = "模拟失败工具。"
    requires_confirmation = False

    def spec(self) -> ToolSpec:
        """返回失败工具规格。"""
        return ToolSpec(
            name=self.name,
            category=self.category,
            risk_level=self.risk_level,
            description=self.description,
            is_read_only=True,
            is_idempotent=True,
            is_concurrency_safe=True,
            input_slots=[SlotSpec("query", "string", required=False)],
            output_slots=[],
        )

    def execute(self, request: ChatRequest, route: RouteDecision, tool_call: ToolCall) -> ToolExecution:
        """执行时抛出运行时错误。"""
        raise RuntimeError("backend 500")


class EmptyStatusTool:
    """测试用空结果工具，模拟 NOT_FOUND。"""

    name = "empty_status"
    category = "status"
    risk_level = "read_only"
    description = "模拟空状态查询。"
    requires_confirmation = False

    def spec(self) -> ToolSpec:
        """返回空结果工具规格。"""
        return ToolSpec(
            name=self.name,
            category=self.category,
            risk_level=self.risk_level,
            description=self.description,
            is_read_only=True,
            is_idempotent=True,
            is_concurrency_safe=True,
            output_slots=[SlotSpec("status", "string", required=False, allow_empty=True)],
        )

    def execute(self, request: ChatRequest, route: RouteDecision, tool_call: ToolCall) -> ToolExecution:
        """返回业务空结果。"""
        result = ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            tool_category=self.category,
            status="success",
            output={},
            envelope=ToolResultEnvelope(success=True, status="NOT_FOUND", message="未找到申请状态。"),
            business_status="NOT_FOUND",
            message="未找到申请状态。",
            terminal=True,
        )
        return ToolExecution(result=result)


def _capability(capability_id: str):
    """从默认目录取 capability。"""
    return {policy.capability_id: policy for policy in default_capability_catalog()}[capability_id]


def _route(allowed_tools: list[str], categories: list[str]) -> RouteDecision:
    """构造测试路由。"""
    return RouteDecision(
        scene="DATA_QUERY",
        intent="CREDIT_LIMIT_QUERY",
        confidence=0.95,
        filled_slots={"company_name": "杭州示例科技有限公司"},
        allowed_tools=allowed_tools,
        allowed_tool_categories=categories,
        route_reason="测试路由。",
        should_call_tool=True,
    )


def _call(tool_name: str, arguments: dict | None = None, category: str | None = None) -> ToolCall:
    """构造工具调用。"""
    return ToolCall(
        tool_call_id=f"call-{tool_name}",
        tool_name=tool_name,
        tool_category=category,
        arguments=arguments or {},
    )


def test_unavailable_tool_returns_observation() -> None:
    """工具未注册时返回 unavailable observation。"""
    registry = default_tool_registry()
    execution, observation = registry.execute_observed(
        ChatRequest(user_message="小微税贷利率是多少？"),
        _route(["query_product_terms"], ["data_query"]),
        _call("query_product_terms", {"query": "小微税贷利率"}, "data_query"),
        capability=_capability("product.terms.read"),
    )

    assert execution.result.business_status == "TOOL_UNAVAILABLE"
    assert observation.kind == "unavailable"
    assert "query_product_terms" not in observation.alternative_tools
    assert observation.hint


def test_blocked_tool_returns_observation() -> None:
    """工具不在白名单时返回 blocked observation。"""
    registry = default_tool_registry()
    _, observation = registry.execute_observed(
        ChatRequest(user_message="查额度"),
        _route(["rag_search"], ["knowledge"]),
        _call("query_credit_amount", {"company_name": "杭州示例科技有限公司"}, "data_query"),
        capability=_capability("credit.limit.read"),
    )

    assert observation.kind == "blocked"


def test_schema_error_returns_observation() -> None:
    """输入 schema 错误时返回 schema_error observation。"""
    registry = default_tool_registry()
    _, observation = registry.execute_observed(
        ChatRequest(user_message="查额度"),
        _route(["query_credit_amount"], ["data_query"]),
        _call("query_credit_amount", {}, "data_query"),
        capability=_capability("credit.limit.read"),
    )

    assert observation.kind == "schema_error"


def test_failed_tool_returns_observation() -> None:
    """工具运行时异常时返回 failed observation。"""
    registry = ToolRegistry([ExplodingTool()])
    _, observation = registry.execute_observed(
        ChatRequest(user_message="测试失败"),
        _route(["explode"], ["utility"]),
        _call("explode", {}, "utility"),
        capability=_capability("unknown.clarify"),
    )

    assert observation.kind == "failed"
    assert "backend 500" in (observation.reason or "")


def test_empty_result_returns_observation() -> None:
    """工具成功但业务为空时返回 empty observation。"""
    registry = ToolRegistry([EmptyStatusTool()])
    _, observation = registry.execute_observed(
        ChatRequest(user_message="查状态"),
        _route(["empty_status"], ["status"]),
        _call("empty_status", {}, "status"),
        capability=_capability("application.status.read"),
    )

    assert observation.kind == "empty"
    assert observation.sources_count == 0


def test_success_result_returns_observation() -> None:
    """工具成功且有数据时返回 success observation。"""
    registry = ToolRegistry([MockCreditAmountTool(), RagSearchTool()])
    _, observation = registry.execute_observed(
        ChatRequest(user_message="查额度"),
        _route(["query_credit_amount"], ["data_query"]),
        _call("query_credit_amount", {"company_name": "杭州示例科技有限公司"}, "data_query"),
        capability=_capability("credit.limit.read"),
    )

    assert observation.kind == "success"
    assert observation.business_status == "CREDIT_AMOUNT_FOUND"
    assert "credit_amount=50万元" in observation.summary
    assert len(observation.summary) <= 500
