from __future__ import annotations

from xinyidai_agent.protocol import ChatRequest, RouteDecision, ToolCall, ToolResult
from xinyidai_agent.runtime import ControlledAgentLoop
from xinyidai_agent.tools.base import SlotSpec, ToolExecution, ToolSpec
from xinyidai_agent.tools.registry import ToolRegistry

from test.e2e.test_react_loop_matrix import CreateAuthorizationLinkTool, MatrixAnswerModel, MatrixRouter


class CreditUnavailableTool:
    """返回无额度数据的授信查询工具。"""

    name = "query_credit_amount"
    category = "data_query"
    risk_level = "read_only"
    description = "授信额度查询不可用。"
    requires_confirmation = False

    def spec(self) -> ToolSpec:
        """声明只读、幂等、并发安全的不可用数据查询工具。"""
        return ToolSpec(
            name=self.name,
            category=self.category,
            risk_level=self.risk_level,
            description=self.description,
            input_slots=[SlotSpec("company_name", "string"), SlotSpec("query", "string", required=False)],
            output_slots=[],
            is_read_only=True,
            is_idempotent=True,
            is_concurrency_safe=True,
            cost_class="cheap",
            max_duration_ms=1000,
            input_schema={"company_name": "企业名称", "query": "用户问题"},
        )

    def execute(self, request: ChatRequest, route: RouteDecision, tool_call: ToolCall) -> ToolExecution:
        """明确返回无数据，不返回任何具体额度。"""
        return ToolExecution(
            result=ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=self.name,
                tool_category=self.category,
                status="success",
                business_status="NOT_FOUND",
                output={"company_name": tool_call.arguments.get("company_name")},
                terminal=True,
                user_visible_message="暂未获取到该企业的授信额度数据，不能给出具体金额。",
            )
        )


def test_unavailable_tool_does_not_yield_concrete_numbers() -> None:
    """产品参数工具未注册时，不得编造利率、额度或期限。"""
    loop = ControlledAgentLoop(
        model=MatrixAnswerModel(),
        router=MatrixRouter(),
        tool_registry=ToolRegistry([]),
    )

    response = loop.answer(ChatRequest(user_message="小微税贷的利率是多少", session_id="hallucination-rate"))

    assert response.business_status == "TOOL_UNAVAILABLE"
    assert not any(token in response.answer for token in ("3.5%", "4.0%", "4.5%", "5.0%", "12个月", "100万"))
    assert any(token in response.answer for token in ("未上线", "暂未", "无法"))


def test_credit_amount_without_data_does_not_fabricate() -> None:
    """授信额度无数据时不能编造金额。"""
    loop = ControlledAgentLoop(
        model=MatrixAnswerModel(),
        router=MatrixRouter(),
        tool_registry=ToolRegistry([CreditUnavailableTool()]),
    )

    response = loop.answer(ChatRequest(user_message="查一下 ABC 公司的授信额度", session_id="hallucination-credit"))

    assert response.business_status == "NOT_FOUND"
    assert not any(token in response.answer for token in ("50万", "100万", "200万", "万元"))
    assert "不能给出具体金额" in response.answer


def test_authorization_link_does_not_yield_fake_url() -> None:
    """授权链接确认前不得在 answer 中给出假 URL。"""
    loop = ControlledAgentLoop(
        model=MatrixAnswerModel(),
        router=MatrixRouter(),
        tool_registry=ToolRegistry([CreateAuthorizationLinkTool()]),
    )

    response = loop.answer(ChatRequest(user_message="帮我给 ABC 公司生成授权链接", session_id="hallucination-auth"))

    assert response.stop_reason == "waiting_confirmation"
    assert response.pending_action is not None
    assert "http://" not in response.answer
    assert "https://" not in response.answer
