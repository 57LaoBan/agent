from __future__ import annotations

from xinyidai_agent.protocol import ChatRequest, RouteDecision, ToolCall, ToolResult, ToolResultEnvelope
from xinyidai_agent.tools.base import SlotSpec, ToolExecution, ToolSpec


class MockCreditAmountTool:
    name = "query_credit_amount"
    category = "data_query"
    risk_level = "read_only"
    description = "查询企业 mock 授信额度、数据时间等只读数值信息。"
    requires_confirmation = False

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            category=self.category,
            risk_level=self.risk_level,
            description=self.description,
            requires_confirmation=self.requires_confirmation,
            input_slots=[
                SlotSpec("company_name", "string", description="企业名称"),
                SlotSpec("query", "string", description="用户原始问题"),
            ],
            output_slots=[
                SlotSpec("company_name", "string", description="企业名称"),
                SlotSpec("credit_amount", "string", description="授信额度展示值"),
                SlotSpec("data_time", "string", description="数据更新时间"),
            ],
            input_schema={
                "company_name": "企业名称",
                "query": "用户原始问题",
            },
        )

    def execute(
        self,
        request: ChatRequest,
        route: RouteDecision,
        tool_call: ToolCall,
    ) -> ToolExecution:
        result = ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            tool_category=self.category,
            status="success",
            output={
                "company_name": tool_call.arguments.get("company_name"),
                "credit_amount": "50万元",
                "data_time": "2026-04-28",
            },
            envelope=ToolResultEnvelope(
                success=True,
                status="CREDIT_AMOUNT_FOUND",
                code=0,
                message="授信额度查询成功。",
                data={
                    "company_name": tool_call.arguments.get("company_name"),
                    "credit_amount": "50万元",
                    "data_time": "2026-04-28",
                },
            ),
            business_status="CREDIT_AMOUNT_FOUND",
            code=0,
            message="授信额度查询成功。",
            terminal=True,
            model_observation="企业授信额度为 50 万元，可向用户说明该结果来自 mock 只读接口。",
        )
        return ToolExecution(result=result)
