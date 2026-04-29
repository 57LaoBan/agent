from __future__ import annotations

from xinyidai_agent.protocol import ChatRequest, RouteDecision, ToolCall, ToolResult
from xinyidai_agent.rag import EmptyRetriever, Retriever
from xinyidai_agent.tools.base import ToolExecution, ToolSpec


class RagSearchTool:
    name = "rag_search"
    category = "knowledge"
    risk_level = "read_only"
    description = "检索政策、产品、准入规则等知识库证据片段。"
    requires_confirmation = False

    def __init__(self, retriever: Retriever | None = None) -> None:
        self._retriever = retriever or EmptyRetriever()

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            category=self.category,
            risk_level=self.risk_level,
            description=self.description,
            requires_confirmation=self.requires_confirmation,
            input_schema={
                "query": "检索问题",
                "top_k": "返回证据数量",
            },
        )

    def execute(
        self,
        request: ChatRequest,
        route: RouteDecision,
        tool_call: ToolCall,
    ) -> ToolExecution:
        query = str(tool_call.arguments.get("query") or request.user_message)
        top_k = int(tool_call.arguments.get("top_k") or request.top_k)
        sources, retrieval_trace = self._retriever.retrieve(query, top_k)
        result = ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            tool_category=self.category,
            status="success",
            output={
                "sources": [source.model_dump() for source in sources],
                "retrieval_trace": retrieval_trace.model_dump(),
            },
            business_status="RAG_RESULT_READY",
            terminal=True,
            model_observation="知识库检索已完成，回答必须基于证据；证据不足时说明缺口。",
        )
        return ToolExecution(result=result, sources=sources, retrieval_trace=retrieval_trace)
