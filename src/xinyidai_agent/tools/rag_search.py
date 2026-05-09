from __future__ import annotations

from xinyidai_agent.protocol import ChatRequest, RouteDecision, ToolCall, ToolResult, ToolResultEnvelope
from xinyidai_agent.rag import EmptyRetriever, Retriever
from xinyidai_agent.rag.citation import CitationGenerator
from xinyidai_agent.tools.base import SlotSpec, ToolExecution, ToolSpec


class RagSearchTool:
    name = "rag_search"
    category = "knowledge"
    risk_level = "read_only"
    description = "检索政策、产品、准入规则等知识库证据片段。"
    requires_confirmation = False

    def __init__(self, retriever: Retriever | None = None) -> None:
        self._retriever = retriever or EmptyRetriever()
        self._citation_generator = CitationGenerator()

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            category=self.category,
            risk_level=self.risk_level,
            description=self.description,
            requires_confirmation=self.requires_confirmation,
            is_read_only=True,
            is_idempotent=True,
            is_concurrency_safe=True,
            cost_class="cheap",
            max_duration_ms=3000,
            input_slots=[
                SlotSpec("query", "string", description="检索问题"),
                SlotSpec("top_k", "integer", required=False, description="返回证据数量"),
            ],
            output_slots=[
                SlotSpec("sources", "array", description="检索证据列表", allow_empty=True),
                SlotSpec("retrieval_trace", "object", description="检索过程 trace"),
                SlotSpec(
                    "citation_context",
                    "string",
                    required=False,
                    description="带编号引用的 Prompt 上下文",
                    allow_empty=True,
                ),
            ],
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
        citation_context = self._citation_generator.format_context(sources)
        has_sources = bool(sources)
        business_status = "RAG_RESULT_READY" if has_sources else "PARTIAL_DATA"
        message = "知识库检索完成。" if has_sources else "知识库暂无可用证据。"
        result = ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            tool_category=self.category,
            status="success",
            output={
                "sources": [source.model_dump() for source in sources],
                "retrieval_trace": retrieval_trace.model_dump(),
                "citation_context": citation_context,
            },
            envelope=ToolResultEnvelope(
                success=True,
                status=business_status,
                code=0,
                message=message,
                data={
                    "sources": [source.model_dump() for source in sources],
                    "retrieval_trace": retrieval_trace.model_dump(),
                    "citation_context": citation_context,
                },
            ),
            business_status=business_status,
            code=0,
            message=message,
            terminal=True,
            model_observation=(
                "知识库检索已完成，回答必须基于证据。"
                if has_sources
                else "知识库未返回证据，不能编造政策或准入结论。"
            ),
            user_visible_message=(
                None
                if has_sources
                else "当前知识库还没有检索到可用证据，暂时不能确认该政策或准入规则。您可以换一种问法，或等真实 RAG 数据接入后再查询。"
            ),
        )
        return ToolExecution(result=result, sources=sources, retrieval_trace=retrieval_trace)
