"""单工具快速通道：符合条件的 capability 跳过 ReAct 多轮。"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.evidence_policy import EvidencePolicy
from xinyidai_agent.llm import ChatModel
from xinyidai_agent.protocol import (
    ChatRequest,
    RetrievalTrace,
    RouteDecision,
    SessionStateSnapshot,
    SourceDocument,
    ToolCall,
    ToolExecutionObservation,
    ToolResult,
)
from xinyidai_agent.tools.base import ToolExecution
from xinyidai_agent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class FastPathOutcome:
    """fast path 执行结果。"""

    succeeded: bool
    final_answer: str | None
    sources: list[SourceDocument]
    tool_result: ToolResult | None
    tool_execution: ToolExecution | None
    observation: ToolExecutionObservation | None
    fallback_reason: str | None = None


class FastPathRunner:
    """单工具快速通道执行器。"""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        model: ChatModel,
        evidence_policy: EvidencePolicy | None = None,
    ) -> None:
        """初始化快速通道依赖。"""
        self._tools = tool_registry
        self._model = model
        self._evidence = evidence_policy or EvidencePolicy()

    def run(
        self,
        request: ChatRequest,
        route: RouteDecision,
        capability: CapabilityPolicy,
        session_state: SessionStateSnapshot,
    ) -> FastPathOutcome:
        """执行快速通道，失败时返回 fallback_reason 交给 ReAct 多轮。"""
        if not capability.is_fast_path_eligible_for_route(route.filled_slots):
            return FastPathOutcome(
                succeeded=False,
                final_answer=None,
                sources=[],
                tool_result=None,
                tool_execution=None,
                observation=None,
                fallback_reason="not_fast_path_eligible",
            )

        tool_name = capability.allowed_tools[0]
        tool_call = ToolCall(
            tool_call_id=str(uuid4()),
            tool_name=tool_name,
            tool_category=route.allowed_tool_categories[0] if route.allowed_tool_categories else None,
            arguments=self._build_arguments(capability, route, request),
            risk_level=capability.risk_level,
            confirmation_required=False,
            reason="fast_path",
        )
        execution, observation = self._tools.execute_observed(
            request,
            route,
            tool_call,
            capability=capability,
            session_state=session_state,
        )

        if observation.kind != "success":
            return FastPathOutcome(
                succeeded=False,
                final_answer=None,
                sources=list(execution.sources),
                tool_result=execution.result,
                tool_execution=execution,
                observation=observation,
                fallback_reason=f"observation_kind_{observation.kind}",
            )

        answer = self._compose_answer(request, route, execution.result, execution.sources)
        return FastPathOutcome(
            succeeded=True,
            final_answer=answer,
            sources=list(execution.sources),
            tool_result=execution.result,
            tool_execution=execution,
            observation=observation,
        )

    def _build_arguments(
        self,
        capability: CapabilityPolicy,
        route: RouteDecision,
        request: ChatRequest,
    ) -> dict[str, object]:
        """根据 capability 显式构造工具参数。"""
        if capability.capability_id == "knowledge.policy.read":
            return {"query": request.user_message, "top_k": request.top_k or 5}
        if capability.capability_id == "application.status.read":
            return {"company_name": route.filled_slots.get("company_name", "")}
        raise ValueError(f"未知 fast path capability: {capability.capability_id}")

    def _compose_answer(
        self,
        request: ChatRequest,
        route: RouteDecision,
        result: ToolResult,
        sources: list[SourceDocument],
    ) -> str:
        """基于工具结果生成最终回答。"""
        if result.user_visible_message:
            return result.user_visible_message

        trace = self._extract_retrieval_trace(result)
        evidence = self._evidence.assess(route, sources, trace)
        if not evidence.ready and evidence.user_message:
            return evidence.user_message

        messages = [
            {
                "role": "system",
                "content": (
                    "你是信易贷聊天助手。必须基于工具结果和证据回答用户问题；"
                    "证据不足时说明缺口，不得编造政策、额度、审批或授权状态。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{request.user_message}\n"
                    f"路由场景：{route.scene}\n"
                    f"工具结果：{result.model_dump()}\n"
                    f"可用证据：{self._format_sources(sources)}"
                ),
            },
        ]
        answer = self._model.complete(messages)
        return self._evidence.attach_citations(answer, sources)

    @staticmethod
    def _extract_retrieval_trace(result: ToolResult) -> RetrievalTrace | None:
        """从工具输出中恢复检索链路。"""
        raw = result.output.get("retrieval_trace")
        if isinstance(raw, dict):
            return RetrievalTrace.model_validate(raw)
        return None

    @staticmethod
    def _format_sources(sources: list[SourceDocument]) -> str:
        """将证据压缩成模型可读文本。"""
        if not sources:
            return "暂无检索证据。"
        return "\n\n".join(
            f"[{index}] {source.title}\n{source.content}"
            for index, source in enumerate(sources, start=1)
        )
