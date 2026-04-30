from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from xinyidai_agent.llm import ChatModel
from xinyidai_agent.memory import MemoryManager
from xinyidai_agent.protocol import (
    AgentEvent,
    AgentEventType,
    AgentState,
    ChatRequest,
    ChatResponse,
    DiagnosticEvent,
    EvidenceState,
    EventVisibility,
    ModelDecision,
    PendingAction,
    PerformanceSummary,
    PerformanceStage,
    RetrievalTrace,
    RouteDecision,
    SessionStateSnapshot,
    SourceDocument,
    StateTransition,
    ToolCall,
    ToolResult,
)
from xinyidai_agent.router import ControlledIntentRouter, IntentRouter
from xinyidai_agent.system_tools import SessionMemorySystemTool
from xinyidai_agent.tools.registry import ToolRegistry, default_tool_registry


class ControlledAgentLoop:
    """单 Agent 受控循环。

    模型可以生成回答，但业务动作由路由、风险策略和工具结果共同约束。
    当前版本先用规则路由和 mock 工具跑通循环骨架，后续再替换为模型意图识别和真实工具。
    """

    def __init__(
        self,
        model: ChatModel,
        router: IntentRouter | None = None,
        tool_registry: ToolRegistry | None = None,
        memory_manager: MemoryManager | None = None,
        max_steps: int = 3,
    ) -> None:
        self._model = model
        self._router = router or ControlledIntentRouter(model=model)
        self._tool_registry = tool_registry or default_tool_registry()
        self._memory = memory_manager or MemoryManager()
        self._session_memory_tool = SessionMemorySystemTool()
        self._max_steps = max_steps

    def run(self, request: ChatRequest) -> Iterator[AgentEvent]:
        context = _RunContext(turn_id=str(uuid4()))
        session_state = self._memory.load(request.session_id)
        yield context.event("turn_started", "diagnostic", {"user_message": request.user_message})
        yield context.event("session_loaded", "diagnostic", session_state.model_dump())
        memory_execution = self._session_memory_tool.plan_and_execute(
            self._model,
            request,
            session_state,
        )
        if memory_execution.status != "noop":
            yield context.event(
                "system_tool_started",
                "diagnostic",
                {
                    "tool_name": memory_execution.tool_name,
                    "arguments": memory_execution.arguments,
                    "visibility": self._session_memory_tool.visibility,
                },
            )
            yield context.event(
                "system_tool_result",
                "diagnostic",
                {
                    "tool_name": memory_execution.tool_name,
                    "status": memory_execution.status,
                    "applied_operations": memory_execution.applied_operations,
                    "rejected_operations": memory_execution.rejected_operations,
                    "error": memory_execution.error,
                },
            )
        if memory_execution.status == "success":
            session_state = memory_execution.state
            self._memory.save(session_state)
            yield context.event("session_updated", "diagnostic", session_state.model_dump())

        request = self._memory.enrich_request(request, session_state)
        yield context.transition("LISTENING", "ROUTING", "开始识别业务意图")
        yield context.event("route_started", "diagnostic", {"strategy": "rule_first_model_then_policy"})

        route = self._router.route(request)
        yield context.event("route_decision", "diagnostic", route.model_dump())

        if route.missing_slots:
            yield context.transition("ROUTING", "CLARIFYING", "缺少必要业务槽位")
            question = self._build_clarifying_question(route)
            yield from context.user_answer_events(question)
            yield context.transition("CLARIFYING", "FINISHED", "等待用户补充信息")
            session_state = self._memory.update(
                session_state,
                request,
                route=route,
                stop_reason="missing_slots",
            )
            yield context.event("session_updated", "diagnostic", session_state.model_dump())
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "missing_slots"})
            return

        for step in range(1, self._max_steps + 1):
            yield context.transition("ROUTING", "PLANNING_TOOL", f"第 {step} 步规划工具动作")
            tool_call = self._plan_tool(request, route)

            if tool_call is None:
                yield context.transition("PLANNING_TOOL", "ANSWERING", "无需工具，直接回答")
                answer = self._answer_without_tool(request, route)
                yield from context.user_answer_events(answer)
                yield context.transition("ANSWERING", "FINISHED", "已生成回答")
                session_state = self._memory.update(
                    session_state,
                    request,
                    route=route,
                    stop_reason="answered_without_tool",
                )
                yield context.event("session_updated", "diagnostic", session_state.model_dump())
                yield context.event("turn_finished", "diagnostic", {"stop_reason": "answered_without_tool"})
                return

            yield context.event("tool_call_proposed", "diagnostic", tool_call.model_dump())

            pending_action = self._build_pending_action(tool_call)
            if pending_action is not None:
                yield context.transition("PLANNING_TOOL", "WAITING_CONFIRMATION", "工具动作需要用户确认")
                yield context.event(
                    "confirmation_required",
                    "user",
                    {"pending_action": pending_action.model_dump()},
                )
                session_state = self._memory.update(
                    session_state,
                    request,
                    route=route,
                    pending_action=pending_action,
                    stop_reason="waiting_confirmation",
                )
                yield context.event("session_updated", "diagnostic", session_state.model_dump())
                yield context.event("turn_finished", "diagnostic", {"stop_reason": "waiting_confirmation"})
                return

            yield context.transition("PLANNING_TOOL", "EXECUTING_TOOL", "工具动作通过风险检查")
            yield context.event("tool_started", "diagnostic", {"tool_call": tool_call.model_dump()})
            tool_result, sources, retrieval_trace = self._execute_tool(request, route, tool_call)
            yield context.event("tool_result", "diagnostic", tool_result.model_dump())
            yield context.transition("EXECUTING_TOOL", "OBSERVING_RESULT", "工具结果已返回")

            if tool_result.status != "success":
                yield context.transition("OBSERVING_RESULT", "ANSWERING", "工具失败，生成可理解提示")
                yield from context.user_answer_events(tool_result.user_visible_message or "工具调用失败，请稍后重试。")
                yield context.transition("ANSWERING", "FINISHED", "失败提示已返回")
                session_state = self._memory.update(
                    session_state,
                    request,
                    route=route,
                    tool_result=tool_result,
                    stop_reason="tool_failed",
                )
                yield context.event("session_updated", "diagnostic", session_state.model_dump())
                yield context.event("turn_finished", "diagnostic", {"stop_reason": "tool_failed"})
                return

            if tool_result.terminal:
                yield context.transition("OBSERVING_RESULT", "ANSWERING", "工具结果满足本轮收敛条件")
                answer = self._answer_from_terminal_tool(request, route, tool_result, sources)
                yield from context.user_answer_events(answer)
                yield context.transition("ANSWERING", "FINISHED", "已基于工具结果生成回答")
                session_state = self._memory.update(
                    session_state,
                    request,
                    route=route,
                    tool_result=tool_result,
                    stop_reason="completed",
                )
                yield context.event("session_updated", "diagnostic", session_state.model_dump())
                yield context.event(
                    "turn_finished",
                    "diagnostic",
                    {
                        "stop_reason": "completed",
                        "retrieval_trace": retrieval_trace.model_dump() if retrieval_trace else None,
                    },
                )
                return

            yield context.transition("OBSERVING_RESULT", "ROUTING", "工具结果要求继续下一步")

        yield context.transition("ROUTING", "REJECTED", "达到最大循环步数")
        yield from context.user_answer_events("本轮未能在安全步数内完成，请补充信息或稍后重试。")
        session_state = self._memory.update(
            session_state,
            request,
            route=route,
            stop_reason="max_steps",
        )
        yield context.event("session_updated", "diagnostic", session_state.model_dump())
        yield context.event("turn_finished", "diagnostic", {"stop_reason": "max_steps"})

    def answer(self, request: ChatRequest) -> ChatResponse:
        started_at = perf_counter()
        events = list(self.run(request))
        duration_ms = (perf_counter() - started_at) * 1000
        final_answer = self._extract_final_answer(events)
        route_decision = self._extract_route(events)
        tool_trace = self._extract_tool_trace(events)
        pending_action = self._extract_pending_action(events)
        session_state = self._extract_session_state(events)
        retrieval_trace = self._extract_retrieval_trace(events)
        sources = self._extract_sources(tool_trace)
        stop_reason = self._extract_stop_reason(events)
        business_status = self._extract_business_status(tool_trace, stop_reason)
        evidence = self._build_evidence(route_decision, tool_trace, sources, retrieval_trace)
        model_decision = self._build_model_decision(route_decision, stop_reason, evidence, pending_action)
        actions = self._extract_actions(tool_trace)
        task_stage = self._task_stage(stop_reason)

        return ChatResponse(
            answer=final_answer,
            stop_reason=stop_reason,
            business_status=business_status,
            task_stage=task_stage,
            sources=sources,
            route_decision=route_decision,
            retrieval_trace=retrieval_trace,
            tool_trace=tool_trace,
            pending_action=pending_action,
            session_state=session_state,
            evidence=evidence,
            model_decision=model_decision,
            performance=PerformanceSummary(
                total_ms=duration_ms,
                stages=[
                    PerformanceStage(
                        name="agent_turn",
                        duration_ms=duration_ms,
                        percentage=100.0,
                    )
                ],
            ),
            actions=actions,
            events=events,
            diagnostics=[
                DiagnosticEvent(
                    name="controlled_loop",
                    detail={
                        "mode": "single_agent_controlled_tool_loop",
                        "max_steps": self._max_steps,
                        "event_count": len(events),
                        "session_id": session_state.session_id if session_state else None,
                        "tool_calls": len(tool_trace),
                        "stop_reason": stop_reason,
                        "business_status": business_status,
                    },
                )
            ],
        )

    def _plan_tool(self, request: ChatRequest, route: RouteDecision) -> ToolCall | None:
        if not route.should_call_tool:
            return None

        if route.scene == "DATA_QUERY":
            return ToolCall(
                tool_call_id=str(uuid4()),
                tool_name="query_credit_amount",
                tool_category="data_query",
                arguments={
                    "company_name": route.filled_slots.get("company_name"),
                    "query": request.user_message,
                },
                risk_level="read_only",
                reason="授信额度属于确定性数值，必须调用只读数据工具。",
            )

        if route.scene == "LOAN_APPLY":
            return ToolCall(
                tool_call_id=str(uuid4()),
                tool_name="create_application",
                tool_category="application",
                arguments={
                    "company_name": route.filled_slots.get("company_name"),
                    "product_name": route.filled_slots.get("product_name"),
                },
                risk_level="state_create",
                confirmation_required=True,
                reason="创建贷款申请会产生业务状态，执行前必须用户确认。",
            )

        if route.scene == "KNOWLEDGE_QA":
            return ToolCall(
                tool_call_id=str(uuid4()),
                tool_name="rag_search",
                tool_category="knowledge",
                arguments={"query": request.user_message, "top_k": request.top_k},
                risk_level="read_only",
                reason="知识问答需要检索证据后再生成回答。",
            )

        return None

    def _build_pending_action(self, tool_call: ToolCall) -> PendingAction | None:
        if not tool_call.confirmation_required:
            return None

        return PendingAction(
            action_id=str(uuid4()),
            tool_call=tool_call,
            title="确认创建贷款申请",
            summary="该操作将创建贷款申请草稿，后续需要企业授权后才能提交。",
            confirm_label="确认创建",
            cancel_label="取消",
            details=[
                {"label": "企业", "value": str(tool_call.arguments.get("company_name", ""))},
                {"label": "产品", "value": str(tool_call.arguments.get("product_name", ""))},
                {"label": "动作", "value": "创建申请草稿"},
            ],
        )

    def _execute_tool(
        self,
        request: ChatRequest,
        route: RouteDecision,
        tool_call: ToolCall,
    ) -> tuple[ToolResult, list[SourceDocument], RetrievalTrace | None]:
        execution = self._tool_registry.execute(request, route, tool_call)
        return execution.result, execution.sources, execution.retrieval_trace

    def _summarize_tool_result(
        self,
        request: ChatRequest,
        route: RouteDecision,
        tool_result: ToolResult,
        sources: list[SourceDocument],
    ) -> str:
        messages = self._build_messages(request.user_message, route, tool_result, sources)
        return self._model.complete(messages)

    def _answer_from_terminal_tool(
        self,
        request: ChatRequest,
        route: RouteDecision,
        tool_result: ToolResult,
        sources: list[SourceDocument],
    ) -> str:
        if tool_result.user_visible_message and tool_result.business_status in {"PARTIAL_DATA", "NOT_FOUND"}:
            return tool_result.user_visible_message
        return self._summarize_tool_result(request, route, tool_result, sources)

    def _answer_without_tool(self, request: ChatRequest, route: RouteDecision) -> str:
        messages = [
            {
                "role": "system",
                "content": "你是信易贷聊天助手。仅回答用户当前问题，不发起工具调用。",
            },
            {
                "role": "user",
                "content": f"用户问题：{request.user_message}\n\n路由结果：{route.model_dump()}",
            },
        ]
        return self._model.complete(messages)

    def _build_messages(
        self,
        user_message: str,
        route: RouteDecision,
        tool_result: ToolResult,
        sources: list[SourceDocument],
    ) -> list[dict[str, str]]:
        source_text = self._format_sources(sources)
        return [
            {
                "role": "system",
                "content": (
                    "你是信易贷聊天助手。你只能基于路由结果、工具结果和证据回答；"
                    "数值类结果必须说明来自工具，证据不足时要说明缺口；"
                    "不要伪造申请状态、授权状态或贷款审批结论。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{user_message}\n\n"
                    f"路由结果：{route.model_dump()}\n\n"
                    f"工具结果：{tool_result.model_dump()}\n\n"
                    f"可用证据：\n{source_text}"
                ),
            },
        ]

    def _build_clarifying_question(self, route: RouteDecision) -> str:
        if "user_intent" in route.missing_slots:
            return "我还没判断出您要办理哪类事项。您是想咨询政策、查询授信额度，还是发起贷款申请？"

        labels = {
            "company_name": "企业名称",
            "product_name": "贷款产品",
            "application_id": "申请编号",
        }
        readable_slots = [labels.get(slot, slot) for slot in route.missing_slots]
        missing = "、".join(readable_slots)
        return f"为了继续处理这个请求，请先补充：{missing}。"

    def _format_sources(self, sources: list[SourceDocument]) -> str:
        if not sources:
            return "暂无检索证据。"

        return "\n\n".join(
            f"[{index}] {source.title}\n{source.content}"
            for index, source in enumerate(sources, start=1)
        )

    def _extract_final_answer(self, events: list[AgentEvent]) -> str:
        for event in reversed(events):
            if event.event_type == "final_answer":
                return str(event.payload.get("answer", ""))
        for event in reversed(events):
            if event.event_type == "assistant_delta":
                return str(event.payload.get("answer", ""))
        return ""

    def _extract_route(self, events: list[AgentEvent]) -> RouteDecision | None:
        for event in events:
            if event.event_type == "route_decision":
                return RouteDecision.model_validate(event.payload)
        return None

    def _extract_tool_trace(self, events: list[AgentEvent]) -> list[ToolResult]:
        return [
            ToolResult.model_validate(event.payload)
            for event in events
            if event.event_type == "tool_result"
        ]

    def _extract_pending_action(self, events: list[AgentEvent]) -> PendingAction | None:
        for event in events:
            if event.event_type == "confirmation_required":
                value = event.payload.get("pending_action")
                if value:
                    return PendingAction.model_validate(value)
        return None

    def _extract_session_state(self, events: list[AgentEvent]) -> SessionStateSnapshot | None:
        for event in reversed(events):
            if event.event_type in {"session_updated", "session_loaded"}:
                return SessionStateSnapshot.model_validate(event.payload)
        return None

    def _extract_retrieval_trace(self, events: list[AgentEvent]) -> RetrievalTrace | None:
        for event in reversed(events):
            if event.event_type == "tool_result":
                output = event.payload.get("output")
                if isinstance(output, dict) and output.get("retrieval_trace"):
                    return RetrievalTrace.model_validate(output["retrieval_trace"])
        return None

    def _extract_sources(self, tool_trace: list[ToolResult]) -> list[SourceDocument]:
        sources: list[SourceDocument] = []
        for result in tool_trace:
            raw_sources = result.output.get("sources")
            if isinstance(raw_sources, list):
                sources.extend(SourceDocument.model_validate(source) for source in raw_sources)
        return sources

    def _extract_stop_reason(self, events: list[AgentEvent]) -> str | None:
        for event in reversed(events):
            if event.event_type == "turn_finished":
                value = event.payload.get("stop_reason")
                return str(value) if value else None
        return None

    def _extract_business_status(self, tool_trace: list[ToolResult], stop_reason: str | None) -> str:
        for result in reversed(tool_trace):
            if result.business_status:
                return result.business_status
            if result.envelope:
                return result.envelope.status
        if stop_reason == "waiting_confirmation":
            return "AUTH_REQUIRED"
        if stop_reason in {"tool_failed", "max_steps"}:
            return "ERROR"
        return "NO_TOOL_USED"

    def _build_evidence(
        self,
        route: RouteDecision | None,
        tool_trace: list[ToolResult],
        sources: list[SourceDocument],
        retrieval_trace: RetrievalTrace | None,
    ) -> EvidenceState:
        if route is not None and route.missing_slots:
            return EvidenceState(
                ready=False,
                reason="missing_slots",
                missing=route.missing_slots,
            )
        if not tool_trace:
            return EvidenceState(
                ready=False,
                reason="no_tool_result",
                missing=[],
            )
        latest = tool_trace[-1]
        if latest.status != "success":
            return EvidenceState(
                ready=False,
                reason=latest.error_message or latest.status,
                signals={"tool_status": latest.status, "business_status": latest.business_status},
            )
        return EvidenceState(
            ready=True,
            reason="tool_result_ready",
            signals={
                "tool_name": latest.tool_name,
                "business_status": latest.business_status,
                "source_count": len(sources),
                "retrieval_results": retrieval_trace.results_count if retrieval_trace else None,
            },
        )

    def _build_model_decision(
        self,
        route: RouteDecision | None,
        stop_reason: str | None,
        evidence: EvidenceState,
        pending_action: PendingAction | None,
    ) -> ModelDecision:
        if pending_action is not None:
            return ModelDecision(
                should_finish=False,
                next_action="ask_user",
                confidence=route.confidence if route else 0.0,
                reason="高风险工具动作等待用户确认。",
                ask_user_question=pending_action.summary,
                source="runtime",
                task_stage="decide",
            )
        if route is not None and route.missing_slots:
            return ModelDecision(
                should_finish=False,
                next_action="ask_user",
                confidence=route.confidence,
                reason="缺少必要业务槽位。",
                missing_info=route.missing_slots,
                ask_user_question=self._build_clarifying_question(route),
                source="policy",
                task_stage="understand",
            )
        return ModelDecision(
            should_finish=stop_reason in {"completed", "answered_without_tool"},
            next_action="finish" if stop_reason in {"completed", "answered_without_tool"} else "reject",
            confidence=route.confidence if route else 0.0,
            reason=evidence.reason,
            source="runtime",
            task_stage=self._task_stage(stop_reason),
        )

    def _extract_actions(self, tool_trace: list[ToolResult]):
        actions = []
        for result in tool_trace:
            actions.extend(result.actions)
            if result.envelope:
                actions.extend(result.envelope.actions)
        return actions

    def _task_stage(self, stop_reason: str | None) -> str:
        if stop_reason == "missing_slots":
            return "understand"
        if stop_reason == "waiting_confirmation":
            return "decide"
        if stop_reason in {"completed", "answered_without_tool"}:
            return "done"
        return "gather_evidence"


class _RunContext:
    def __init__(self, turn_id: str) -> None:
        self.turn_id = turn_id
        self.sequence = 0

    def event(
        self,
        event_type: AgentEventType,
        visibility: EventVisibility,
        payload: dict[str, object],
    ) -> AgentEvent:
        self.sequence += 1
        return AgentEvent(
            turn_id=self.turn_id,
            sequence=self.sequence,
            event_type=event_type,
            visibility=visibility,
            timestamp=datetime.now(UTC).isoformat(),
            payload=payload,
        )

    def transition(self, from_state: AgentState, to_state: AgentState, reason: str) -> AgentEvent:
        transition = StateTransition(from_state=from_state, to_state=to_state, reason=reason)
        return self.event("state_changed", "diagnostic", transition.model_dump())

    def user_answer_events(self, answer: str) -> Iterator[AgentEvent]:
        chunks = [answer[index : index + 12] for index in range(0, len(answer), 12)] or [""]
        current = ""
        for chunk in chunks:
            current += chunk
            yield self.event("assistant_delta", "user", {"text": chunk, "answer": current})
        yield self.event("final_answer", "user", {"answer": answer})
