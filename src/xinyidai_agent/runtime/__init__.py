"""ReAct 运行时公开入口。"""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

from xinyidai_agent.capabilities.catalog import CapabilityCatalog, default_capability_catalog
from xinyidai_agent.evidence_policy import EvidencePolicy
from xinyidai_agent.llm import ChatModel
from xinyidai_agent.memory import ConversationStore, MemoryManager
from xinyidai_agent.protocol import (
    AgentAction,
    AgentEvent,
    ChatRequest,
    ChatResponse,
    ConfirmActionRequest,
    DiagnosticEvent,
    EvidenceState,
    ModelDecision,
    PendingAction,
    PerformanceStage,
    PerformanceSummary,
    RetrievalTrace,
    RouteDecision,
    SessionStateSnapshot,
    SourceDocument,
    ToolResult,
)
from xinyidai_agent.router import ControlledIntentRouter
from xinyidai_agent.router.rules_guard import RulesGuard
from xinyidai_agent.router.scene_direct import SceneDirectDispatcher
from xinyidai_agent.runtime.fast_path import FastPathRunner
from xinyidai_agent.runtime.react_engine import ReactStepEngine
from xinyidai_agent.runtime.react_loop import ReactRuntime, measure_events
from xinyidai_agent.runtime.react_observation_builder import ReactObservationBuilder
from xinyidai_agent.tools.registry import ToolRegistry, default_tool_registry


class ControlledAgentLoop:
    """兼容既有 API 的受控 Agent 入口，内部委托给 ReActRuntime。"""

    def __init__(
        self,
        model: ChatModel,
        router: object | None = None,
        tool_registry: ToolRegistry | None = None,
        memory_manager: MemoryManager | None = None,
        conversation_store: ConversationStore | None = None,
        capability_catalog: CapabilityCatalog | None = None,
        react_runtime: ReactRuntime | None = None,
        max_steps: int = 3,
        **_unused: object,
    ) -> None:
        """初始化对外兼容层和生产运行时依赖。"""
        self._model = model
        self._tool_registry = tool_registry or default_tool_registry()
        self._memory = memory_manager or MemoryManager()
        self._conversation_store = conversation_store
        self._max_steps = max_steps
        self._capability_catalog = capability_catalog or CapabilityCatalog(default_capability_catalog())
        if router is None:
            router = ControlledIntentRouter(
                model=model,
                guard=RulesGuard(),
                scene_direct=SceneDirectDispatcher(self._capability_catalog),
            )
        self._router = router
        self._react_runtime = react_runtime or ReactRuntime(
            model=model,
            router=self._router,
            tool_registry=self._tool_registry,
            memory_manager=self._memory,
            capability_catalog=self._capability_catalog,
            fast_path=FastPathRunner(self._tool_registry, model),
            react_engine=ReactStepEngine(model, self._tool_registry),
            observation_builder=ReactObservationBuilder(self._tool_registry),
            evidence_policy=EvidencePolicy(),
        )

    @property
    def conversation_store(self) -> ConversationStore | None:
        """返回当前 Agent 使用的会话持久化存储。"""
        return self._conversation_store

    def run(self, request: ChatRequest) -> Iterator[AgentEvent]:
        """执行一轮普通聊天请求。"""
        yield from self._react_runtime.run(request)

    def run_confirmation(self, request: ConfirmActionRequest) -> Iterator[AgentEvent]:
        """执行确认或取消请求。"""
        yield from self._react_runtime.run_confirmation(request)

    def answer(self, request: ChatRequest) -> ChatResponse:
        """执行一轮聊天并返回结构化响应。"""
        events, duration_ms = measure_events(self.run(request))
        return self._build_chat_response(events, duration_ms)

    def confirm(self, request: ConfirmActionRequest) -> ChatResponse:
        """确认或取消上一轮生成的待确认动作。"""
        events, duration_ms = measure_events(self.run_confirmation(request))
        return self._build_chat_response(events, duration_ms)

    def record_events(self, events: list[AgentEvent]) -> None:
        """把运行事件写入 transcript 和可选会话记录。"""
        state = self._extract_session_state(events)
        if state is None:
            return
        for event in events:
            self._memory.append_event(state.session_id, event)
        if events:
            self._memory.append_turn_summary(
                state.session_id,
                events[0].turn_id,
                {
                    "stop_reason": self._extract_stop_reason(events),
                    "event_count": len(events),
                    "route": self._extract_route(events).model_dump() if self._extract_route(events) else None,
                },
            )
        self._record_conversation_events(state.session_id, events)

    def _build_chat_response(self, events: list[AgentEvent], duration_ms: float) -> ChatResponse:
        """从事件流组装旧接口使用的 ChatResponse。"""
        self.record_events(events)
        final_answer = self._extract_final_answer(events)
        route = self._extract_route(events)
        tool_trace = self._extract_tool_trace(events)
        pending_action = self._extract_pending_action(events)
        session_state = self._extract_session_state(events)
        retrieval_trace = self._extract_retrieval_trace(events)
        sources = self._extract_sources(tool_trace)
        stop_reason = self._extract_stop_reason(events)
        business_status = self._extract_business_status(tool_trace, stop_reason)
        evidence = self._build_evidence(route, tool_trace, sources, retrieval_trace)
        model_decision = self._build_model_decision(route, stop_reason, evidence, pending_action)
        actions = self._extract_actions(tool_trace)

        return ChatResponse(
            answer=final_answer,
            stop_reason=stop_reason,
            business_status=business_status,
            task_stage=self._task_stage(stop_reason),
            sources=sources,
            route_decision=route,
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

    def _record_conversation_events(self, session_id: str, events: list[AgentEvent]) -> None:
        """把完整回合事件写入可选 PG 会话记录。"""
        if self._conversation_store is None or not events:
            return
        self._conversation_store.ensure_session(
            session_id,
            metadata={"entrypoint": "chat_window", "turn_id": events[0].turn_id},
        )
        user_message = _extract_user_message(events)
        if user_message:
            self._conversation_store.append_message(
                session_id=session_id,
                role="user",
                content=user_message,
                payload={"event": events[0].model_dump(mode="json")},
                event_type=events[0].event_type,
                turn_id=events[0].turn_id,
            )
        for event in events:
            self._conversation_store.append_audit_event(session_id=session_id, event=event)
        final_answer = self._extract_final_answer(events)
        if final_answer:
            final_event = next(
                (event for event in reversed(events) if event.event_type == "final_answer"),
                None,
            )
            self._conversation_store.append_message(
                session_id=session_id,
                role="assistant",
                content=final_answer,
                payload={"event": final_event.model_dump(mode="json")} if final_event else {},
                event_type=final_event.event_type if final_event else None,
                turn_id=final_event.turn_id if final_event else events[0].turn_id,
            )

    @staticmethod
    def _extract_final_answer(events: list[AgentEvent]) -> str:
        """提取最终答案。"""
        for event in reversed(events):
            if event.event_type == "final_answer":
                return str(event.payload.get("answer", ""))
        for event in reversed(events):
            if event.event_type == "assistant_delta":
                return str(event.payload.get("answer", ""))
        return ""

    @staticmethod
    def _extract_route(events: list[AgentEvent]) -> RouteDecision | None:
        """提取路由决策。"""
        for event in events:
            if event.event_type == "route_decision":
                return RouteDecision.model_validate(event.payload)
        return None

    @staticmethod
    def _extract_tool_trace(events: list[AgentEvent]) -> list[ToolResult]:
        """提取工具执行结果。"""
        return [
            ToolResult.model_validate(event.payload)
            for event in events
            if event.event_type == "tool_result"
        ]

    @staticmethod
    def _extract_pending_action(events: list[AgentEvent]) -> PendingAction | None:
        """提取待确认动作。"""
        for event in events:
            if event.event_type == "confirmation_required":
                value = event.payload.get("pending_action")
                if value:
                    return PendingAction.model_validate(value)
        return None

    @staticmethod
    def _extract_session_state(events: list[AgentEvent]) -> SessionStateSnapshot | None:
        """提取最新会话状态。"""
        for event in reversed(events):
            if event.event_type in {"session_updated", "session_loaded"}:
                return SessionStateSnapshot.model_validate(event.payload)
        return None

    @staticmethod
    def _extract_retrieval_trace(events: list[AgentEvent]) -> RetrievalTrace | None:
        """提取检索 trace。"""
        for event in reversed(events):
            if event.event_type == "tool_result":
                output = event.payload.get("output")
                if isinstance(output, dict) and output.get("retrieval_trace"):
                    return RetrievalTrace.model_validate(output["retrieval_trace"])
        return None

    @staticmethod
    def _extract_sources(tool_trace: list[ToolResult]) -> list[SourceDocument]:
        """从工具输出提取证据源。"""
        sources: list[SourceDocument] = []
        for result in tool_trace:
            raw_sources = result.output.get("sources")
            if isinstance(raw_sources, list):
                sources.extend(SourceDocument.model_validate(source) for source in raw_sources)
        return sources

    @staticmethod
    def _extract_stop_reason(events: list[AgentEvent]) -> str | None:
        """提取停止原因。"""
        for event in reversed(events):
            if event.event_type == "turn_finished":
                value = event.payload.get("stop_reason")
                return str(value) if value else None
        return None

    @staticmethod
    def _extract_business_status(tool_trace: list[ToolResult], stop_reason: str | None) -> str:
        """根据最后工具结果或停止原因确定业务状态。"""
        for result in reversed(tool_trace):
            if result.business_status:
                return result.business_status
            if result.envelope:
                return result.envelope.status
        if stop_reason == "waiting_confirmation":
            return "AUTH_REQUIRED"
        if stop_reason in {"tool_failed", "max_steps", "runtime_error"}:
            return "ERROR"
        return "NO_TOOL_USED"

    @staticmethod
    def _build_evidence(
        route: RouteDecision | None,
        tool_trace: list[ToolResult],
        sources: list[SourceDocument],
        retrieval_trace: RetrievalTrace | None,
    ) -> EvidenceState:
        """组装证据状态。"""
        if route is not None and route.missing_slots:
            return EvidenceState(ready=False, reason="missing_slots", missing=route.missing_slots)
        if not tool_trace:
            return EvidenceState(ready=False, reason="no_tool_result", missing=[])
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
        """组装前端可读的模型决策摘要。"""
        if pending_action is not None:
            return ModelDecision(
                should_finish=False,
                next_action="ask_user",
                confidence=route.confidence if route else 0.0,
                reason="高风险工具动作等待用户确认。",
                ask_user_question=pending_action.summary,
                source="react_runtime",
                task_stage="decide",
            )
        if route is not None and route.missing_slots:
            reason = route.route_failure.user_reason if route.route_failure else "缺少必要业务槽位。"
            return ModelDecision(
                should_finish=False,
                next_action="ask_user",
                confidence=route.confidence,
                reason=reason,
                missing_info=route.missing_slots,
                ask_user_question=_clarifying_question(route),
                source="policy",
                task_stage="understand",
            )
        finished = stop_reason in {"completed", "answered_without_tool"}
        return ModelDecision(
            should_finish=finished,
            next_action="finish" if finished else "reject",
            confidence=route.confidence if route else 0.0,
            reason=evidence.reason,
            source="react_runtime",
            task_stage=self._task_stage(stop_reason),
        )

    @staticmethod
    def _extract_actions(tool_trace: list[ToolResult]) -> list[AgentAction]:
        """提取工具返回的前端动作。"""
        actions: list[AgentAction] = []
        for result in tool_trace:
            actions.extend(result.actions)
            if result.envelope:
                actions.extend(result.envelope.actions)
        return actions

    @staticmethod
    def _task_stage(stop_reason: str | None) -> str:
        """把停止原因映射为粗粒度任务阶段。"""
        if stop_reason == "missing_slots":
            return "understand"
        if stop_reason == "waiting_confirmation":
            return "decide"
        if stop_reason in {"completed", "answered_without_tool"}:
            return "done"
        return "gather_evidence"


def _extract_user_message(events: list[AgentEvent]) -> str:
    """从事件流提取用户输入，用于会话落库。"""
    if not events:
        return ""
    payload = events[0].payload
    if payload.get("user_message"):
        return str(payload["user_message"])
    confirmation = payload.get("confirmation")
    if isinstance(confirmation, dict):
        action_id = confirmation.get("action_id", "")
        action = "确认" if confirmation.get("confirmed") else "取消"
        return f"{action}待确认动作：{action_id}"
    return ""


def _clarifying_question(route: RouteDecision) -> str:
    """生成澄清问题，优先使用结构化路由失败给出的用户可读原因。"""
    if route.route_failure is not None:
        questions = " ".join(route.route_failure.suggested_questions)
        if questions:
            return f"{route.route_failure.user_reason} {questions}"
        return route.route_failure.user_reason
    labels = {
        "company_name": "企业名称",
        "product_name": "贷款产品",
        "application_id": "申请编号",
        "user_intent": "办理事项",
    }
    readable = [labels.get(slot, slot) for slot in route.missing_slots]
    return f"为了继续处理这个请求，请先补充：{'、'.join(readable)}。"


def _request_id(request: ChatRequest) -> str | None:
    """从请求 metadata 中读取审计请求 ID。"""
    value = request.metadata.get("request_id")
    return str(value) if value else None


def _new_request_id() -> str:
    """生成请求 ID，预留给外部持久化扩展。"""
    return str(uuid4())


__all__ = ["ControlledAgentLoop", "ReactRuntime"]
