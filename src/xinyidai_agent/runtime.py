from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from xinyidai_agent.llm import JSON_OBJECT_RESPONSE_FORMAT, ChatModel
from xinyidai_agent.memory import ConversationStore, MemoryManager
from xinyidai_agent.evidence_policy import EvidencePolicy
from xinyidai_agent.policies import RiskPolicy
from xinyidai_agent.protocol import (
    AgentEvent,
    AgentEventType,
    AgentState,
    ChatRequest,
    ChatResponse,
    ConfirmActionRequest,
    DiagnosticEvent,
    EvidenceState,
    EventVisibility,
    ModelDecision,
    PendingAction,
    PerformanceSummary,
    PerformanceStage,
    RetrievalTrace,
    RouteDecision,
    RuntimeStepDecision,
    SessionStateSnapshot,
    SourceDocument,
    StateTransition,
    ToolCall,
    ToolResult,
)
from xinyidai_agent.router import ControlledIntentRouter, IntentRouter, RoutePolicy
from xinyidai_agent.pending_actions import PendingActionBuilder
from xinyidai_agent.runtime_policy import RuntimeStepContext, RuntimeStepPolicy
from xinyidai_agent.runtime_state import RuntimeStateReducer
from xinyidai_agent.skills import SkillRegistry
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
        conversation_store: ConversationStore | None = None,
        skill_registry: SkillRegistry | None = None,
        max_steps: int = 3,
    ) -> None:
        """初始化受控 Agent 循环及其可选的会话持久化组件。"""
        self._model = model
        self._router = router or ControlledIntentRouter(model=model)
        self._tool_registry = tool_registry or default_tool_registry()
        self._memory = memory_manager or MemoryManager()
        self._conversation_store = conversation_store
        self._skills = skill_registry or SkillRegistry()
        self._session_memory_tool = SessionMemorySystemTool()
        self._action_policy = RoutePolicy()
        self._state_reducer = RuntimeStateReducer()
        self._pending_action_builder = PendingActionBuilder()
        self._step_policy = RuntimeStepPolicy()
        self._evidence_policy = EvidencePolicy()
        self._risk_policy = RiskPolicy()
        self._max_steps = max_steps

    @property
    def conversation_store(self) -> ConversationStore | None:
        """返回当前 Agent 使用的会话记录存储。"""
        return self._conversation_store

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
        yield context.event("route_started", "diagnostic", {"strategy": "guard_model_then_policy"})

        route = self._router.route(request)
        yield context.event("route_decision", "diagnostic", route.model_dump())
        session_state = self._state_reducer.apply_route(session_state, route)

        last_tool_result: ToolResult | None = None
        last_sources: list[SourceDocument] = []
        executed_tool_names: list[str] = []
        last_retrieval_trace: RetrievalTrace | None = None
        for step in range(1, self._max_steps + 1):
            yield context.event(
                "runtime_step_started",
                "diagnostic",
                {"step_index": step, "max_steps": self._max_steps, "stage": session_state.current_stage},
            )
            step_context = RuntimeStepContext(
                step_index=step,
                max_steps=self._max_steps,
                request=request,
                session_state=session_state,
                route=route,
                last_tool_result=last_tool_result,
                last_sources=last_sources,
                pending_action=session_state.pending_action,
                executed_tool_names=executed_tool_names,
            )
            decision = self._step_policy.decide_next_step(step_context)
            yield context.event("runtime_step_decision", "diagnostic", decision.model_dump())

            if decision.step_type == "ASK_USER":
                yield context.transition("ROUTING", "CLARIFYING", decision.reason)
                question = decision.question or self._build_clarifying_question(route)
                yield from context.user_answer_events(question)
                yield context.transition("CLARIFYING", "FINISHED", "等待用户补充信息")
                session_state = self._memory.update(
                    session_state,
                    request,
                    route=route,
                    stop_reason="missing_slots",
                    final_answer=question,
                )
                yield context.event("session_updated", "diagnostic", session_state.model_dump())
                yield context.event("turn_finished", "diagnostic", {"stop_reason": "missing_slots"})
                return

            if decision.step_type == "WAIT_CONFIRMATION":
                answer = "当前已有待确认动作，请先确认或取消后再继续。"
                yield from context.user_answer_events(answer)
                yield context.event("session_updated", "diagnostic", session_state.model_dump())
                yield context.event("turn_finished", "diagnostic", {"stop_reason": "waiting_confirmation"})
                return

            if decision.step_type == "ANSWER_WITHOUT_TOOL":
                yield context.transition("ROUTING", "ANSWERING", "无需工具，直接回答")
                answer = self._answer_without_tool(request, route)
                yield from context.user_answer_events(answer)
                session_state = self._memory.update(
                    session_state,
                    request,
                    route=route,
                    stop_reason="answered_without_tool",
                    final_answer=answer,
                )
                yield context.event("session_updated", "diagnostic", session_state.model_dump())
                yield context.event("turn_finished", "diagnostic", {"stop_reason": "answered_without_tool"})
                return

            if decision.step_type == "GENERATE_FINAL_ANSWER":
                yield context.transition("ROUTING", "ANSWERING", decision.reason)
                if last_tool_result is None:
                    answer = decision.answer or self._answer_without_tool(request, route)
                    stop_reason = "answered_without_tool"
                elif last_tool_result.status != "success":
                    answer = last_tool_result.user_visible_message or "工具调用失败，请稍后重试。"
                    stop_reason = "tool_failed"
                else:
                    answer = self._answer_from_terminal_tool(request, route, last_tool_result, last_sources)
                    stop_reason = "completed"
                yield from context.user_answer_events(answer)
                session_state = self._memory.update(
                    session_state,
                    request,
                    route=route,
                    tool_result=last_tool_result,
                    stop_reason=stop_reason,
                    final_answer=answer,
                )
                yield context.event("session_updated", "diagnostic", session_state.model_dump())
                yield context.event(
                    "turn_finished",
                    "diagnostic",
                    {
                        "stop_reason": stop_reason,
                        "retrieval_trace": last_retrieval_trace.model_dump() if last_retrieval_trace else None,
                    },
                )
                return

            if decision.step_type in {"HANDOFF", "STOP"}:
                answer = decision.handoff_reason or decision.reason or "当前请求需要人工处理或稍后重试。"
                yield context.event(
                    "handoff_required" if decision.step_type == "HANDOFF" else "runtime_step_finished",
                    "diagnostic",
                    {"reason": answer, "step_index": step},
                )
                yield from context.user_answer_events(answer)
                session_state = self._memory.update(
                    session_state,
                    request,
                    route=route,
                    stop_reason="handoff" if decision.step_type == "HANDOFF" else "max_steps",
                    final_answer=answer,
                )
                yield context.event("session_updated", "diagnostic", session_state.model_dump())
                yield context.event(
                    "turn_finished",
                    "diagnostic",
                    {"stop_reason": "handoff" if decision.step_type == "HANDOFF" else "max_steps"},
                )
                return

            yield context.transition("ROUTING", "PLANNING_TOOL", f"第 {step} 步规划工具动作")
            tool_call = decision.tool_call or self._plan_tool(request, route, session_state=session_state)

            if tool_call is None:
                yield context.transition("PLANNING_TOOL", "ANSWERING", "无需工具，直接回答")
                answer = self._answer_without_tool(request, route)
                yield context.event(
                    "runtime_step_decision",
                    "diagnostic",
                    RuntimeStepDecision(
                        step_type="ANSWER_WITHOUT_TOOL",
                        reason="当前路由无需工具或无法形成安全工具动作。",
                        confidence=route.confidence,
                        answer=answer,
                    ).model_dump(),
                )
                yield from context.user_answer_events(answer)
                yield context.transition("ANSWERING", "FINISHED", "已生成回答")
                session_state = self._memory.update(
                    session_state,
                    request,
                    route=route,
                    stop_reason="answered_without_tool",
                    final_answer=answer,
                )
                yield context.event("session_updated", "diagnostic", session_state.model_dump())
                yield context.event("turn_finished", "diagnostic", {"stop_reason": "answered_without_tool"})
                return

            yield context.event("tool_call_proposed", "diagnostic", tool_call.model_dump())
            tool_specs = self._tool_registry.specs_for_categories(route.allowed_tool_categories)
            proposal_check = self._action_policy.validate_tool_action(
                route,
                tool_call,
                tool_specs,
                session_state=session_state,
                allow_unconfirmed=True,
            )
            if not proposal_check.allowed:
                answer = f"当前工具动作未通过安全校验：{proposal_check.reason}"
                yield context.event(
                    "runtime_step_decision",
                    "diagnostic",
                    RuntimeStepDecision(
                        step_type="HANDOFF",
                        reason=proposal_check.reason,
                        confidence=route.confidence,
                        handoff_reason=proposal_check.reason,
                    ).model_dump(),
                )
                yield context.event(
                    "handoff_required",
                    "diagnostic",
                    {"reason": proposal_check.reason, "tool_call": tool_call.model_dump()},
                )
                yield from context.user_answer_events(answer)
                session_state = self._memory.update(
                    session_state,
                    request,
                    route=route,
                    stop_reason="tool_blocked",
                    final_answer=answer,
                )
                yield context.event("session_updated", "diagnostic", session_state.model_dump())
                yield context.event("turn_finished", "diagnostic", {"stop_reason": "tool_blocked"})
                return

            pending_action = (
                self._build_pending_action(request, route, session_state, tool_call)
                if decision.step_type == "PROPOSE_PENDING_ACTION" or tool_call.confirmation_required
                else None
            )
            if pending_action is not None:
                yield context.transition("PLANNING_TOOL", "WAITING_CONFIRMATION", "工具动作需要用户确认")
                yield context.event(
                    "runtime_step_decision",
                    "diagnostic",
                    RuntimeStepDecision(
                        step_type="PROPOSE_PENDING_ACTION",
                        reason="工具动作涉及状态变化或需要用户确认。",
                        confidence=route.confidence,
                        tool_call=tool_call,
                        pending_action=pending_action,
                    ).model_dump(),
                )
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
                    final_answer=pending_action.summary,
                )
                yield context.event("session_updated", "diagnostic", session_state.model_dump())
                yield context.event("turn_finished", "diagnostic", {"stop_reason": "waiting_confirmation"})
                return

            yield context.transition("PLANNING_TOOL", "EXECUTING_TOOL", "工具动作通过风险检查")
            execution_check = self._action_policy.validate_tool_action(
                route,
                tool_call,
                tool_specs,
                session_state=session_state,
            )
            yield context.event(
                "pending_action_validated",
                "diagnostic",
                {
                    "allowed": execution_check.allowed,
                    "reason": execution_check.reason,
                    "tool_name": tool_call.tool_name,
                },
            )
            if not execution_check.allowed:
                answer = f"当前工具动作未通过执行前校验：{execution_check.reason}"
                yield from context.user_answer_events(answer)
                session_state = self._memory.update(
                    session_state,
                    request,
                    route=route,
                    stop_reason="tool_blocked",
                    final_answer=answer,
                )
                yield context.event("session_updated", "diagnostic", session_state.model_dump())
                yield context.event("turn_finished", "diagnostic", {"stop_reason": "tool_blocked"})
                return
            yield context.event(
                "runtime_step_decision",
                "diagnostic",
                RuntimeStepDecision(
                    step_type="EXECUTE_TOOL",
                    reason=execution_check.reason,
                    confidence=route.confidence,
                    tool_call=tool_call,
                ).model_dump(),
            )
            yield context.event("tool_started", "diagnostic", {"tool_call": tool_call.model_dump()})
            tool_result, sources, retrieval_trace = self._execute_tool(request, route, tool_call)
            executed_tool_names.append(tool_call.tool_name)
            yield context.event("tool_result", "diagnostic", tool_result.model_dump())
            session_state = self._state_reducer.apply_tool_result(session_state, route, tool_result)
            last_tool_result = tool_result
            last_sources = sources
            last_retrieval_trace = retrieval_trace
            yield context.transition("EXECUTING_TOOL", "OBSERVING_RESULT", "工具结果已返回")
            yield context.transition("OBSERVING_RESULT", "ROUTING", "工具结果要求继续下一步")
            yield context.event(
                "runtime_step_finished",
                "diagnostic",
                {"step_index": step, "stage": session_state.current_stage, "continue": True},
            )

        yield context.transition("ROUTING", "REJECTED", "达到最大循环步数")
        answer = "本轮未能在安全步数内完成，请补充信息或稍后重试。"
        yield from context.user_answer_events(answer)
        session_state = self._memory.update(
            session_state,
            request,
            route=route,
            stop_reason="max_steps",
            final_answer=answer,
        )
        yield context.event("session_updated", "diagnostic", session_state.model_dump())
        yield context.event("turn_finished", "diagnostic", {"stop_reason": "max_steps"})

    def stream(self, request: ChatRequest) -> Iterator[AgentEvent]:
        """执行一轮对话，并把用户消息、运行事件和助手答案同步落库。"""
        request = self._prepare_persisted_request(request)
        if self._conversation_store is None:
            yield from self.run(request)
            return

        self._record_user_message(request)
        try:
            for event in self.run(request):
                self._record_runtime_event(request, event)
                yield event
        except Exception as exc:
            self._record_runtime_error(request, exc)
            raise

    def answer(self, request: ChatRequest) -> ChatResponse:
        """执行一轮非流式对话并返回完整响应。"""
        started_at = perf_counter()
        events = list(self.run(request))
        duration_ms = (perf_counter() - started_at) * 1000
        return self._build_chat_response(events, duration_ms)

    def confirm(self, request: ConfirmActionRequest) -> ChatResponse:
        started_at = perf_counter()
        events = list(self.run_confirmation(request))
        duration_ms = (perf_counter() - started_at) * 1000
        return self._build_chat_response(events, duration_ms)

    def _build_chat_response(self, events: list[AgentEvent], duration_ms: float) -> ChatResponse:
        self.record_events(events)
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

    def _prepare_persisted_request(self, request: ChatRequest) -> ChatRequest:
        """为需要持久化的请求补齐会话 ID 和请求 ID。"""
        if self._conversation_store is None:
            return request
        metadata = dict(request.metadata)
        metadata.setdefault("request_id", str(uuid4()))
        return request.model_copy(
            update={
                "session_id": request.session_id or str(uuid4()),
                "metadata": metadata,
            }
        )

    def _record_user_message(self, request: ChatRequest) -> None:
        """在运行 Agent 前先落库用户原始消息。"""
        if self._conversation_store is None or request.session_id is None:
            return
        request_id = _request_id(request)
        self._conversation_store.ensure_session(
            request.session_id,
            metadata={
                "entrypoint": "chat_window",
                "request_id": request_id,
            },
        )
        self._conversation_store.append_message(
            session_id=request.session_id,
            role="user",
            content=request.user_message,
            payload={
                "top_k": request.top_k,
                "metadata": request.metadata,
            },
            request_id=request_id,
        )

    def _record_runtime_event(self, request: ChatRequest, event: AgentEvent) -> None:
        """把运行事件写入审计表，并把最终答案写入聊天记录表。"""
        if self._conversation_store is None or request.session_id is None:
            return
        request_id = _request_id(request)
        self._conversation_store.append_audit_event(
            session_id=request.session_id,
            event=event,
            request_id=request_id,
        )
        if event.event_type == "final_answer":
            answer = str(event.payload.get("answer", ""))
            if answer:
                self._conversation_store.append_message(
                    session_id=request.session_id,
                    role="assistant",
                    content=answer,
                    payload={"event": event.model_dump(mode="json")},
                    event_type=event.event_type,
                    request_id=request_id,
                    turn_id=event.turn_id,
                )

    def _record_runtime_error(self, request: ChatRequest, exc: Exception) -> None:
        """当运行异常时写入审计错误事件，保证失败也可追溯。"""
        if self._conversation_store is None or request.session_id is None:
            return
        event = AgentEvent(
            turn_id=str(uuid4()),
            sequence=0,
            event_type="error",
            visibility="diagnostic",
            timestamp=datetime.now(UTC).isoformat(),
            payload={"error_type": exc.__class__.__name__, "error": str(exc)},
        )
        self._conversation_store.append_audit_event(
            session_id=request.session_id,
            event=event,
            request_id=_request_id(request),
        )

    def record_events(self, events: list[AgentEvent]) -> None:
        session_state = self._extract_session_state(events)
        if session_state is None:
            return
        for event in events:
            self._memory.append_event(session_state.session_id, event)
        self._record_conversation_events(session_state.session_id, events)
        stop_reason = self._extract_stop_reason(events)
        if events:
            self._memory.append_turn_summary(
                session_state.session_id,
                events[0].turn_id,
                {
                    "stop_reason": stop_reason,
                    "event_count": len(events),
                    "route": self._extract_route(events).model_dump() if self._extract_route(events) else None,
                },
            )

    def _record_conversation_events(self, session_id: str, events: list[AgentEvent]) -> None:
        """把完整回合事件写入 PG 会话记录和审计表。"""
        if self._conversation_store is None or not events:
            return

        self._conversation_store.ensure_session(
            session_id,
            metadata={
                "entrypoint": "chat_window",
                "turn_id": events[0].turn_id,
            },
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

    def run_confirmation(self, request: ConfirmActionRequest) -> Iterator[AgentEvent]:
        context = _RunContext(turn_id=str(uuid4()))
        session_state = self._memory.load(request.session_id)
        yield context.event(
            "turn_started",
            "diagnostic",
            {
                "confirmation": {
                    "session_id": request.session_id,
                    "action_id": request.action_id,
                    "confirmed": request.confirmed,
                }
            },
        )
        yield context.event("session_loaded", "diagnostic", session_state.model_dump())

        action = session_state.pending_action
        if action is None or action.action_id != request.action_id:
            answer = "待确认动作不存在或已失效，请重新发起业务请求。"
            yield from context.user_answer_events(answer)
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "confirmation_blocked"})
            return

        if not request.confirmed:
            session_state = self._state_reducer.apply_confirmation(session_state, request.action_id, confirmed=False)
            self._memory.save(session_state)
            answer = "已取消本次业务动作，未执行任何工具。"
            yield from context.user_answer_events(answer)
            yield context.event("session_updated", "diagnostic", session_state.model_dump())
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "confirmation_cancelled"})
            return

        route = session_state.last_route
        if route is None:
            answer = "确认动作缺少原始路由快照，已阻断执行。"
            yield from context.user_answer_events(answer)
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "confirmation_blocked"})
            return

        confirmed_state = self._state_reducer.apply_confirmation(
            session_state,
            request.action_id,
            confirmed=True,
        )
        tool_call = action.tool_call
        spec = self._tool_registry.spec_for_name(tool_call.tool_name)
        if spec is None:
            answer = f"确认动作引用的工具未注册：{tool_call.tool_name}"
            yield from context.user_answer_events(answer)
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "confirmation_blocked"})
            return

        validation = self._action_policy.validate_tool_action(
            route,
            tool_call,
            [spec],
            session_state=confirmed_state,
        )
        yield context.event(
            "pending_action_validated",
            "diagnostic",
            {
                "allowed": validation.allowed,
                "code": validation.code,
                "reason": validation.reason,
                "action_id": action.action_id,
            },
        )
        if not validation.allowed:
            blocked_state = confirmed_state.model_copy(
                update={"pending_action": None, "confirmation_status": "cancelled"}
            )
            self._memory.save(blocked_state)
            answer = f"确认动作未通过安全校验：{validation.code} {validation.reason}"
            yield from context.user_answer_events(answer)
            yield context.event("session_updated", "diagnostic", blocked_state.model_dump())
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "confirmation_blocked"})
            return

        yield context.event("tool_started", "diagnostic", {"tool_call": tool_call.model_dump()})
        execution = self._tool_registry.execute(
            ChatRequest(
                user_message=request.user_message or action.summary,
                session_id=request.session_id,
                metadata=request.metadata,
            ),
            route,
            tool_call,
            session_state=confirmed_state,
            confirmed_action=action,
        )
        tool_result = execution.result
        yield context.event("tool_result", "diagnostic", tool_result.model_dump())

        updated_state = self._state_reducer.apply_tool_result(confirmed_state, route, tool_result)
        updated_state = updated_state.model_copy(update={"pending_action": None, "confirmation_status": "none"})
        self._memory.save(updated_state)

        if tool_result.status != "success":
            answer = tool_result.user_visible_message or "确认动作执行失败，请稍后重试。"
            stop_reason = "tool_failed"
        else:
            answer = self._answer_from_terminal_tool(
                ChatRequest(user_message=request.user_message or action.summary, session_id=request.session_id),
                route,
                tool_result,
                execution.sources,
            )
            stop_reason = "completed"

        yield from context.user_answer_events(answer)
        yield context.event("session_updated", "diagnostic", updated_state.model_dump())
        yield context.event("turn_finished", "diagnostic", {"stop_reason": stop_reason})

    def _plan_tool(
        self,
        request: ChatRequest,
        route: RouteDecision,
        session_state: SessionStateSnapshot | None = None,
    ) -> ToolCall | None:
        if not route.should_call_tool:
            return None

        # 获取当前场景允许的工具描述
        tool_specs = self._tool_registry.specs_for_categories(route.allowed_tool_categories)
        if not tool_specs:
            return None

        # 按当前阶段展开 Skill：先给索引摘要，再给工具规划所需章节。
        skill_section = self._build_skill_prompt(route.scene, "tool_planning")

        # 构建工具描述
        tool_descriptions = "\n".join(
            f"- `{spec.name}`（{spec.category}）：{spec.description}"
            + (
                f"，必填参数：{', '.join(s.name for s in spec.input_slots if s.required)}"
                if spec.input_slots
                else ""
            )
            for spec in tool_specs
        )

        # Skill 指南放在最前面，作为核心业务指导。
        system_parts: list[str] = []
        if skill_section:
            system_parts.append(
                "# 业务操作指南（必须严格遵守）\n\n"
                "以下 Skill 采用渐进式披露：索引用于确认能力边界，当前 Skill 展开部分才是本阶段的执行依据。"
                "你必须按照当前 Skill 的决策流程决定是否调用工具、调用哪个工具。\n\n"
            )
            system_parts.append(skill_section)
            system_parts.append("\n\n---\n\n")

        system_parts.append(
            "# 你的角色\n"
            "你是信易贷业务工具规划器。严格按照上方业务操作指南中的决策流程，"
            "结合用户问题和当前状态，决定是否调用工具以及调用哪个工具。\n\n"
        )
        system_parts.append(f"## 可用工具\n{tool_descriptions}\n\n")
        system_parts.append(
            "## 输出格式（只输出 JSON，不要输出其他文字）\n"
            "如果需要调用工具，输出：\n"
            '{"tool_name": "工具名", "arguments": {参数}, "reason": "调用原因"}\n'
            "如果不需要调用工具（例如缺少必填信息需要追问），输出：\n"
            '{"tool_name": null, "reason": "不调用工具的原因，以及需要向用户追问的内容"}\n'
        )

        # 构建用户消息，包含会话状态摘要
        user_parts = [
            f"用户问题：{request.user_message}\n",
            f"路由场景：{route.scene}\n",
            f"已填槽位：{route.filled_slots}\n",
            f"缺失槽位：{route.missing_slots}\n",
            f"允许工具：{route.allowed_tools}",
        ]
        if session_state and session_state.short_summary:
            user_parts.insert(0, f"会话摘要：{session_state.short_summary}\n")
        if session_state and session_state.last_tool_results:
            last_results = [
                f"  - {r.tool_name}: {r.business_status or 'unknown'}" for r in session_state.last_tool_results[-2:]
            ]
            user_parts.append(f"\n最近工具调用：\n" + "\n".join(last_results))

        messages = [
            {"role": "system", "content": "".join(system_parts)},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

        raw = self._model.complete(messages, response_format=JSON_OBJECT_RESPONSE_FORMAT)
        planned_call = self._parse_tool_plan(raw, route, tool_specs)
        if planned_call is not None:
            return planned_call

        return self._build_deterministic_tool_call(request, route, tool_specs)

    def _parse_tool_plan(
        self,
        raw: str,
        route: RouteDecision,
        tool_specs: list,
    ) -> ToolCall | None:
        """解析模型的工具规划输出。"""
        import json
        import re

        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None

        try:
            payload = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return None

        tool_name = payload.get("tool_name")
        if not tool_name:
            return None

        # 校验工具是否在允许列表中
        if tool_name not in route.allowed_tools:
            return None

        arguments = payload.get("arguments", {})
        reason = payload.get("reason", "")

        # 从 tool_specs 中查找对应工具的元信息
        spec_map = {spec.name: spec for spec in tool_specs}
        spec = spec_map.get(tool_name)
        tool_category = spec.category if spec else "utility"
        risk_level = spec.risk_level if spec else "read_only"
        requires_confirmation = spec.requires_confirmation if spec else False

        return ToolCall(
            tool_call_id=str(uuid4()),
            tool_name=tool_name,
            tool_category=tool_category,
            arguments=arguments,
            risk_level=risk_level,
            confirmation_required=self._risk_policy.requires_confirmation(
                risk_level,
                requires_confirmation or route.confirmation_required,
            ),
            reason=reason,
        )

    def _build_deterministic_tool_call(
        self,
        request: ChatRequest,
        route: RouteDecision,
        tool_specs: list,
    ) -> ToolCall | None:
        """当路由只允许一个明确工具时，用后端槽位策略兜底生成调用。"""
        if len(route.allowed_tools) != 1 or len(tool_specs) != 1:
            return None

        spec = tool_specs[0]
        if spec.name != route.allowed_tools[0]:
            return None

        arguments: dict[str, object] = {}
        for slot in spec.input_slots:
            value = self._default_tool_argument(slot.name, request, route)
            if value is None and slot.required:
                return None
            if value is not None:
                arguments[slot.name] = value

        return ToolCall(
            tool_call_id=str(uuid4()),
            tool_name=spec.name,
            tool_category=spec.category,
            arguments=arguments,
            risk_level=spec.risk_level,
            confirmation_required=self._risk_policy.requires_confirmation(
                spec.risk_level,
                spec.requires_confirmation or route.confirmation_required,
            ),
            reason="模型未返回有效工具规划，后端根据单一允许工具和已确认槽位生成兜底调用。",
        )

    def _default_tool_argument(
        self,
        slot_name: str,
        request: ChatRequest,
        route: RouteDecision,
    ) -> object | None:
        """从稳定上下文中提取工具入参，不猜测业务字段。"""
        if slot_name == "query":
            return request.user_message
        if slot_name == "top_k":
            return request.top_k
        if slot_name in route.filled_slots:
            return route.filled_slots[slot_name]
        if slot_name in request.metadata:
            return request.metadata[slot_name]
        return None

    def _build_pending_action(
        self,
        request: ChatRequest,
        route: RouteDecision,
        session_state: SessionStateSnapshot,
        tool_call: ToolCall,
    ) -> PendingAction | None:
        if not tool_call.confirmation_required:
            return None

        spec = self._tool_registry.spec_for_name(tool_call.tool_name)
        if spec is None:
            return None

        return self._pending_action_builder.build(
            request=request,
            route=route,
            state=session_state,
            tool_call=tool_call,
            spec=spec,
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
        answer = self._summarize_tool_result(request, route, tool_result, sources)
        return self._evidence_policy.attach_citations(answer, sources)

    def _answer_without_tool(self, request: ChatRequest, route: RouteDecision) -> str:
        skill_section = self._build_skill_prompt(route.scene, "direct_answer")
        system_parts: list[str] = []
        if skill_section:
            system_parts.append(
                "# 业务操作指南（必须严格遵守）\n\n"
                f"{skill_section}\n\n---\n\n"
            )
        system_parts.append("你是信易贷聊天助手。严格按照上方业务操作指南回答用户问题，不发起工具调用。")
        messages = [
            {
                "role": "system",
                "content": "".join(system_parts),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{request.user_message}\n\n"
                    f"路由场景：{route.scene}\n"
                    f"已填槽位：{route.filled_slots}\n"
                    f"缺失槽位：{route.missing_slots}"
                ),
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
        skill_section = self._build_skill_prompt(route.scene, "tool_result_answer")
        system_parts: list[str] = []
        if skill_section:
            system_parts.append(
                "# 业务操作指南（必须严格遵守）\n\n"
                f"{skill_section}\n\n---\n\n"
            )
        system_parts.append(
            "你是信易贷聊天助手。严格按照上方业务操作指南，基于工具结果和证据回答用户问题；"
            "数值类结果必须说明来自工具，证据不足时要说明缺口；"
            "不要伪造申请状态、授权状态或贷款审批结论。"
        )
        return [
            {
                "role": "system",
                "content": "".join(system_parts),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{user_message}\n\n"
                    f"路由场景：{route.scene}\n"
                    f"已填槽位：{route.filled_slots}\n\n"
                    f"工具结果：{tool_result.model_dump()}\n\n"
                    f"可用证据：\n{source_text}"
                ),
            },
        ]

    def _build_skill_prompt(self, scene: str, disclosure: str) -> str:
        """按渐进式披露组装 Skill prompt。"""
        index_section = self._skills.get_index_prompt_section()
        current_section = self._skills.get_prompt_section(scene, disclosure)
        parts: list[str] = []
        if index_section:
            parts.append(index_section)
        if current_section:
            parts.append("# 当前 Skill 展开（按需层）\n\n" + current_section)
        return "\n\n---\n\n".join(parts)

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


def _extract_user_message(events: list[AgentEvent]) -> str:
    """从回合事件中提取可落库展示的用户输入。"""
    if not events:
        return ""
    first_payload = events[0].payload
    if first_payload.get("user_message"):
        return str(first_payload["user_message"])
    confirmation = first_payload.get("confirmation")
    if isinstance(confirmation, dict):
        action_id = confirmation.get("action_id", "")
        confirmed = confirmation.get("confirmed")
        action = "确认" if confirmed else "取消"
        return f"{action}待确认动作：{action_id}"
    return ""


def _request_id(request: ChatRequest) -> str | None:
    """从请求 metadata 中读取审计请求 ID。"""
    value = request.metadata.get("request_id")
    return str(value) if value else None


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
