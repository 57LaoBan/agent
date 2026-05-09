"""ReAct 主运行时：路由、快速通道、多步观察和确认流的唯一编排入口。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.capabilities.catalog import CapabilityCatalog, default_capability_catalog
from xinyidai_agent.evidence_policy import EvidencePolicy
from xinyidai_agent.llm import ChatModel
from xinyidai_agent.memory import MemoryManager
from xinyidai_agent.pending_actions import PendingActionBuilder
from xinyidai_agent.protocol import (
    AgentEvent,
    AgentEventType,
    AgentState,
    ChatRequest,
    ConfirmActionRequest,
    EventVisibility,
    PendingAction,
    ReactAction,
    ReactStepDecision,
    RetrievalTrace,
    RouteDecision,
    SessionStateSnapshot,
    SourceDocument,
    StateTransition,
    ToolCall,
    ToolExecutionObservation,
    ToolResult,
)
from xinyidai_agent.router.action_validator import PendingActionValidator
from xinyidai_agent.runtime.fast_path import FastPathRunner
from xinyidai_agent.runtime.react_engine import ReactStepEngine, StepOutcome
from xinyidai_agent.runtime.react_observation_builder import ReactObservationBuilder
from xinyidai_agent.runtime_state import RuntimeStateReducer
from xinyidai_agent.system_tools import SessionMemorySystemTool
from xinyidai_agent.tools.base import ToolSpec
from xinyidai_agent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ReactRunResult:
    """ReAct runtime 的内聚执行结果，主要供测试和外层响应组装使用。"""

    events: list[AgentEvent]


class ReactRuntime:
    """生产 ReAct 主循环，所有状态更新都通过不可变快照和 reducer 完成。"""

    def __init__(
        self,
        *,
        model: ChatModel,
        router: object,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        capability_catalog: CapabilityCatalog | None = None,
        fast_path: FastPathRunner | None = None,
        react_engine: ReactStepEngine | None = None,
        observation_builder: ReactObservationBuilder | None = None,
        pending_action_builder: PendingActionBuilder | None = None,
        evidence_policy: EvidencePolicy | None = None,
    ) -> None:
        """初始化运行时依赖图。"""
        self._model = model
        self._router = router
        self._tools = tool_registry
        self._memory = memory_manager
        self._catalog = capability_catalog or CapabilityCatalog(default_capability_catalog())
        self._fast_path = fast_path or FastPathRunner(tool_registry, model)
        self._react_engine = react_engine or ReactStepEngine(model, tool_registry)
        self._observation_builder = observation_builder or ReactObservationBuilder(tool_registry)
        self._pending_action_builder = pending_action_builder or PendingActionBuilder()
        self._evidence = evidence_policy or EvidencePolicy()
        self._state_reducer = RuntimeStateReducer()
        self._pending_validator = PendingActionValidator()
        self._session_memory_tool = SessionMemorySystemTool()

    @property
    def memory(self) -> MemoryManager:
        """暴露内存管理器，兼容既有测试和调试入口。"""
        return self._memory

    def run(self, request: ChatRequest) -> Iterator[AgentEvent]:
        """执行一轮普通聊天请求。"""
        context = _RunContext(turn_id=str(uuid4()))
        state = self._memory.load(request.session_id)
        request = request.model_copy(update={"session_id": state.session_id})
        yield context.event("turn_started", "diagnostic", {"user_message": request.user_message})
        yield context.event("session_loaded", "diagnostic", state.model_dump())

        pre_route = self._pre_route(request, has_pending_action=state.pending_action is not None)
        if pre_route is not None:
            yield from self._finish_direct(context, state, request, pre_route[0], pre_route[1])
            return

        memory_execution = self._session_memory_tool.plan_and_execute(
            self._model,
            request,
            state,
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
            state = memory_execution.state
            self._memory.save(state)
            yield context.event("session_updated", "diagnostic", state.model_dump())

        enriched = self._memory.enrich_request(request, state)

        yield context.transition("LISTENING", "ROUTING", "开始识别业务意图")
        yield context.event("route_started", "diagnostic", {"strategy": "rules_guard_scene_direct_model_policy"})
        route = self._route(enriched)
        yield context.event("route_decision", "diagnostic", route.model_dump())
        routed_state = self._state_reducer.apply_route(state, route)

        capability = self._capability_for_route(route)
        if route.missing_slots:
            answer = self._clarifying_question(route)
            yield context.event(
                "runtime_step_decision",
                "diagnostic",
                self._step_event_payload("ask_user", "missing_slots", route.confidence),
            )
            yield from context.user_answer_events(answer)
            updated = self._memory.update(
                state,
                enriched,
                route=route,
                stop_reason="missing_slots",
                final_answer=answer,
            )
            yield context.event("session_updated", "diagnostic", updated.model_dump())
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "missing_slots"})
            return

        if routed_state.pending_action is not None:
            answer = "当前已有待确认动作，请先确认或取消后再继续。"
            yield context.event(
                "runtime_step_decision",
                "diagnostic",
                self._step_event_payload("ask_user", "pending_action_exists", route.confidence),
            )
            yield from context.user_answer_events(answer)
            yield context.event("session_updated", "diagnostic", routed_state.model_dump())
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "waiting_confirmation"})
            return

        if not route.should_call_tool or not capability.allowed_tools:
            answer = self._compose_no_tool_answer(enriched, route)
            yield context.event(
                "runtime_step_decision",
                "diagnostic",
                self._step_event_payload("answer", "no_tool_required", route.confidence),
            )
            yield from context.user_answer_events(answer)
            updated = self._memory.update(
                state,
                enriched,
                route=route,
                stop_reason="answered_without_tool",
                final_answer=answer,
            )
            yield context.event("session_updated", "diagnostic", updated.model_dump())
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "answered_without_tool"})
            return

        fast = self._fast_path.run(enriched, route, capability, routed_state)
        if fast.succeeded and fast.tool_result is not None:
            yield context.event(
                "runtime_step_decision",
                "diagnostic",
                self._step_event_payload("call_tool", "fast_path", route.confidence),
            )
            yield context.event("tool_result", "diagnostic", fast.tool_result.model_dump())
            answer = fast.final_answer or self._compose_tool_answer(enriched, route, fast.tool_result, fast.sources)
            yield from context.user_answer_events(answer)
            updated = self._memory.update(
                state,
                enriched,
                route=route,
                tool_result=fast.tool_result,
                stop_reason="completed",
                final_answer=answer,
            )
            yield context.event("session_updated", "diagnostic", updated.model_dump())
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "completed"})
            return

        direct_pending = self._pending_from_route(enriched, route, capability, routed_state)
        if direct_pending is not None:
            answer = direct_pending.summary
            yield context.event(
                "runtime_step_decision",
                "diagnostic",
                self._step_event_payload("call_tool", "confirmation_required", route.confidence),
            )
            yield context.event(
                "confirmation_required",
                "diagnostic",
                {"pending_action": direct_pending.model_dump()},
            )
            yield from context.user_answer_events(answer)
            updated = self._memory.update(
                state,
                enriched,
                route=route,
                pending_action=direct_pending,
                stop_reason="waiting_confirmation",
                final_answer=answer,
            )
            yield context.event("session_updated", "diagnostic", updated.model_dump())
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "waiting_confirmation"})
            return

        direct_tool = self._tool_call_from_route(enriched, route, capability)
        if direct_tool is not None and self._can_execute_route_tool(route, direct_tool):
            yield from self._execute_direct_tool(context, state, routed_state, enriched, route, capability, direct_tool)
            return

        yield from self._run_react_steps(context, state, routed_state, enriched, route, capability)

    def run_confirmation(self, request: ConfirmActionRequest) -> Iterator[AgentEvent]:
        """执行用户对 pending action 的确认或取消。"""
        context = _RunContext(turn_id=str(uuid4()))
        state = self._memory.load(request.session_id)
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
        yield context.event("session_loaded", "diagnostic", state.model_dump())

        action = state.pending_action
        if action is None or action.action_id != request.action_id:
            answer = "待确认动作不存在或已失效，请重新发起业务请求。"
            yield from context.user_answer_events(answer)
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "confirmation_blocked"})
            return

        if not request.confirmed:
            cancelled = self._state_reducer.apply_confirmation(state, request.action_id, confirmed=False)
            self._memory.save(cancelled)
            answer = "已取消本次业务动作，未执行任何工具。"
            yield from context.user_answer_events(answer)
            yield context.event("session_updated", "diagnostic", cancelled.model_dump())
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "confirmation_cancelled"})
            return

        route = state.last_route
        if route is None:
            answer = "确认动作缺少原始路由快照，已阻断执行。"
            yield from context.user_answer_events(answer)
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "confirmation_blocked"})
            return

        spec = self._tools.spec_for_name(action.tool_call.tool_name)
        if spec is None:
            answer = f"确认动作引用的工具未注册：{action.tool_call.tool_name}"
            yield from context.user_answer_events(answer)
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "confirmation_blocked"})
            return

        confirmed_state = self._state_reducer.apply_confirmation(state, request.action_id, confirmed=True)
        validation = self._pending_validator.validate_pending_action_snapshot(
            state=confirmed_state,
            route=route,
            action=action,
            tool_call=action.tool_call,
            spec=spec,
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
            blocked = confirmed_state.model_copy(
                update={"pending_action": None, "confirmation_status": "cancelled"}
            )
            self._memory.save(blocked)
            answer = f"确认动作未通过安全校验：{validation.code} {validation.reason}"
            yield from context.user_answer_events(answer)
            yield context.event("session_updated", "diagnostic", blocked.model_dump())
            yield context.event("turn_finished", "diagnostic", {"stop_reason": "confirmation_blocked"})
            return

        chat_request = ChatRequest(
            user_message=request.user_message or action.summary,
            session_id=request.session_id,
            metadata=request.metadata,
        )
        yield context.event("tool_started", "diagnostic", {"tool_call": action.tool_call.model_dump()})
        execution, _observation = self._tools.execute_observed(
            chat_request,
            route,
            action.tool_call,
            capability=self._capability_for_route(route),
            session_state=confirmed_state,
            confirmed_action=action,
            pending_action_validated=True,
        )
        result = execution.result
        yield context.event("tool_result", "diagnostic", result.model_dump())

        if result.status == "success":
            stop_reason = "completed"
            answer = self._compose_tool_answer(chat_request, route, result, execution.sources)
        else:
            stop_reason = "tool_failed"
            answer = result.user_visible_message or "确认动作执行失败，请稍后重试。"

        updated = self._memory.update(
            confirmed_state,
            chat_request,
            route=route,
            tool_result=result,
            stop_reason=stop_reason,
            final_answer=answer,
        )
        yield from context.user_answer_events(answer)
        yield context.event("session_updated", "diagnostic", updated.model_dump())
        yield context.event("turn_finished", "diagnostic", {"stop_reason": stop_reason})

    def _run_react_steps(
        self,
        context: _RunContext,
        initial_state: SessionStateSnapshot,
        routed_state: SessionStateSnapshot,
        request: ChatRequest,
        route: RouteDecision,
        capability: CapabilityPolicy,
    ) -> Iterator[AgentEvent]:
        """执行多步 ReAct，并将每次工具执行转换为 observation。"""
        executed: list[ToolExecutionObservation] = []
        last_observation: ToolExecutionObservation | None = None
        invalid_hint: str | None = None
        last_tool_result: ToolResult | None = None
        last_sources: list[SourceDocument] = []

        max_steps = capability.max_react_steps
        for step in range(1, max_steps + 1):
            yield context.event(
                "runtime_step_started",
                "diagnostic",
                {"step_index": step, "max_steps": max_steps, "stage": routed_state.current_stage},
            )
            observation = self._observation_builder.build(
                step=step,
                max_steps=max_steps,
                request=request,
                route=route,
                capability=capability,
                session_state=routed_state,
                executed_tools=executed,
                last_observation=last_observation,
                invalid_action_hint=invalid_hint,
            )
            try:
                outcome = self._react_engine.step(observation)
            except RuntimeError as exc:
                fallback = self._tool_call_from_route(request, route, capability)
                if fallback is not None and self._can_execute_route_tool(route, fallback):
                    yield from self._execute_direct_tool(
                        context,
                        initial_state,
                        routed_state,
                        request,
                        route,
                        capability,
                        fallback,
                    )
                    return
                answer = f"运行时无法解析模型动作，已阻断执行：{type(exc).__name__}"
                yield context.event(
                    "runtime_step_decision",
                    "diagnostic",
                    self._step_event_payload("handoff", "runtime_error", route.confidence),
                )
                yield from context.user_answer_events(answer)
                updated = self._memory.update(
                    initial_state,
                    request,
                    route=route,
                    stop_reason="runtime_error",
                    final_answer=answer,
                )
                yield context.event("session_updated", "diagnostic", updated.model_dump())
                yield context.event("turn_finished", "diagnostic", {"stop_reason": "runtime_error"})
                return

            yield context.event("runtime_step_decision", "diagnostic", outcome.decision.model_dump())
            if outcome.kind == "invalid_action":
                invalid_hint = outcome.invalid_hint
                continue
            if outcome.pending_action_required and outcome.pending_tool_call is not None:
                pending = self._build_pending_action(request, route, routed_state, outcome.pending_tool_call)
                yield context.event("confirmation_required", "diagnostic", {"pending_action": pending.model_dump()})
                yield from context.user_answer_events(pending.summary)
                updated = self._memory.update(
                    initial_state,
                    request,
                    route=route,
                    pending_action=pending,
                    stop_reason="waiting_confirmation",
                    final_answer=pending.summary,
                )
                yield context.event("session_updated", "diagnostic", updated.model_dump())
                yield context.event("turn_finished", "diagnostic", {"stop_reason": "waiting_confirmation"})
                return
            if outcome.kind == "call_tool_executed" and outcome.tool_execution is not None:
                last_tool_result = outcome.tool_execution.result
                last_sources = list(outcome.tool_execution.sources)
                if outcome.tool_observation is not None:
                    last_observation = outcome.tool_observation
                    executed.append(outcome.tool_observation)
                yield context.event("tool_result", "diagnostic", last_tool_result.model_dump())
                if last_tool_result.terminal:
                    answer = self._compose_tool_answer(request, route, last_tool_result, last_sources)
                    yield from context.user_answer_events(answer)
                    updated = self._memory.update(
                        initial_state,
                        request,
                        route=route,
                        tool_result=last_tool_result,
                        stop_reason="completed" if last_tool_result.status == "success" else "tool_failed",
                        final_answer=answer,
                    )
                    yield context.event("session_updated", "diagnostic", updated.model_dump())
                    yield context.event(
                        "turn_finished",
                        "diagnostic",
                        {
                            "stop_reason": "completed" if last_tool_result.status == "success" else "tool_failed"
                        },
                    )
                    return
                invalid_hint = None
                continue
            if outcome.kind == "terminal":
                yield from self._finish_terminal(
                    context,
                    initial_state,
                    request,
                    route,
                    outcome,
                    last_tool_result,
                    last_sources,
                )
                return

        answer = "本轮工具推理已达到最大步数，已停止执行，请补充更明确的信息或转人工处理。"
        yield from context.user_answer_events(answer)
        updated = self._memory.update(
            initial_state,
            request,
            route=route,
            tool_result=last_tool_result,
            stop_reason="max_steps",
            final_answer=answer,
        )
        yield context.event("session_updated", "diagnostic", updated.model_dump())
        yield context.event("turn_finished", "diagnostic", {"stop_reason": "max_steps"})

    def _finish_terminal(
        self,
        context: _RunContext,
        initial_state: SessionStateSnapshot,
        request: ChatRequest,
        route: RouteDecision,
        outcome: StepOutcome,
        last_tool_result: ToolResult | None,
        last_sources: list[SourceDocument],
    ) -> Iterator[AgentEvent]:
        """收敛 ReAct 终止动作。"""
        action = outcome.decision.action
        if action.type == "ask_user":
            answer = action.message or self._clarifying_question(route)
            stop_reason = "missing_slots" if route.missing_slots else "ask_user"
        elif action.type == "handoff":
            answer = action.message or "当前请求需要人工处理。"
            stop_reason = "handoff"
        elif action.type == "answer":
            answer = action.final_answer or ""
            if last_sources:
                answer = self._evidence.attach_citations(answer, last_sources)
            stop_reason = "completed"
        elif last_tool_result is not None:
            answer = self._compose_tool_answer(request, route, last_tool_result, last_sources)
            stop_reason = "completed" if last_tool_result.status == "success" else "tool_failed"
        else:
            answer = self._compose_no_tool_answer(request, route)
            stop_reason = "answered_without_tool"

        yield from context.user_answer_events(answer)
        updated = self._memory.update(
            initial_state,
            request,
            route=route,
            tool_result=last_tool_result,
            stop_reason=stop_reason,
            final_answer=answer,
        )
        yield context.event("session_updated", "diagnostic", updated.model_dump())
        yield context.event("turn_finished", "diagnostic", {"stop_reason": stop_reason})

    def _execute_direct_tool(
        self,
        context: _RunContext,
        initial_state: SessionStateSnapshot,
        routed_state: SessionStateSnapshot,
        request: ChatRequest,
        route: RouteDecision,
        capability: CapabilityPolicy,
        tool_call: ToolCall,
    ) -> Iterator[AgentEvent]:
        """执行由路由白名单唯一确定的只读工具。"""
        yield context.event(
            "runtime_step_decision",
            "diagnostic",
            self._step_event_payload("call_tool", "single_tool_route", route.confidence),
        )
        execution, _observation = self._tools.execute_observed(
            request,
            route,
            tool_call,
            capability=capability,
            session_state=routed_state,
        )
        result = execution.result
        yield context.event("tool_result", "diagnostic", result.model_dump())
        answer = self._compose_tool_answer(request, route, result, execution.sources)
        stop_reason = "completed" if result.business_status == "TOOL_UNAVAILABLE" else (
            "completed" if result.status == "success" else "tool_failed"
        )
        yield from context.user_answer_events(answer)
        updated = self._memory.update(
            initial_state,
            request,
            route=route,
            tool_result=result,
            stop_reason=stop_reason,
            final_answer=answer,
        )
        yield context.event("session_updated", "diagnostic", updated.model_dump())
        yield context.event("turn_finished", "diagnostic", {"stop_reason": stop_reason})

    def _finish_direct(
        self,
        context: _RunContext,
        state: SessionStateSnapshot,
        request: ChatRequest,
        answer: str,
        stop_reason: str,
    ) -> Iterator[AgentEvent]:
        """完成规则守卫或场景直达返回。"""
        yield from context.user_answer_events(answer)
        updated = self._memory.update(
            state,
            request,
            stop_reason=stop_reason,
            final_answer=answer,
        )
        yield context.event("session_updated", "diagnostic", updated.model_dump())
        yield context.event("turn_finished", "diagnostic", {"stop_reason": stop_reason})

    def _pre_route(
        self,
        request: ChatRequest,
        *,
        has_pending_action: bool,
    ) -> tuple[str, str] | None:
        """调用路由器的前置短路能力；旧测试路由器没有该方法时自动跳过。"""
        pre_route = getattr(self._router, "pre_route", None)
        if pre_route is None:
            return None
        outcome = pre_route(request, has_pending_action=has_pending_action)
        if outcome.kind in {"guard_reject", "scene_direct"} and outcome.final_answer:
            return outcome.final_answer, outcome.stop_reason or outcome.kind
        return None

    def _route(self, request: ChatRequest) -> RouteDecision:
        """通过注入的路由器产出 RouteDecision。"""
        route = getattr(self._router, "route")
        return route(request)

    def _capability_for_route(self, route: RouteDecision) -> CapabilityPolicy:
        """根据路由 capability_id 或 intent 找到能力策略，找不到时 fail-closed 到 unknown。"""
        for policy in self._catalog.policies:
            if route.capability_id and policy.capability_id == route.capability_id:
                return policy
            if policy.standard_intent == route.intent:
                return policy
        for policy in self._catalog.policies:
            if policy.capability_id == "unknown.clarify":
                return policy
        raise RuntimeError("能力目录缺少 unknown.clarify 兜底能力")

    def _pending_from_route(
        self,
        request: ChatRequest,
        route: RouteDecision,
        capability: CapabilityPolicy,
        routed_state: SessionStateSnapshot,
    ) -> PendingAction | None:
        """对需要确认的状态变更动作生成 pending action 快照。"""
        tool_call = self._tool_call_from_route(request, route, capability)
        if tool_call is None:
            return None
        spec = self._tools.spec_for_name(tool_call.tool_name)
        if spec is None or not _requires_confirmation(route, spec):
            return None
        return self._build_pending_action(request, route, routed_state, tool_call)

    def _build_pending_action(
        self,
        request: ChatRequest,
        route: RouteDecision,
        routed_state: SessionStateSnapshot,
        tool_call: ToolCall,
    ) -> PendingAction:
        """生成可校验的 pending action。"""
        spec = self._tools.spec_for_name(tool_call.tool_name)
        if spec is None:
            raise RuntimeError(f"pending action 工具未注册：{tool_call.tool_name}")
        return self._pending_action_builder.build(
            request=request,
            route=route,
            state=routed_state,
            tool_call=tool_call,
            spec=spec,
        )

    def _tool_call_from_route(
        self,
        request: ChatRequest,
        route: RouteDecision,
        capability: CapabilityPolicy,
    ) -> ToolCall | None:
        """根据路由白名单构造工具调用，只使用显式声明的工具和槽位。"""
        candidate_names = list(route.allowed_tools or capability.allowed_tools)
        if not candidate_names:
            candidate_names = self._tools.names_for_categories(route.allowed_tool_categories)
        for name in candidate_names:
            spec = self._tools.spec_for_name(name)
            if spec is None:
                if route.allowed_tools:
                    return ToolCall(
                        tool_call_id=str(uuid4()),
                        tool_name=name,
                        tool_category=route.allowed_tool_categories[0] if route.allowed_tool_categories else None,
                        arguments={},
                        risk_level=route.risk_level,
                        confirmation_required=route.confirmation_required,
                        reason="route_declared_unregistered_tool",
                    )
                continue
            arguments = self._arguments_for_spec(request, route, spec)
            if self._missing_required_arguments(spec, arguments):
                continue
            return ToolCall(
                tool_call_id=str(uuid4()),
                tool_name=spec.name,
                tool_category=spec.category,
                arguments=arguments,
                risk_level=spec.risk_level,
                confirmation_required=_requires_confirmation(route, spec),
                reason="route_single_tool",
            )
        return None

    def _arguments_for_spec(
        self,
        request: ChatRequest,
        route: RouteDecision,
        spec: ToolSpec,
    ) -> dict[str, object]:
        """从请求、路由槽位和会话 metadata 中组装工具入参。"""
        values: dict[str, object] = {}
        merged = {**request.metadata, **route.filled_slots}
        for slot in spec.input_slots:
            if slot.name in merged and merged[slot.name] not in (None, ""):
                values[slot.name] = merged[slot.name]
            elif slot.name == "query":
                values[slot.name] = request.user_message
            elif slot.name == "top_k":
                values[slot.name] = request.top_k
        return values

    @staticmethod
    def _missing_required_arguments(spec: ToolSpec, arguments: dict[str, object]) -> bool:
        """判断工具必填入参是否缺失。"""
        for slot in spec.input_slots:
            value = arguments.get(slot.name)
            if slot.required and (value is None or value == ""):
                return True
        return False

    def _can_execute_route_tool(self, route: RouteDecision, tool_call: ToolCall) -> bool:
        """只允许无需确认的工具在路由直连路径执行。"""
        spec = self._tools.spec_for_name(tool_call.tool_name)
        if spec is None:
            return True
        return not _requires_confirmation(route, spec)

    def _compose_tool_answer(
        self,
        request: ChatRequest,
        route: RouteDecision,
        result: ToolResult,
        sources: list[SourceDocument],
    ) -> str:
        """基于工具结果生成用户可见回答。"""
        if result.user_visible_message:
            return result.user_visible_message

        trace = self._retrieval_trace(result)
        evidence = self._evidence.assess(route, sources, trace)
        if not evidence.ready and evidence.user_message:
            return evidence.user_message

        messages = [
            {
                "role": "system",
                "content": (
                    "你是信易贷聊天助手。必须基于工具结果和证据回答；"
                    "不要编造政策、授信额度、申请状态或授权链接。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{request.user_message}\n"
                    f"路由：{route.model_dump()}\n"
                    f"工具结果：{result.model_dump()}\n"
                    f"证据：{self._format_sources(sources)}"
                ),
            },
        ]
        answer = self._model.complete(messages)
        return self._evidence.attach_citations(answer, sources)

    def _compose_no_tool_answer(self, request: ChatRequest, route: RouteDecision) -> str:
        """对无需工具的闲聊或澄清场景生成回答。"""
        if route.scene == "UNKNOWN":
            return self._clarifying_question(route)
        messages = [
            {
                "role": "system",
                "content": "你是信易贷聊天助手。当前场景无需调用工具，请简洁回答用户。",
            },
            {
                "role": "user",
                "content": f"用户问题：{request.user_message}\n路由：{route.model_dump()}",
            },
        ]
        return self._model.complete(messages)

    @staticmethod
    def _retrieval_trace(result: ToolResult) -> RetrievalTrace | None:
        """从工具结果中恢复检索 trace。"""
        raw = result.output.get("retrieval_trace")
        if isinstance(raw, dict):
            return RetrievalTrace.model_validate(raw)
        return None

    @staticmethod
    def _format_sources(sources: list[SourceDocument]) -> str:
        """格式化证据片段。"""
        if not sources:
            return "暂无检索证据。"
        return "\n\n".join(
            f"[{index}] {source.title}\n{source.content}"
            for index, source in enumerate(sources, start=1)
        )

    @staticmethod
    def _clarifying_question(route: RouteDecision) -> str:
        """根据缺失槽位生成澄清问题。"""
        if "user_intent" in route.missing_slots:
            return "我还没判断出您要办理哪类事项。您想咨询政策、查询授信额度，还是发起贷款申请？"
        labels = {
            "company_name": "企业名称",
            "product_name": "贷款产品",
            "application_id": "申请编号",
        }
        readable = [labels.get(slot, slot) for slot in route.missing_slots]
        if readable:
            return f"为了继续处理这个请求，请先补充：{'、'.join(readable)}。"
        return "请补充更明确的业务需求。"

    @staticmethod
    def _step_event_payload(action_type: str, reason: str, confidence: float) -> dict[str, object]:
        """生成运行时决策事件，保持旧事件名但使用 ReAct action 结构。"""
        return ReactStepDecision(
            thought=reason,
            action=ReactAction(type=action_type, message=reason) if action_type != "call_tool" else ReactAction(
                type="call_tool",
                tool_name="route_selected_tool",
            ),
            confidence=confidence,
        ).model_dump()


def _requires_confirmation(route: RouteDecision, spec: ToolSpec) -> bool:
    """统一判断工具是否需要用户确认。"""
    return spec.requires_confirmation or route.confirmation_required or spec.risk_level in {
        "link_create",
        "state_create",
        "state_update",
        "final_submit",
    }


class _RunContext:
    """单轮事件上下文，保证事件序号单调递增。"""

    def __init__(self, turn_id: str) -> None:
        self.turn_id = turn_id
        self.sequence = 0

    def event(
        self,
        event_type: AgentEventType,
        visibility: EventVisibility,
        payload: dict[str, object],
    ) -> AgentEvent:
        """创建标准 AgentEvent。"""
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
        """创建状态迁移事件。"""
        transition = StateTransition(from_state=from_state, to_state=to_state, reason=reason)
        return self.event("state_changed", "diagnostic", transition.model_dump())

    def user_answer_events(self, answer: str) -> Iterator[AgentEvent]:
        """按 chunk 输出用户可见答案，并在最后给出 final_answer。"""
        chunks = [answer[index : index + 12] for index in range(0, len(answer), 12)] or [""]
        current = ""
        for chunk in chunks:
            current += chunk
            yield self.event("assistant_delta", "user", {"text": chunk, "answer": current})
        yield self.event("final_answer", "user", {"answer": answer})


def measure_events(call: Iterator[AgentEvent]) -> tuple[list[AgentEvent], float]:
    """收集事件并返回耗时，供兼容层构造 ChatResponse。"""
    started = perf_counter()
    events = list(call)
    return events, (perf_counter() - started) * 1000
