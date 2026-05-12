from __future__ import annotations

from collections.abc import Iterable
import time

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.protocol import (
    ChatRequest,
    PendingAction,
    ReactObservationKind,
    RouteDecision,
    SessionStateSnapshot,
    ToolCall,
    ToolCategory,
    ToolExecutionObservation,
    ToolResult,
    ToolResultEnvelope,
)
from xinyidai_agent.policies import RiskPolicy
from xinyidai_agent.rag import Retriever
from xinyidai_agent.router.action_validator import PendingActionValidator
from xinyidai_agent.tools.base import AgentTool, SlotSpec, ToolExecution, ToolSpec
from xinyidai_agent.tools.mock_credit import MockCreditAmountTool
from xinyidai_agent.tools.rag_search import RagSearchTool


class ToolRegistry:
    def __init__(self, tools: Iterable[AgentTool]) -> None:
        tool_list = list(tools)
        self._tools = {tool.name: tool for tool in tool_list}
        self._pending_action_validator = PendingActionValidator()
        self._risk_policy = RiskPolicy()
        self._tools_by_category: dict[ToolCategory, list[AgentTool]] = {}
        for tool in tool_list:
            self._tools_by_category.setdefault(tool.category, []).append(tool)

    def names_for_categories(self, categories: Iterable[ToolCategory]) -> list[str]:
        names: list[str] = []
        for category in categories:
            names.extend(tool.name for tool in self._tools_by_category.get(category, []))
        return names

    def specs_for_categories(self, categories: Iterable[ToolCategory]) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for category in categories:
            specs.extend(tool.spec() for tool in self._tools_by_category.get(category, []))
        return specs

    def spec_for_name(self, name: str) -> ToolSpec | None:
        tool = self._tools.get(name)
        return tool.spec() if tool is not None else None

    def register_external(self, tool: AgentTool) -> None:
        """运行期注入外部工具，例如 McpToolAdapter。

        重名直接失败，避免无声覆盖已有工具。
        """
        if tool.name in self._tools:
            raise ValueError(f"工具名重复注册：{tool.name}")
        self._tools[tool.name] = tool
        self._tools_by_category.setdefault(tool.category, []).append(tool)

    def execute(
        self,
        request: ChatRequest,
        route: RouteDecision,
        tool_call: ToolCall,
        session_state: SessionStateSnapshot | None = None,
        confirmed_action: PendingAction | None = None,
        pending_action_validated: bool = False,
    ) -> ToolExecution:
        tool = self._tools.get(tool_call.tool_name)
        if tool is None:
            result = ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                tool_category=tool_call.tool_category,
                status="failed",
                envelope=ToolResultEnvelope(
                    success=False,
                    status="ERROR",
                    code=404,
                    message=f"未注册工具：{tool_call.tool_name}",
                ),
                error_message=f"未注册工具：{tool_call.tool_name}",
                business_status="ERROR",
                code=404,
                message=f"未注册工具：{tool_call.tool_name}",
                terminal=True,
                user_visible_message="当前工具尚未接入，请稍后再试。",
            )
            return ToolExecution(result=result)

        if not self._is_allowed(route, tool):
            result = ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                tool_category=tool.category,
                status="blocked",
                envelope=ToolResultEnvelope(
                    success=False,
                    status="TOOL_BLOCKED",
                    code=403,
                    message=f"当前路由不允许调用工具：{tool_call.tool_name}",
                ),
                error_message=f"当前路由不允许调用工具：{tool_call.tool_name}",
                business_status="TOOL_BLOCKED",
                code=403,
                message=f"当前路由不允许调用工具：{tool_call.tool_name}",
                terminal=True,
                user_visible_message="当前请求不能执行这个工具动作。",
                model_observation="工具被注册表拦截：该工具不在当前路由允许的工具名或工具分类内。",
            )
            return ToolExecution(result=result)

        spec = tool.spec()
        if tool_call.tool_category and tool_call.tool_category != tool.category:
            result = self._blocked_result(
                tool_call,
                tool.category,
                f"工具分类不匹配：调用方={tool_call.tool_category}，注册表={tool.category}",
                "工具分类和注册表声明不一致，已拦截执行。",
            )
            return ToolExecution(result=result)

        input_errors = self._validate_slots(tool_call.arguments, spec.input_slots)
        if input_errors:
            result = self._blocked_result(
                tool_call,
                tool.category,
                f"工具输入槽位校验失败：{'; '.join(input_errors)}",
                "工具输入槽位校验失败，未执行工具。",
            )
            return ToolExecution(result=result)

        confirmation_required = self._risk_policy.requires_confirmation(
            spec.risk_level,
            spec.requires_confirmation or route.confirmation_required or tool_call.confirmation_required,
        )
        if confirmation_required and not pending_action_validated:
            action = confirmed_action or (session_state.pending_action if session_state else None)
            if session_state is None or action is None:
                result = self._blocked_result(
                    tool_call,
                    tool.category,
                    "高风险工具缺少已确认的 pending_action 快照。",
                    "ToolRegistry fail-closed：未提供确认快照，拒绝执行状态变化工具。",
                    code=403,
                )
                return ToolExecution(result=result)
            validation = self._pending_action_validator.validate_pending_action_snapshot(
                state=session_state,
                route=route,
                action=action,
                tool_call=tool_call,
                spec=spec,
            )
            if not validation.allowed:
                result = self._blocked_result(
                    tool_call,
                    tool.category,
                    f"pending_action 校验失败：{validation.code} {validation.reason}",
                    "ToolRegistry fail-closed：确认快照与当前状态不一致。",
                    code=403,
                )
                return ToolExecution(result=result)

        execution = tool.execute(request, route, tool_call)
        output_errors = self._validate_slots(execution.result.output, spec.output_slots)
        if output_errors:
            result = ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                tool_category=tool.category,
                status="failed",
                envelope=ToolResultEnvelope(
                    success=False,
                    status="TOOL_SCHEMA_ERROR",
                    code=422,
                    message=f"工具输出槽位校验失败：{'; '.join(output_errors)}",
                    data=execution.result.output,
                ),
                error_message=f"工具输出槽位校验失败：{'; '.join(output_errors)}",
                business_status="TOOL_SCHEMA_ERROR",
                code=422,
                message=f"工具输出槽位校验失败：{'; '.join(output_errors)}",
                terminal=True,
                user_visible_message="工具返回结果不完整，请稍后再试。",
                model_observation="工具输出槽位校验失败，不能基于不完整结果回答用户。",
            )
            return ToolExecution(result=result, sources=execution.sources, retrieval_trace=execution.retrieval_trace)

        return execution

    def execute_observed(
        self,
        request: ChatRequest,
        route: RouteDecision,
        tool_call: ToolCall,
        *,
        capability: CapabilityPolicy,
        session_state: SessionStateSnapshot | None = None,
        confirmed_action: PendingAction | None = None,
        pending_action_validated: bool = False,
    ) -> tuple[ToolExecution, ToolExecutionObservation]:
        """执行工具并产出结构化 observation。"""
        started = time.perf_counter()

        if tool_call.tool_name not in self._tools:
            execution = ToolExecution(result=self._unavailable_result(tool_call))
            observation = ToolExecutionObservation(
                kind="unavailable",
                tool_name=tool_call.tool_name,
                arguments=dict(tool_call.arguments),
                reason=f"{tool_call.tool_name} 未在工具注册表接入。",
                alternative_tools=[
                    name
                    for name in capability.allowed_tools
                    if name in self._tools and name != tool_call.tool_name
                ],
                hint=self._build_unavailable_hint(capability, tool_call.tool_name),
                duration_ms=self._elapsed_ms(started),
            )
            return execution, observation

        try:
            execution = self.execute(
                request,
                route,
                tool_call,
                session_state=session_state,
                confirmed_action=confirmed_action,
                pending_action_validated=pending_action_validated,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            execution = ToolExecution(result=self._failed_result(tool_call, exc))
        observation = self._build_observation_from_execution(tool_call, execution, started_at=started)
        return execution, observation

    def _is_allowed(self, route: RouteDecision, tool: AgentTool) -> bool:
        if not route.allowed_tools and not route.allowed_tool_categories:
            return True

        name_allowed = not route.allowed_tools or tool.name in route.allowed_tools
        category_allowed = (
            not route.allowed_tool_categories or tool.category in route.allowed_tool_categories
        )
        return name_allowed and category_allowed

    def _validate_slots(self, values: dict[str, object], slots: list[SlotSpec]) -> list[str]:
        errors: list[str] = []
        for slot in slots:
            value = values.get(slot.name)
            if slot.required and not self._has_value(value, allow_empty=slot.allow_empty):
                errors.append(f"{slot.name} 缺失")
                continue
            if not self._has_value(value, allow_empty=slot.allow_empty):
                continue
            if not self._matches_type(value, slot.value_type):
                errors.append(f"{slot.name} 类型应为 {slot.value_type}，实际为 {type(value).__name__}")
        return errors

    def _has_value(self, value: object, allow_empty: bool = False) -> bool:
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        if isinstance(value, list | dict | tuple | set) and not value:
            return allow_empty
        return True

    def _matches_type(self, value: object, value_type: str) -> bool:
        if value_type == "string":
            return isinstance(value, str)
        if value_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if value_type == "number":
            return isinstance(value, int | float) and not isinstance(value, bool)
        if value_type == "boolean":
            return isinstance(value, bool)
        if value_type == "object":
            return isinstance(value, dict)
        if value_type == "array":
            return isinstance(value, list)
        return True

    def _build_observation_from_execution(
        self,
        tool_call: ToolCall,
        execution: ToolExecution,
        started_at: float,
    ) -> ToolExecutionObservation:
        """把 ToolResult 翻译成 ToolExecutionObservation。"""
        result = execution.result
        duration_ms = self._elapsed_ms(started_at)

        if result.status == "blocked":
            kind: ReactObservationKind = (
                "schema_error"
                if result.error_message and "槽位校验" in result.error_message
                else "blocked"
            )
            return ToolExecutionObservation(
                kind=kind,
                tool_name=tool_call.tool_name,
                arguments=dict(tool_call.arguments),
                reason=result.error_message or result.message or "工具被风险策略阻断",
                hint=(
                    "工具入参不符合 schema，请修正 arguments。"
                    if kind == "schema_error"
                    else "工具被白名单或风险策略拦截，请重新选择 action。"
                ),
                duration_ms=duration_ms,
            )

        if result.status == "failed":
            kind = "schema_error" if result.business_status == "TOOL_SCHEMA_ERROR" else "failed"
            return ToolExecutionObservation(
                kind=kind,
                tool_name=tool_call.tool_name,
                arguments=dict(tool_call.arguments),
                reason=result.error_message or result.message or "工具执行失败",
                hint="工具失败，可重试一次或改用 ask_user / handoff。",
                duration_ms=duration_ms,
            )

        if result.business_status in {"NOT_FOUND", "PARTIAL_DATA", "EMPTY"}:
            return ToolExecutionObservation(
                kind="empty",
                tool_name=tool_call.tool_name,
                arguments=dict(tool_call.arguments),
                summary=self._summarize_output(result),
                business_status=result.business_status,
                sources_count=len(execution.sources),
                hint="结果为空或不完整，请基于现有信息回答或追问用户。",
                duration_ms=duration_ms,
            )

        return ToolExecutionObservation(
            kind="success",
            tool_name=tool_call.tool_name,
            arguments=dict(tool_call.arguments),
            summary=self._summarize_output(result),
            business_status=result.business_status,
            sources_count=len(execution.sources),
            duration_ms=duration_ms,
        )

    def _summarize_output(self, result: ToolResult) -> str:
        """生成不超过 500 字符的工具结果摘要。"""
        parts: list[str] = []
        if result.envelope and result.envelope.message:
            parts.append(result.envelope.message[:200])
        data = result.envelope.data if result.envelope else result.output
        for key in (
            "amount",
            "limit",
            "credit_amount",
            "status",
            "url",
            "application_id",
            "authorization_id",
        ):
            if key in data:
                parts.append(f"{key}={data[key]}")
        if not parts:
            parts.append(result.message or "工具已成功返回")
        return "; ".join(parts)[:500]

    def _unavailable_result(self, tool_call: ToolCall) -> ToolResult:
        """工具未注册时构造非 terminal ToolResult。"""
        message = f"工具 {tool_call.tool_name} 暂未接入"
        return ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            tool_category=tool_call.tool_category,
            status="failed",
            envelope=ToolResultEnvelope(
                success=False,
                status="TOOL_UNAVAILABLE",
                code=503,
                message=message,
            ),
            error_message=message,
            business_status="TOOL_UNAVAILABLE",
            code=503,
            message=message,
            terminal=False,
            user_visible_message="该数据查询暂未上线。",
            model_observation=message,
        )

    def _failed_result(self, tool_call: ToolCall, exc: RuntimeError | ValueError | TypeError) -> ToolResult:
        """工具抛出运行时异常时构造失败结果。"""
        message = f"工具 {tool_call.tool_name} 执行失败：{type(exc).__name__}: {exc}"
        return ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            tool_category=tool_call.tool_category,
            status="failed",
            envelope=ToolResultEnvelope(
                success=False,
                status="ERROR",
                code=500,
                message=message,
            ),
            error_message=message,
            business_status="ERROR",
            code=500,
            message=message,
            terminal=True,
            user_visible_message="工具执行失败，请稍后再试或转人工处理。",
            model_observation=message,
        )

    def _build_unavailable_hint(self, capability: CapabilityPolicy, tool_name: str) -> str:
        """根据 capability 给出硬编码提示。"""
        hints: dict[str, str] = {
            "knowledge.policy.read": (
                "知识库检索工具未接入时禁止编造政策答案，请 ask_user 让用户稍后再试或转人工。"
            ),
            "credit.limit.read": (
                "授信额度查询后端尚未接入。请明确告知用户该数据查询暂未上线，禁止编造金额。"
            ),
            "product.terms.read": (
                "产品参数查询后端尚未接入。可改用 rag_search 查询政策范围，禁止编造具体数值。"
            ),
            "authorization.link.create": (
                "授权链接工具未接入时不能生成链接，请 handoff 转人工或 ask_user 稍后再试。"
            ),
            "application.draft.create": "申请草稿后端尚未接入，请 handoff 转人工受理。",
            "application.status.read": (
                "申请状态查询工具未接入时不能编造进度，请告知用户暂未上线并建议人工咨询。"
            ),
            "smalltalk.respond": "闲聊能力不应调用工具，请改用 answer 直接回应用户。",
            "unknown.clarify": "未知意图不应调用工具，请改用 ask_user 追问用户真实需求。",
        }
        return hints[capability.capability_id] if capability.capability_id in hints else (
            f"{tool_name} 暂未接入，请改用 answer / ask_user / handoff。"
        )

    def _elapsed_ms(self, started_at: float) -> float:
        """计算从 started_at 到当前的毫秒耗时。"""
        return (time.perf_counter() - started_at) * 1000

    def _blocked_result(
        self,
        tool_call: ToolCall,
        tool_category: ToolCategory,
        error_message: str,
        model_observation: str,
        code: int = 422,
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            tool_category=tool_category,
            status="blocked",
            envelope=ToolResultEnvelope(
                success=False,
                status="TOOL_BLOCKED",
                code=code,
                message=error_message,
            ),
            error_message=error_message,
            business_status="TOOL_BLOCKED",
            code=code,
            message=error_message,
            terminal=True,
            user_visible_message="当前请求不能执行这个工具动作。",
            model_observation=model_observation,
        )


def default_tool_registry(retriever: Retriever | None = None) -> ToolRegistry:
    return ToolRegistry(
        [
            MockCreditAmountTool(),
            RagSearchTool(retriever=retriever),
        ]
    )


def build_tool_registry(
    retriever: Retriever | None = None,
    mcp_adapters: Iterable[AgentTool] | None = None,
) -> ToolRegistry:
    """组合默认工具和可选 MCP 适配器。"""
    registry = default_tool_registry(retriever)
    if mcp_adapters:
        for adapter in mcp_adapters:
            registry.register_external(adapter)
    return registry
