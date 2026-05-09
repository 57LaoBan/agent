"""根据 runtime 当前状态构造 ReactObservation 喂给模型。"""

from __future__ import annotations

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.protocol import (
    ChatRequest,
    RouteDecision,
    SessionStateSnapshot,
    ToolExecutionObservation,
)
from xinyidai_agent.react_observation import ReactObservation, ToolSpecBrief
from xinyidai_agent.tools.registry import ToolRegistry


class ReactObservationBuilder:
    """ReactObservation 构造器。"""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        """初始化工具注册表依赖。"""
        self._tools = tool_registry

    def build(
        self,
        *,
        step: int,
        max_steps: int,
        request: ChatRequest,
        route: RouteDecision,
        capability: CapabilityPolicy,
        session_state: SessionStateSnapshot,
        executed_tools: list[ToolExecutionObservation],
        last_observation: ToolExecutionObservation | None,
        invalid_action_hint: str | None = None,
    ) -> ReactObservation:
        """构造模型单步推理所需的完整 observation。"""
        return ReactObservation(
            step=step,
            max_steps=max_steps,
            request=request,
            route=route,
            capability=capability,
            session_state=session_state,
            available_tools=self._render_tool_briefs(capability, executed_tools),
            executed_tools=executed_tools,
            last_observation=last_observation,
            invalid_action_hint=invalid_action_hint,
        )

    def _render_tool_briefs(
        self,
        capability: CapabilityPolicy,
        executed_tools: list[ToolExecutionObservation],
    ) -> list[ToolSpecBrief]:
        """仅暴露当前 capability.allowed_tools 中声明的工具。"""
        briefs: list[ToolSpecBrief] = []
        for tool_name in capability.allowed_tools:
            spec = self._tools.spec_for_name(tool_name)
            executed_count = sum(1 for observation in executed_tools if observation.tool_name == tool_name)
            if spec is None:
                briefs.append(
                    ToolSpecBrief(
                        name=tool_name,
                        description=f"（{tool_name} 当前未在工具注册表接入，调用会返回 unavailable）",
                        risk_level="read_only",
                        is_read_only=True,
                        input_schema={},
                        required_arguments=[],
                        already_executed_count=executed_count,
                    )
                )
                continue
            briefs.append(
                ToolSpecBrief(
                    name=spec.name,
                    description=spec.description,
                    risk_level=spec.risk_level,
                    is_read_only=spec.is_read_only,
                    input_schema=dict(spec.input_schema),
                    required_arguments=spec.required_argument_names(),
                    already_executed_count=executed_count,
                )
            )
        return briefs
