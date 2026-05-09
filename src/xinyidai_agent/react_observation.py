"""ReAct 主循环喂给模型的 observation 容器。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.protocol import (
    ChatRequest,
    RouteDecision,
    SessionStateSnapshot,
    ToolExecutionObservation,
)


class ToolSpecBrief(BaseModel):
    """喂给模型的工具说明裁剪版。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    risk_level: str
    is_read_only: bool
    input_schema: dict[str, Any] = Field(default_factory=dict)
    required_arguments: list[str] = Field(default_factory=list)
    already_executed_count: int = 0


class ReactObservation(BaseModel):
    """每轮喂给模型的完整观察。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int
    max_steps: int
    request: ChatRequest
    route: RouteDecision
    capability: CapabilityPolicy
    session_state: SessionStateSnapshot
    available_tools: list[ToolSpecBrief]
    executed_tools: list[ToolExecutionObservation] = Field(default_factory=list)
    last_observation: ToolExecutionObservation | None = None
    invalid_action_hint: str | None = None
