from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from xinyidai_agent.protocol import (
    ChatRequest,
    RetrievalTrace,
    RouteDecision,
    RiskLevel,
    SlotName,
    SourceDocument,
    ToolCategory,
    ToolCall,
    ToolResult,
)

SlotValueType = str


@dataclass(frozen=True)
class ToolExecution:
    result: ToolResult
    sources: list[SourceDocument] = field(default_factory=list)
    retrieval_trace: RetrievalTrace | None = None


@dataclass(frozen=True)
class SlotSpec:
    name: SlotName
    value_type: SlotValueType
    required: bool = True
    description: str = ""
    allow_empty: bool = False


@dataclass(frozen=True)
class ToolSpec:
    """工具能力说明，默认按 fail-closed 策略声明运行属性。"""

    name: str
    category: ToolCategory
    risk_level: RiskLevel
    description: str
    requires_confirmation: bool = False
    input_slots: list[SlotSpec] = field(default_factory=list)
    output_slots: list[SlotSpec] = field(default_factory=list)
    input_schema: dict[str, object] = field(default_factory=dict)
    is_read_only: bool = False
    """是否只读；默认 False，工具必须显式声明才能进入只读快速通道。"""

    is_idempotent: bool = False
    """同名同参重复调用是否幂等；默认 False，避免误重试状态变化工具。"""

    is_concurrency_safe: bool = False
    """是否可并发执行；默认 False，涉及共享状态的工具不能并发。"""

    cost_class: Literal["cheap", "medium", "expensive"] = "medium"
    """单次调用成本档位，供调度器与重试策略参考。"""

    max_duration_ms: int = 5000
    """单次调用硬超时，超出由 runtime 强制截断。"""

    def required_argument_names(self) -> list[str]:
        """返回必填入参名称列表。"""
        return [slot.name for slot in self.input_slots if slot.required]


class AgentTool(Protocol):
    name: str
    category: ToolCategory
    risk_level: RiskLevel
    description: str
    requires_confirmation: bool

    def execute(
        self,
        request: ChatRequest,
        route: RouteDecision,
        tool_call: ToolCall,
    ) -> ToolExecution:
        """执行工具并返回统一结果。"""
        raise NotImplementedError

    def spec(self) -> ToolSpec:
        """返回工具规格说明。"""
        raise NotImplementedError
