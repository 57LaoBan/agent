from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from xinyidai_agent.protocol import (
    ChatRequest,
    RetrievalTrace,
    RouteDecision,
    RiskLevel,
    SourceDocument,
    ToolCategory,
    ToolCall,
    ToolResult,
)


@dataclass(frozen=True)
class ToolExecution:
    result: ToolResult
    sources: list[SourceDocument] = field(default_factory=list)
    retrieval_trace: RetrievalTrace | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: ToolCategory
    risk_level: RiskLevel
    description: str
    requires_confirmation: bool = False
    input_schema: dict[str, object] = field(default_factory=dict)


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
        ...

    def spec(self) -> ToolSpec:
        ...
