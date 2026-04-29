from __future__ import annotations

from dataclasses import dataclass, field

from xinyidai_agent.protocol import RiskLevel, RouteScene, ToolCategory


@dataclass(frozen=True)
class CapabilityPolicy:
    """业务能力策略。

    capability 是系统真实允许的业务能力单元，工具权限由它派生，而不是由模型直接决定。
    """

    capability_id: str
    scene: RouteScene
    standard_intent: str
    description: str
    raw_intent_aliases: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    required_slots: list[str] = field(default_factory=list)
    optional_slots: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    allowed_tool_categories: list[ToolCategory] = field(default_factory=list)
    risk_level: RiskLevel = "read_only"
    confirmation_required: bool = False
    min_confidence: float = 0.7
