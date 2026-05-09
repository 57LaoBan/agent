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
    required_slots: list[str] = field(default_factory=list)
    optional_slots: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    allowed_tool_categories: list[ToolCategory] = field(default_factory=list)
    risk_level: RiskLevel = "read_only"
    confirmation_required: bool = False
    min_confidence: float = 0.7
    fast_path_eligible: bool = False
    """单工具只读快速通道开关；默认关闭，必须由 catalog 显式放行。"""

    requires_evidence: bool = False
    """回答是否必须基于工具 sources；知识问答类能力必须显式开启。"""

    max_react_steps: int = 4
    """该 capability 单次 ReAct 循环允许的最大步数。"""

    def is_fast_path_eligible_for_route(self, route_filled_slots: dict[str, object]) -> bool:
        """判断当前 route 是否满足单工具只读快速通道条件。"""
        if not self.fast_path_eligible:
            return False
        if len(self.allowed_tools) != 1:
            return False
        if self.risk_level != "read_only":
            return False
        for slot in self.required_slots:
            value = route_filled_slots.get(slot)
            if value is None or (isinstance(value, str) and not value.strip()):
                return False
        return True
