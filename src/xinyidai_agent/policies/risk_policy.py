from __future__ import annotations

from dataclasses import dataclass

from xinyidai_agent.protocol import RiskLevel


@dataclass(frozen=True)
class RiskDecision:
    auto_execute: bool
    requires_confirmation: bool
    requires_handoff: bool


RISK_POLICY: dict[RiskLevel, RiskDecision] = {
    "read_only": RiskDecision(
        auto_execute=True,
        requires_confirmation=False,
        requires_handoff=False,
    ),
    "link_create": RiskDecision(
        auto_execute=False,
        requires_confirmation=True,
        requires_handoff=False,
    ),
    "state_create": RiskDecision(
        auto_execute=False,
        requires_confirmation=True,
        requires_handoff=False,
    ),
    "state_update": RiskDecision(
        auto_execute=False,
        requires_confirmation=True,
        requires_handoff=True,
    ),
    "final_submit": RiskDecision(
        auto_execute=False,
        requires_confirmation=True,
        requires_handoff=True,
    ),
}


class RiskPolicy:
    def decision_for(self, risk_level: RiskLevel) -> RiskDecision:
        return RISK_POLICY[risk_level]

    def can_auto_execute(self, risk_level: RiskLevel) -> bool:
        return self.decision_for(risk_level).auto_execute

    def requires_confirmation(
        self,
        risk_level: RiskLevel,
        explicit_confirmation: bool = False,
    ) -> bool:
        return explicit_confirmation or self.decision_for(risk_level).requires_confirmation

    def requires_handoff(self, risk_level: RiskLevel) -> bool:
        return self.decision_for(risk_level).requires_handoff
