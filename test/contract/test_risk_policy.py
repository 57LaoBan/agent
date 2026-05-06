from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.policies.risk_policy import RiskPolicy  # noqa: E402
from xinyidai_agent.protocol import RouteDecision  # noqa: E402


class RiskPolicyTest(unittest.TestCase):
    def test_final_submit_requires_confirmation_and_handoff(self) -> None:
        decision = RiskPolicy().decision_for("final_submit")

        self.assertFalse(decision.auto_execute)
        self.assertTrue(decision.requires_confirmation)
        self.assertTrue(decision.requires_handoff)

    def test_read_only_is_the_only_auto_executable_risk(self) -> None:
        policy = RiskPolicy()

        self.assertTrue(policy.can_auto_execute("read_only"))
        for risk_level in ["link_create", "state_create", "state_update", "final_submit"]:
            with self.subTest(risk_level=risk_level):
                self.assertFalse(policy.can_auto_execute(risk_level))

    def test_route_confirmation_is_derived_from_policy_not_model_claim(self) -> None:
        route = RouteDecision(
            scene="LOAN_APPLY",
            intent="SUBMIT_APPLICATION",
            confidence=0.9,
            route_reason="submit",
            risk_level="final_submit",
            confirmation_required=False,
        )

        self.assertTrue(RiskPolicy().requires_confirmation(route.risk_level, route.confirmation_required))


if __name__ == "__main__":
    unittest.main()
