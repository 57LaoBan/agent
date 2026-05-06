from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.protocol import RuntimeStepDecision  # noqa: E402


class RuntimeStepContractTest(unittest.TestCase):
    def test_runtime_step_decision_round_trip(self) -> None:
        decision = RuntimeStepDecision(
            step_type="ASK_USER",
            reason="缺少企业名称。",
            confidence=0.82,
            question="请补充企业名称。",
            missing_slots=["company_name"],
        )

        payload = decision.model_dump()
        restored = RuntimeStepDecision.model_validate(payload)

        self.assertEqual(restored.step_type, "ASK_USER")
        self.assertEqual(restored.missing_slots, ["company_name"])

    def test_invalid_step_type_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RuntimeStepDecision.model_validate({"step_type": "FREE_REACT", "confidence": 0.5})

    def test_confidence_range_is_validated(self) -> None:
        with self.assertRaises(ValidationError):
            RuntimeStepDecision(step_type="STOP", confidence=1.2)


if __name__ == "__main__":
    unittest.main()
