from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "test" / "fixtures" / "business_dialogues"
sys.path.insert(0, str(ROOT / "src"))


class BusinessDialogueReplayTest(unittest.TestCase):
    def test_required_dialogue_fixtures_exist_and_are_structured(self) -> None:
        required = [
            "credit_amount_missing_company.json",
            "credit_amount_success.json",
            "loan_apply_pending_create.json",
            "loan_apply_switch_company_invalidates_pending.json",
            "knowledge_qa_no_evidence.json",
        ]
        for name in required:
            path = FIXTURE_DIR / name
            self.assertTrue(path.exists(), f"缺少业务回放 fixture: {name}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("case_id", payload)
            self.assertIsInstance(payload.get("turns"), list)
            self.assertGreater(len(payload["turns"]), 0)


if __name__ == "__main__":
    unittest.main()
