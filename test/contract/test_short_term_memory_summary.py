from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.memory import MemoryManager  # noqa: E402
from xinyidai_agent.protocol import ChatRequest, RouteDecision  # noqa: E402


class ShortTermMemorySummaryTest(unittest.TestCase):
    def test_old_recent_turns_are_merged_into_short_summary(self) -> None:
        memory = MemoryManager(max_recent_turns=2)
        state = memory.load("session-1")
        route = RouteDecision(
            scene="LOAN_APPLY",
            intent="CREATE_APPLICATION",
            capability_id="application.draft.create",
            confidence=0.9,
            filled_slots={"company_name": "杭州示例科技有限公司", "product_name": "小微税贷"},
            route_reason="申请贷款。",
        )

        for index in range(3):
            state = memory.update(
                state,
                ChatRequest(user_message=f"第 {index} 轮", session_id="session-1"),
                route=route,
                stop_reason="answered_without_tool",
                final_answer=f"第 {index} 轮回答",
            )

        self.assertEqual(len(state.recent_turns), 2)
        self.assertIn("历史业务摘要", state.short_summary)
        self.assertEqual(state.confirmed_slots["company_name"], "杭州示例科技有限公司")
        self.assertEqual(state.confirmed_slots["product_name"], "小微税贷")


if __name__ == "__main__":
    unittest.main()
