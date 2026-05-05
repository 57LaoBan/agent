from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.memory import MemoryManager  # noqa: E402
from xinyidai_agent.protocol import ChatRequest, RouteDecision, ToolResult  # noqa: E402


class ShortTermMemoryTest(unittest.TestCase):
    def test_session_state_records_stage_and_recent_turn(self) -> None:
        memory = MemoryManager(max_recent_turns=2)
        state = memory.load("session-1")
        route = RouteDecision(
            scene="DATA_QUERY",
            intent="CREDIT_LIMIT_QUERY",
            capability_id="credit.limit.read",
            confidence=0.9,
            required_slots=["company_name"],
            filled_slots={"company_name": "杭州示例科技有限公司"},
            route_reason="查询额度。",
        )
        result = ToolResult(
            tool_call_id="tool-1",
            tool_name="query_credit_amount",
            tool_category="data_query",
            status="success",
            business_status="CREDIT_AMOUNT_FOUND",
            output={
                "company_name": "杭州示例科技有限公司",
                "credit_amount": "50万元",
                "data_time": "2026-04-28",
            },
        )

        updated = memory.update(
            state,
            ChatRequest(user_message="杭州示例科技有限公司能贷多少？", session_id="session-1"),
            route=route,
            tool_result=result,
            stop_reason="completed",
            final_answer="该企业当前授信额度为 50万元。",
        )

        self.assertEqual(updated.current_stage, "DATA_QUERY.answered")
        self.assertEqual(updated.recent_turns[-1]["business_status"], "CREDIT_AMOUNT_FOUND")
        self.assertEqual(updated.confirmed_slots["company_name"], "杭州示例科技有限公司")
        self.assertIn("stage=DATA_QUERY.answered", updated.short_summary)

    def test_recent_turns_are_windowed_without_losing_slots(self) -> None:
        memory = MemoryManager(max_recent_turns=2)
        state = memory.load("session-1")
        route = RouteDecision(
            scene="DATA_QUERY",
            intent="CREDIT_LIMIT_QUERY",
            capability_id="credit.limit.read",
            confidence=0.9,
            filled_slots={"company_name": "杭州示例科技有限公司"},
            route_reason="查询额度。",
        )

        for index in range(3):
            state = memory.update(
                state,
                ChatRequest(user_message=f"第 {index} 轮", session_id="session-1"),
                route=route,
                stop_reason="answered_without_tool",
                final_answer="测试回答",
            )

        self.assertEqual(len(state.recent_turns), 2)
        self.assertEqual(state.confirmed_slots["company_name"], "杭州示例科技有限公司")


if __name__ == "__main__":
    unittest.main()
