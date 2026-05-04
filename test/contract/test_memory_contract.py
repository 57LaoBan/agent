from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.protocol import ChatRequest, RouteDecision  # noqa: E402
from xinyidai_agent.runtime import ControlledAgentLoop  # noqa: E402
from xinyidai_agent.tools.mock_credit import MockCreditAmountTool  # noqa: E402
from xinyidai_agent.tools.registry import ToolRegistry  # noqa: E402


class FakeModel:
    def complete(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | None = None,
    ) -> str:
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        if "update_session_state" in system:
            if "重庆征信" in user:
                return """
                {
                  "tool_name": "update_session_state",
                  "arguments": {
                    "operations": [
                      {
                        "op": "set_slot",
                        "slot": "company_name",
                        "value": "重庆征信",
                        "source": "user_explicit",
                        "confidence": 0.96,
                        "reason": "用户明确表示换一家企业"
                      },
                      {
                        "op": "clear_result",
                        "target": "last_credit_amount",
                        "source": "user_explicit",
                        "confidence": 0.95,
                        "reason": "企业主体切换，上一家企业额度结果失效"
                      }
                    ]
                  }
                }
                """
            if "杭州示例科技有限公司" in user:
                return """
                {
                  "tool_name": "update_session_state",
                  "arguments": {
                    "operations": [
                      {
                        "op": "set_slot",
                        "slot": "company_name",
                        "value": "杭州示例科技有限公司",
                        "source": "user_explicit",
                        "confidence": 0.95,
                        "reason": "用户补充了企业名称"
                      }
                    ]
                  }
                }
                """
            return '{"tool_name":"update_session_state","arguments":{"operations":[]}}'

        if "意图识别器" in system:
            if "重庆征信" in user:
                return """
                {
                  "scene": "DATA_QUERY",
                  "intent": "CREDIT_LIMIT_QUERY",
                  "confidence": 0.92,
                  "filled_slots": {"company_name": "重庆征信"},
                  "route_reason": "测试模型根据用户输入判断为额度查询。"
                }
                """
            if "杭州示例科技有限公司" in user:
                return """
                {
                  "scene": "DATA_QUERY",
                  "intent": "CREDIT_LIMIT_QUERY",
                  "confidence": 0.92,
                  "filled_slots": {"company_name": "杭州示例科技有限公司"},
                  "route_reason": "测试模型根据用户输入判断为额度查询。"
                }
                """
            return """
            {
              "scene": "DATA_QUERY",
              "intent": "CREDIT_LIMIT_QUERY",
              "confidence": 0.9,
              "route_reason": "测试模型识别为额度查询。"
            }
            """

        if "50万元" in user:
            return "已根据工具结果生成回答：50万元。"
        return "已根据工具结果生成回答。"


class SlotMemoryRouter:
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def route(self, request: ChatRequest) -> RouteDecision:
        self.requests.append(request)
        company_name = request.metadata.get("company_name")
        if company_name:
            return RouteDecision(
                scene="DATA_QUERY",
                intent="CREDIT_LIMIT_QUERY",
                raw_intent="CREDIT_LIMIT_QUERY",
                confidence=0.9,
                required_slots=["company_name"],
                filled_slots={"company_name": company_name},
                allowed_tools=["query_credit_amount"],
                allowed_tool_categories=["data_query"],
                risk_level="read_only",
                route_reason="测试路由从会话记忆补齐企业名称。",
                route_source="test_router",
                should_call_tool=True,
            )

        return RouteDecision(
            scene="DATA_QUERY",
            intent="CREDIT_LIMIT_QUERY",
            raw_intent="CREDIT_LIMIT_QUERY",
            confidence=0.9,
            required_slots=["company_name"],
            missing_slots=["company_name"],
            allowed_tools=["query_credit_amount"],
            allowed_tool_categories=["data_query"],
            risk_level="read_only",
            route_reason="测试路由要求补齐企业名称。",
            route_source="test_router",
            should_call_tool=True,
        )


class MemoryContractTest(unittest.TestCase):
    def test_session_state_resumes_missing_slot_on_next_turn(self) -> None:
        router = SlotMemoryRouter()
        loop = ControlledAgentLoop(
            model=FakeModel(),
            router=router,
            tool_registry=ToolRegistry([MockCreditAmountTool()]),
        )

        first = loop.answer(ChatRequest(user_message="我能贷多少钱？", session_id="session-1"))

        self.assertEqual(first.stop_reason, "missing_slots")
        self.assertIsNotNone(first.session_state)
        self.assertEqual(first.session_state.session_id, "session-1")
        self.assertEqual(first.session_state.awaiting_slots, ["company_name"])

        second = loop.answer(ChatRequest(user_message="杭州示例科技有限公司", session_id="session-1"))

        self.assertEqual(second.stop_reason, "completed")
        self.assertIsNotNone(second.session_state)
        self.assertEqual(second.route_decision.scene, "DATA_QUERY")
        self.assertEqual(second.tool_trace[0].tool_name, "query_credit_amount")
        self.assertEqual(second.session_state.awaiting_slots, [])
        self.assertEqual(second.session_state.confirmed_slots["company_name"], "杭州示例科技有限公司")
        self.assertEqual(second.session_state.last_credit_amount["credit_amount"], "50万元")
        self.assertEqual(router.requests[-1].metadata["company_name"], "杭州示例科技有限公司")

    def test_model_visible_system_tool_overrides_previous_company(self) -> None:
        loop = ControlledAgentLoop(
            model=FakeModel(),
            tool_registry=ToolRegistry([MockCreditAmountTool()]),
        )

        first = loop.answer(ChatRequest(user_message="杭州示例科技有限公司能贷多少钱？", session_id="session-2"))
        self.assertEqual(first.stop_reason, "completed")
        self.assertEqual(first.tool_trace[-1].output["company_name"], "杭州示例科技有限公司")

        second = loop.answer(ChatRequest(user_message="换一家企业，重庆征信，我能贷多少钱", session_id="session-2"))

        self.assertEqual(second.stop_reason, "completed")
        self.assertEqual(second.route_decision.scene, "DATA_QUERY")
        self.assertEqual(second.route_decision.filled_slots["company_name"], "重庆征信")
        self.assertEqual(second.tool_trace[-1].output["company_name"], "重庆征信")
        self.assertEqual(second.session_state.confirmed_slots["company_name"], "重庆征信")
        self.assertEqual(second.session_state.last_credit_amount["company_name"], "重庆征信")
        self.assertTrue(
            any(event.event_type == "system_tool_result" for event in second.events),
            "切换企业应由模型调用系统工具更新 session",
        )


if __name__ == "__main__":
    unittest.main()
