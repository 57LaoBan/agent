from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.protocol import ChatRequest, RouteDecision  # noqa: E402
from xinyidai_agent.router import ControlledIntentRouter, RoutePolicy  # noqa: E402


class JsonRouterModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        self.messages = messages
        return self.content


class RouterContractTest(unittest.TestCase):
    def test_business_scene_is_judged_by_model(self) -> None:
        model = JsonRouterModel(
            """
            {
              "scene": "DATA_QUERY",
              "intent": "CREDIT_LIMIT_QUERY",
              "confidence": 0.91,
              "filled_slots": {"company_name": "杭州示例科技有限公司"},
              "route_reason": "模型判断用户在查询企业授信额度。"
            }
            """
        )
        router = ControlledIntentRouter(model=model)

        route = router.route(ChatRequest(user_message="杭州示例科技有限公司能贷多少钱？"))

        self.assertEqual(route.scene, "DATA_QUERY")
        self.assertEqual(route.intent, "CREDIT_LIMIT_QUERY")
        self.assertEqual(route.allowed_tool_categories, ["data_query"])
        self.assertEqual(route.allowed_tools, ["query_credit_amount"])
        self.assertEqual(route.route_source, "model")
        self.assertEqual(model.calls, 1)

    def test_low_information_input_is_clarified_without_model_or_rag(self) -> None:
        model = JsonRouterModel(
            '{"scene":"KNOWLEDGE_QA","intent":"POLICY_OR_PRODUCT_QA","confidence":0.99,"route_reason":"x"}'
        )
        router = ControlledIntentRouter(model=model)

        route = router.route(ChatRequest(user_message="123"))

        self.assertEqual(route.scene, "UNKNOWN")
        self.assertEqual(route.intent, "LOW_CONFIDENCE")
        self.assertEqual(route.missing_slots, ["user_intent"])
        self.assertEqual(route.allowed_tools, [])
        self.assertEqual(route.route_source, "local_guard")
        self.assertFalse(route.should_call_tool)
        self.assertEqual(model.calls, 0)

    def test_model_router_handles_free_form_business_intent(self) -> None:
        model = JsonRouterModel(
            """
            {
              "scene": "DATA_QUERY",
              "intent": "CREDIT_ELIGIBILITY_QUERY",
              "confidence": 0.82,
              "filled_slots": {"company_name": "杭州示例科技有限公司"},
              "risk_level": "read_only",
              "route_reason": "用户想判断企业能否获得贷款支持。",
              "should_call_model": true,
              "should_call_tool": true
            }
            """
        )
        router = ControlledIntentRouter(model=model)

        route = router.route(ChatRequest(user_message="判断杭州示例科技有限公司是否符合融资准入"))

        self.assertEqual(route.scene, "DATA_QUERY")
        self.assertEqual(route.intent, "CREDIT_LIMIT_QUERY")
        self.assertEqual(route.raw_intent, "CREDIT_ELIGIBILITY_QUERY")
        self.assertEqual(route.capability_id, "credit.limit.read")
        self.assertEqual(route.capability_source, "scene_default")
        self.assertEqual(route.filled_slots["company_name"], "杭州示例科技有限公司")
        self.assertEqual(route.allowed_tool_categories, ["data_query"])
        self.assertEqual(route.allowed_tools, ["query_credit_amount"])
        self.assertEqual(route.route_source, "model")
        self.assertTrue(route.should_call_tool)
        self.assertEqual(model.calls, 1)

    def test_model_cannot_expose_unapproved_tools(self) -> None:
        model = JsonRouterModel(
            """
            {
              "scene": "KNOWLEDGE_QA",
              "intent": "query_eligibility_criteria",
              "confidence": 0.95,
              "allowed_tools": ["get_eligibility_rules"],
              "allowed_tool_categories": ["utility"],
              "route_reason": "用户询问信易贷适用企业类型。",
              "should_call_model": true,
              "should_call_tool": true
            }
            """
        )
        router = ControlledIntentRouter(model=model)

        route = router.route(ChatRequest(user_message="信易贷适合哪些企业？"))

        self.assertEqual(route.scene, "KNOWLEDGE_QA")
        self.assertEqual(route.intent, "POLICY_OR_PRODUCT_QA")
        self.assertEqual(route.raw_intent, "query_eligibility_criteria")
        self.assertEqual(route.capability_id, "knowledge.policy.read")
        self.assertEqual(route.capability_source, "scene_default")
        self.assertEqual(route.route_source, "model")
        self.assertEqual(route.allowed_tools, ["rag_search"])
        self.assertEqual(route.allowed_tool_categories, ["knowledge"])

    def test_authorization_scene_resolves_link_create_capability(self) -> None:
        model = JsonRouterModel(
            """
            {
              "scene": "AUTHORIZATION",
              "raw_intent": "generate_company_authorization_link",
              "confidence": 0.92,
              "filled_slots": {"company_name": "杭州示例科技有限公司"},
              "route_reason": "用户想生成企业授权链接。"
            }
            """
        )
        router = ControlledIntentRouter(model=model)

        route = router.route(ChatRequest(user_message="帮我生成杭州示例科技有限公司的授权链接"))

        self.assertEqual(route.scene, "AUTHORIZATION")
        self.assertEqual(route.intent, "CREATE_AUTHORIZATION_LINK")
        self.assertEqual(route.raw_intent, "generate_company_authorization_link")
        self.assertEqual(route.capability_id, "authorization.link.create")
        self.assertTrue(route.confirmation_required)
        self.assertEqual(route.allowed_tools, ["create_authorization_link"])
        self.assertEqual(route.allowed_tool_categories, ["authorization"])

    def test_model_route_normalizes_empty_filled_slots(self) -> None:
        model = JsonRouterModel(
            """
            {
              "scene": "KNOWLEDGE_QA",
              "intent": "POLICY_OR_PRODUCT_QA",
              "confidence": 0.88,
              "filled_slots": [],
              "route_reason": "错误结构：filled_slots 被模型写成数组。",
              "should_call_model": true,
              "should_call_tool": true
            }
            """
        )
        router = ControlledIntentRouter(model=model)

        route = router.route(ChatRequest(user_message="帮我看看融资政策"))

        self.assertEqual(route.scene, "KNOWLEDGE_QA")
        self.assertEqual(route.filled_slots, {})
        self.assertEqual(route.allowed_tools, ["rag_search"])

    def test_unparseable_model_route_asks_user_instead_of_running_rag(self) -> None:
        model = JsonRouterModel("我觉得应该查知识库，但我没有输出 JSON")
        router = ControlledIntentRouter(model=model)

        route = router.route(ChatRequest(user_message="帮我看看融资政策"))

        self.assertEqual(route.scene, "UNKNOWN")
        self.assertEqual(route.intent, "LOW_CONFIDENCE")
        self.assertEqual(route.missing_slots, ["user_intent"])
        self.assertEqual(route.allowed_tools, [])
        self.assertFalse(route.should_call_tool)

    def test_policy_blocks_low_confidence_route(self) -> None:
        policy = RoutePolicy(min_confidence=0.7)
        route = RouteDecision(
            scene="LOAN_APPLY",
            intent="CREATE_APPLICATION",
            confidence=0.42,
            route_reason="模型不确定。",
            should_call_tool=True,
        )

        guarded = policy.apply(ChatRequest(user_message="随便看看"), route)

        self.assertEqual(guarded.scene, "UNKNOWN")
        self.assertEqual(guarded.intent, "LOW_CONFIDENCE")
        self.assertEqual(guarded.missing_slots, ["user_intent"])
        self.assertFalse(guarded.should_call_tool)

    def test_policy_marks_missing_slots_before_tool_execution(self) -> None:
        policy = RoutePolicy()
        route = RouteDecision(
            scene="LOAN_APPLY",
            intent="CREATE_APPLICATION",
            confidence=0.91,
            filled_slots={"product_name": "小微税贷"},
            route_reason="用户想申请贷款。",
            should_call_tool=True,
        )

        guarded = policy.apply(ChatRequest(user_message="帮我申请小微税贷"), route)

        self.assertEqual(guarded.scene, "LOAN_APPLY")
        self.assertEqual(guarded.missing_slots, ["company_name"])
        self.assertEqual(guarded.allowed_tool_categories, ["knowledge", "application", "authorization"])

    def test_policy_ignores_model_invented_missing_slots(self) -> None:
        policy = RoutePolicy()
        route = RouteDecision(
            scene="LOAN_APPLY",
            intent="CREATE_APPLICATION",
            confidence=0.91,
            filled_slots={"product_name": "小微税贷"},
            missing_slots=["company_name", "loan_purpose", "monthly_income", "credit_score"],
            route_reason="模型发明了后端能力策略没有声明的必填字段。",
            should_call_tool=True,
        )

        guarded = policy.apply(ChatRequest(user_message="我要申请小微税贷"), route)

        self.assertEqual(guarded.scene, "LOAN_APPLY")
        self.assertEqual(guarded.required_slots, ["company_name", "product_name"])
        self.assertEqual(guarded.missing_slots, ["company_name"])


if __name__ == "__main__":
    unittest.main()
