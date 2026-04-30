from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.protocol import ChatRequest, RouteDecision, SourceDocument, ToolCall  # noqa: E402
from xinyidai_agent.runtime import ControlledAgentLoop  # noqa: E402
from xinyidai_agent.tools.base import SlotSpec, ToolExecution, ToolSpec  # noqa: E402
from xinyidai_agent.tools.mock_credit import MockCreditAmountTool  # noqa: E402
from xinyidai_agent.tools.rag_search import RagSearchTool  # noqa: E402
from xinyidai_agent.tools.registry import ToolRegistry, default_tool_registry  # noqa: E402


class FakeModel:
    def complete(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"] if messages else ""
        if "意图识别器" in system:
            return """
            {
              "scene": "KNOWLEDGE_QA",
              "intent": "POLICY_OR_PRODUCT_QA",
              "confidence": 0.82,
              "route_reason": "测试模型识别为知识问答。",
              "should_call_model": true,
              "should_call_tool": true
            }
            """
        self.messages = messages
        return "这是一个测试回答。"


class StaticRetriever:
    def retrieve(self, query: str, top_k: int):
        from xinyidai_agent.protocol import RetrievalTrace

        sources = [
            SourceDocument(
                source_id="policy-1",
                title="测试政策",
                content="小微企业可申请信贷支持。",
                source_type="policy",
                score=0.9,
            )
        ]
        trace = RetrievalTrace(
            query=query,
            top_k=top_k,
            results_count=len(sources),
            rerank_applied=False,
            steps=[{"name": "static_retriever"}],
        )
        return sources, trace


class BrokenCreditAmountTool:
    name = "query_credit_amount"
    category = "data_query"
    risk_level = "read_only"
    description = "测试用坏工具，故意缺少输出字段。"
    requires_confirmation = False

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            category=self.category,
            risk_level=self.risk_level,
            description=self.description,
            input_slots=[
                SlotSpec("company_name", "string"),
                SlotSpec("query", "string"),
            ],
            output_slots=[
                SlotSpec("company_name", "string"),
                SlotSpec("credit_amount", "string"),
                SlotSpec("data_time", "string"),
            ],
        )

    def execute(self, request: ChatRequest, route: RouteDecision, tool_call: ToolCall) -> ToolExecution:
        from xinyidai_agent.protocol import ToolResult

        return ToolExecution(
            result=ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                tool_category=self.category,
                status="success",
                output={"company_name": "杭州示例科技有限公司"},
                terminal=True,
            )
        )


def build_test_loop() -> ControlledAgentLoop:
    registry = ToolRegistry(
        [
            MockCreditAmountTool(),
            RagSearchTool(retriever=StaticRetriever()),
        ]
    )
    return ControlledAgentLoop(model=FakeModel(), tool_registry=registry)


class ChatContractTest(unittest.TestCase):
    def test_response_contains_answer_sources_and_trace(self) -> None:
        loop = build_test_loop()
        response = loop.answer(ChatRequest(user_message="信易贷适合哪些企业？", top_k=3))
        payload = response.to_dict()

        self.assertEqual(payload["answer"], "这是一个测试回答。")
        self.assertEqual(payload["sources"][0]["source_id"], "policy-1")
        self.assertEqual(payload["retrieval_trace"]["top_k"], 3)
        self.assertEqual(payload["tool_trace"][0]["tool_name"], "rag_search")
        self.assertEqual(payload["tool_trace"][0]["tool_category"], "knowledge")
        self.assertEqual(payload["tool_trace"][0]["envelope"]["status"], "RAG_RESULT_READY")
        self.assertEqual(payload["route_decision"]["scene"], "KNOWLEDGE_QA")
        self.assertEqual(payload["route_decision"]["allowed_tool_categories"], ["knowledge"])
        self.assertEqual(payload["business_status"], "RAG_RESULT_READY")
        self.assertTrue(payload["evidence"]["ready"])
        self.assertEqual(payload["model_decision"]["next_action"], "finish")
        self.assertGreaterEqual(payload["performance"]["total_ms"], 0)
        self.assertEqual(payload["protocol_version"], "2026-04-30")
        self.assertEqual(payload["diagnostics"][0]["detail"]["mode"], "single_agent_controlled_tool_loop")

    def test_application_request_returns_pending_action_before_tool_execution(self) -> None:
        loop = build_test_loop()
        response = loop.answer(ChatRequest(user_message="我要申请小微税贷"))

        self.assertIn("企业名称", response.answer)
        self.assertIsNone(response.pending_action)
        self.assertEqual(response.route_decision.scene, "LOAN_APPLY")
        self.assertEqual(response.route_decision.missing_slots, ["company_name"])
        self.assertEqual(response.business_status, "NO_TOOL_USED")
        self.assertEqual(response.model_decision.next_action, "ask_user")

    def test_run_emits_ordered_agent_events(self) -> None:
        loop = build_test_loop()
        events = list(
            loop.run(
                ChatRequest(
                    user_message="我能贷多少钱？",
                    metadata={"company_name": "杭州示例科技有限公司"},
                )
            )
        )

        self.assertEqual(events[0].event_type, "turn_started")
        self.assertEqual([event.sequence for event in events], list(range(1, len(events) + 1)))
        self.assertIn("route_decision", [event.event_type for event in events])
        self.assertIn("tool_result", [event.event_type for event in events])
        self.assertEqual(events[-1].event_type, "turn_finished")

    def test_tool_registry_exposes_specs_by_category(self) -> None:
        registry = ToolRegistry([MockCreditAmountTool(), RagSearchTool(retriever=StaticRetriever())])
        specs = registry.specs_for_categories(["knowledge"])

        self.assertEqual([spec.name for spec in specs], ["rag_search"])
        self.assertEqual(specs[0].category, "knowledge")
        self.assertEqual(specs[0].risk_level, "read_only")

    def test_tool_registry_blocks_disallowed_category(self) -> None:
        registry = ToolRegistry([MockCreditAmountTool(), RagSearchTool(retriever=StaticRetriever())])
        route = RouteDecision(
            scene="KNOWLEDGE_QA",
            intent="POLICY_OR_PRODUCT_QA",
            confidence=0.8,
            allowed_tool_categories=["knowledge"],
            route_reason="测试只允许知识类工具。",
            should_call_tool=True,
        )
        tool_call = ToolCall(
            tool_call_id="tool-1",
            tool_name="query_credit_amount",
            tool_category="data_query",
        )

        execution = registry.execute(ChatRequest(user_message="测试"), route, tool_call)

        self.assertEqual(execution.result.status, "blocked")
        self.assertEqual(execution.result.tool_category, "data_query")

    def test_tool_registry_blocks_missing_required_input_slots(self) -> None:
        registry = ToolRegistry([MockCreditAmountTool()])
        route = RouteDecision(
            scene="DATA_QUERY",
            intent="CREDIT_LIMIT_QUERY",
            confidence=0.9,
            allowed_tool_categories=["data_query"],
            route_reason="测试数值查询。",
            should_call_tool=True,
        )
        tool_call = ToolCall(
            tool_call_id="tool-1",
            tool_name="query_credit_amount",
            tool_category="data_query",
            arguments={"query": "能贷多少钱？"},
        )

        execution = registry.execute(ChatRequest(user_message="能贷多少钱？"), route, tool_call)

        self.assertEqual(execution.result.status, "blocked")
        self.assertIn("company_name", execution.result.error_message)

    def test_tool_registry_blocks_wrong_input_slot_type(self) -> None:
        registry = ToolRegistry([MockCreditAmountTool()])
        route = RouteDecision(
            scene="DATA_QUERY",
            intent="CREDIT_LIMIT_QUERY",
            confidence=0.9,
            allowed_tool_categories=["data_query"],
            route_reason="测试数值查询。",
            should_call_tool=True,
        )
        tool_call = ToolCall(
            tool_call_id="tool-1",
            tool_name="query_credit_amount",
            tool_category="data_query",
            arguments={"company_name": 123, "query": "能贷多少钱？"},
        )

        execution = registry.execute(ChatRequest(user_message="能贷多少钱？"), route, tool_call)

        self.assertEqual(execution.result.status, "blocked")
        self.assertIn("company_name 类型应为 string", execution.result.error_message)

    def test_tool_registry_fails_missing_required_output_slots(self) -> None:
        registry = ToolRegistry([BrokenCreditAmountTool()])
        route = RouteDecision(
            scene="DATA_QUERY",
            intent="CREDIT_LIMIT_QUERY",
            confidence=0.9,
            allowed_tool_categories=["data_query"],
            route_reason="测试数值查询。",
            should_call_tool=True,
        )
        tool_call = ToolCall(
            tool_call_id="tool-1",
            tool_name="query_credit_amount",
            tool_category="data_query",
            arguments={"company_name": "杭州示例科技有限公司", "query": "能贷多少钱？"},
        )

        execution = registry.execute(ChatRequest(user_message="能贷多少钱？"), route, tool_call)

        self.assertEqual(execution.result.status, "failed")
        self.assertIn("credit_amount", execution.result.error_message)
        self.assertIn("data_time", execution.result.error_message)

    def test_empty_rag_result_is_partial_data_not_schema_error(self) -> None:
        loop = ControlledAgentLoop(model=FakeModel(), tool_registry=default_tool_registry())
        response = loop.answer(ChatRequest(user_message="信易贷适合哪些企业？"))

        self.assertEqual(response.route_decision.scene, "KNOWLEDGE_QA")
        self.assertEqual(response.tool_trace[0].tool_name, "rag_search")
        self.assertEqual(response.tool_trace[0].status, "success")
        self.assertEqual(response.business_status, "PARTIAL_DATA")
        self.assertEqual(response.sources, [])
        self.assertIn("当前知识库还没有检索到可用证据", response.answer)


if __name__ == "__main__":
    unittest.main()
