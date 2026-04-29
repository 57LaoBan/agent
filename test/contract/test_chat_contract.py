from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.protocol import ChatRequest, RouteDecision, SourceDocument, ToolCall  # noqa: E402
from xinyidai_agent.runtime import ControlledAgentLoop  # noqa: E402
from xinyidai_agent.tools.mock_credit import MockCreditAmountTool  # noqa: E402
from xinyidai_agent.tools.rag_search import RagSearchTool  # noqa: E402
from xinyidai_agent.tools.registry import ToolRegistry  # noqa: E402


class FakeModel:
    def complete(self, messages: list[dict[str, str]]) -> str:
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
        self.assertEqual(payload["route_decision"]["scene"], "KNOWLEDGE_QA")
        self.assertEqual(payload["route_decision"]["allowed_tool_categories"], ["knowledge"])
        self.assertEqual(payload["diagnostics"][0]["detail"]["mode"], "single_agent_controlled_tool_loop")

    def test_application_request_returns_pending_action_before_tool_execution(self) -> None:
        loop = build_test_loop()
        response = loop.answer(ChatRequest(user_message="我要申请小微税贷"))

        self.assertEqual(response.answer, "")
        self.assertIsNotNone(response.pending_action)
        self.assertEqual(response.pending_action.tool_call.tool_name, "create_application")
        self.assertEqual(response.pending_action.tool_call.tool_category, "application")
        self.assertTrue(response.pending_action.tool_call.confirmation_required)
        self.assertEqual(response.route_decision.scene, "LOAN_APPLY")

    def test_run_emits_ordered_agent_events(self) -> None:
        loop = build_test_loop()
        events = list(loop.run(ChatRequest(user_message="我能贷多少钱？")))

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


if __name__ == "__main__":
    unittest.main()
