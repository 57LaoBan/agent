from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.protocol import ChatRequest, SourceDocument  # noqa: E402
from xinyidai_agent.runtime import ControlledAgentLoop  # noqa: E402


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


class ChatContractTest(unittest.TestCase):
    def test_response_contains_answer_sources_and_trace(self) -> None:
        loop = ControlledAgentLoop(model=FakeModel(), retriever=StaticRetriever())
        response = loop.answer(ChatRequest(user_message="信易贷适合哪些企业？", top_k=3))
        payload = response.to_dict()

        self.assertEqual(payload["answer"], "这是一个测试回答。")
        self.assertEqual(payload["sources"][0]["source_id"], "policy-1")
        self.assertEqual(payload["retrieval_trace"]["top_k"], 3)
        self.assertEqual(payload["tool_trace"][0]["tool_name"], "rag_search")
        self.assertEqual(payload["route_decision"]["scene"], "KNOWLEDGE_QA")
        self.assertEqual(payload["diagnostics"][0]["detail"]["mode"], "single_agent_controlled_tool_loop")

    def test_application_request_returns_pending_action_before_tool_execution(self) -> None:
        loop = ControlledAgentLoop(model=FakeModel(), retriever=StaticRetriever())
        response = loop.answer(ChatRequest(user_message="我要申请小微税贷"))

        self.assertEqual(response.answer, "")
        self.assertIsNotNone(response.pending_action)
        self.assertEqual(response.pending_action.tool_call.tool_name, "create_application")
        self.assertTrue(response.pending_action.tool_call.confirmation_required)
        self.assertEqual(response.route_decision.scene, "LOAN_APPLY")

    def test_run_emits_ordered_agent_events(self) -> None:
        loop = ControlledAgentLoop(model=FakeModel(), retriever=StaticRetriever())
        events = list(loop.run(ChatRequest(user_message="我能贷多少钱？")))

        self.assertEqual(events[0].event_type, "turn_started")
        self.assertEqual([event.sequence for event in events], list(range(1, len(events) + 1)))
        self.assertIn("route_decision", [event.event_type for event in events])
        self.assertIn("tool_result", [event.event_type for event in events])
        self.assertEqual(events[-1].event_type, "turn_finished")


if __name__ == "__main__":
    unittest.main()
