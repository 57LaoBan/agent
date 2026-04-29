from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "e2e_cases.json"
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.protocol import ChatRequest, RetrievalTrace, SourceDocument  # noqa: E402
from xinyidai_agent.runtime import ControlledAgentLoop  # noqa: E402
from xinyidai_agent.tools.mock_credit import MockCreditAmountTool  # noqa: E402
from xinyidai_agent.tools.rag_search import RagSearchTool  # noqa: E402
from xinyidai_agent.tools.registry import ToolRegistry  # noqa: E402


class ScriptedModel:
    """确定性模型桩：路由阶段返回预设 JSON，回答阶段返回可断言文本。"""

    def __init__(self, route_payload: dict[str, Any] | None) -> None:
        self._route_payload = route_payload
        self.route_calls = 0
        self.answer_calls = 0

    def complete(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"] if messages else ""
        if "意图识别器" in system:
            self.route_calls += 1
            return json.dumps(self._route_payload or {}, ensure_ascii=False)

        self.answer_calls += 1
        body = messages[-1]["content"] if messages else ""
        if "50万元" in body:
            return "根据 mock 授信额度工具查询，杭州示例科技有限公司当前授信额度为 50万元。"
        if "小微企业" in body:
            return "根据 RAG 检索证据，信易贷主要适合依法经营、信用记录良好、有融资需求的小微企业。"
        return "我是信易贷 Agent，可以帮你查询政策、查询额度，也可以引导贷款申请流程。"


class StaticRetriever:
    def retrieve(self, query: str, top_k: int):
        sources = [
            SourceDocument(
                source_id="policy-1",
                title="信易贷适用对象测试政策",
                content="信易贷主要面向依法经营、信用记录良好、有融资需求的小微企业和个体工商户。",
                source_type="policy",
                score=0.91,
            )
        ]
        trace = RetrievalTrace(
            query=query,
            top_k=top_k,
            results_count=len(sources),
            rerank_applied=False,
            steps=[{"name": "static_e2e_retriever"}],
        )
        return sources, trace


class ControlledRouteE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        cls.cases = payload["controlled_route_cases"]

    def test_controlled_route_matrix(self) -> None:
        for case in self.cases:
            with self.subTest(case=case["id"]):
                model = ScriptedModel(case.get("model_route"))
                registry = ToolRegistry(
                    [
                        MockCreditAmountTool(),
                        RagSearchTool(retriever=StaticRetriever()),
                    ]
                )
                loop = ControlledAgentLoop(model=model, tool_registry=registry)
                response = loop.answer(
                    ChatRequest(
                        user_message=case["message"],
                        top_k=case.get("top_k", 5),
                        metadata=case.get("metadata", {}),
                    )
                )

                self.assertEqual(model.route_calls, case["expected_model_route_calls"])
                self.assertIsNotNone(response.route_decision)
                route = response.route_decision
                self.assertEqual(route.scene, case["expected_scene"])
                self.assertEqual(route.intent, case["expected_intent"])
                self.assertEqual(route.allowed_tool_categories, case["expected_allowed_tool_categories"])
                self.assertEqual(route.missing_slots, case.get("expected_missing_slots", []))
                self.assertEqual(response.stop_reason, case["expected_stop_reason"])
                self.assertEqual(response.business_status, case["expected_business_status"])

                if expected_tool := case.get("expected_tool"):
                    self.assertTrue(response.tool_trace, "期望工具执行，但 tool_trace 为空")
                    tool_result = response.tool_trace[-1]
                    self.assertEqual(tool_result.tool_name, expected_tool)
                    self.assertEqual(tool_result.tool_category, case["expected_tool_category"])

                if expected_pending_tool := case.get("expected_pending_tool"):
                    self.assertIsNotNone(response.pending_action)
                    tool_call = response.pending_action.tool_call
                    self.assertEqual(tool_call.tool_name, expected_pending_tool)
                    self.assertEqual(tool_call.tool_category, case["expected_pending_tool_category"])

                for text in case.get("expected_answer_contains", []):
                    self.assertIn(text, response.answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
