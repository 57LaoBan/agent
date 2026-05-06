from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.evidence_policy import EvidencePolicy  # noqa: E402
from xinyidai_agent.protocol import RetrievalTrace, RouteDecision, SourceDocument  # noqa: E402
from xinyidai_agent.rag import EmptyRetriever  # noqa: E402


class EvidencePolicyTest(unittest.TestCase):
    def route(self) -> RouteDecision:
        return RouteDecision(
            scene="KNOWLEDGE_QA",
            intent="POLICY_OR_PRODUCT_QA",
            confidence=0.9,
            route_reason="知识问答。",
        )

    def test_knowledge_qa_without_sources_is_not_ready(self) -> None:
        decision = EvidencePolicy().assess(
            self.route(),
            [],
            RetrievalTrace(query="信易贷适合哪些企业", top_k=5, results_count=0),
        )

        self.assertFalse(decision.ready)
        self.assertEqual(decision.business_status, "PARTIAL_DATA")
        self.assertIn("未检索到依据", decision.user_message)

    def test_sources_make_evidence_ready(self) -> None:
        source = SourceDocument(
            source_id="policy-1",
            title="信易贷适用对象政策",
            content="信易贷支持信用良好的小微企业。",
            source_type="policy",
        )
        decision = EvidencePolicy().assess(
            self.route(),
            [source],
            RetrievalTrace(query="信易贷适合哪些企业", top_k=5, results_count=1),
        )

        self.assertTrue(decision.ready)
        self.assertEqual(decision.source_titles, ["信易贷适用对象政策"])

    def test_answer_is_decorated_with_source_titles(self) -> None:
        source = SourceDocument(
            source_id="policy-1",
            title="信易贷适用对象政策",
            content="信易贷支持信用良好的小微企业。",
            source_type="policy",
        )

        answer = EvidencePolicy().attach_citations("可申请信贷支持。", [source])

        self.assertIn("参考来源", answer)
        self.assertIn("信易贷适用对象政策", answer)


    def test_retrieval_trace_exposes_retriever_identity_and_mock_marker(self) -> None:
        _, trace = EmptyRetriever().retrieve("policy question", 3)

        self.assertEqual(trace.retriever_type, "empty")
        self.assertEqual(trace.index_version, "not_configured")
        self.assertTrue(trace.mock)
        self.assertEqual(trace.results_count, 0)


if __name__ == "__main__":
    unittest.main()
