from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.evaluation.answer_metrics import (  # noqa: E402
    calculate_citation_accuracy,
    calculate_reference_overlap,
    extract_citation_ids,
)
from xinyidai_agent.evaluation.evaluator import RAGEvaluator  # noqa: E402
from xinyidai_agent.evaluation.retrieval_metrics import (  # noqa: E402
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    evaluate_retrieved_ids,
)


class RetrievalMetricsTest(unittest.TestCase):
    """检索指标计算测试。"""

    def test_recall_precision_mrr_and_ndcg(self) -> None:
        retrieved = ["c2", "c1", "c3", "c4"]
        relevant = ["c1", "c3"]

        self.assertEqual(calculate_recall_at_k(retrieved, relevant, 1), 0.0)
        self.assertEqual(calculate_recall_at_k(retrieved, relevant, 3), 1.0)
        self.assertEqual(calculate_precision_at_k(retrieved, relevant, 2), 0.5)
        self.assertEqual(calculate_mrr(retrieved, relevant), 0.5)
        self.assertGreater(calculate_ndcg_at_k(retrieved, relevant, 5), 0.6)

    def test_evaluate_retrieved_ids_returns_full_metric_set(self) -> None:
        metrics = evaluate_retrieved_ids(["a", "b"], ["b"])

        self.assertEqual(metrics["recall_at_1"], 0.0)
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)


class AnswerMetricsTest(unittest.TestCase):
    """答案指标计算测试。"""

    def test_extract_citations_and_calculate_accuracy(self) -> None:
        answer = "依据 [chunk_1] 和【chunk_2】可判断。"

        self.assertEqual(extract_citation_ids(answer), ["chunk_1", "chunk_2"])
        self.assertEqual(calculate_citation_accuracy(answer, [], ["chunk_1"]), 0.5)

    def test_reference_overlap_is_high_for_same_answer(self) -> None:
        answer = "经营周转贷适合连续经营满 2 年的中小企业。"

        self.assertEqual(calculate_reference_overlap(answer, answer), 1.0)


class EvaluatorTest(unittest.TestCase):
    """RAG 评测器测试。"""

    def test_oracle_retriever_uses_expected_rank_for_dry_run(self) -> None:
        cases = [
            {
                "id": "r1",
                "query": "查询",
                "relevant_chunks": ["c1"],
                "expected_rank": ["c1", "c2"],
            }
        ]

        metrics = asyncio.run(RAGEvaluator(retriever=None).evaluate_retrieval(cases))

        self.assertEqual(metrics.query_count, 1)
        self.assertEqual(metrics.recall_at_1, 1.0)
        self.assertEqual(metrics.mrr, 1.0)


class EvaluationDatasetTest(unittest.TestCase):
    """评测数据集结构测试。"""

    def test_qa_pairs_cover_required_scale_and_scenarios(self) -> None:
        qa_pairs = _read_json(ROOT / "rag_corpus" / "evaluation" / "qa_pairs.json")
        categories = {item["category"] for item in qa_pairs}
        difficulties = {item["difficulty"] for item in qa_pairs}

        self.assertGreaterEqual(len(qa_pairs), 20)
        self.assertTrue({"easy", "medium", "hard"}.issubset(difficulties))
        self.assertTrue(
            {"产品问答", "准入规则", "授权流程", "风险标签", "企业信息", "政策更新"}.issubset(
                categories
            )
        )
        for item in qa_pairs:
            self.assertTrue(item["ground_truth_docs"])
            self.assertTrue(item["ground_truth_chunks"])
            self.assertIn("requires_multi_doc", item["metadata"])

    def test_retrieval_cases_cover_required_scale_and_fields(self) -> None:
        cases = _read_json(ROOT / "rag_corpus" / "evaluation" / "retrieval_test.json")
        scenarios = {item["metadata"]["scenario"] for item in cases}

        self.assertGreaterEqual(len(cases), 30)
        self.assertTrue({"精确匹配", "语义匹配", "多文档召回", "负样本", "排序测试"}.issubset(scenarios))
        for item in cases:
            self.assertTrue(item["relevant_docs"])
            self.assertTrue(item["relevant_chunks"])
            self.assertTrue(item["irrelevant_docs"])
            self.assertTrue(item["expected_rank"])


def _read_json(path: Path) -> list[dict]:
    """读取评测 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
