from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.evaluation.business_metrics import (  # noqa: E402
    AbstentionEvaluation,
    AnswerEvaluation,
    HallucinationEvaluation,
    TaskResult,
    calculate_abstention_accuracy,
    calculate_hallucination_rate,
    calculate_task_success_rate,
    summarize_business_metrics,
)
from xinyidai_agent.evaluation.cost_metrics import (  # noqa: E402
    ResourceUtilization,
    calculate_query_cost,
    summarize_cost_metrics,
    summarize_resource_metrics,
)
from xinyidai_agent.evaluation.dataset import (  # noqa: E402
    load_qa_dataset,
    validate_qa_dataset,
    validate_retrieval_dataset,
)
from xinyidai_agent.evaluation.monitoring import (  # noqa: E402
    MonitoringSnapshot,
    RAGMetricTargets,
    evaluate_alerts,
    export_prometheus_metrics,
)
from xinyidai_agent.evaluation.performance_metrics import (  # noqa: E402
    LatencyBreakdown,
    calculate_cache_hit_rate,
    calculate_percentile,
    calculate_qps,
    summarize_performance_metrics,
)
from xinyidai_agent.evaluation.retrieval_metrics import (  # noqa: E402
    calculate_multi_hop_recall,
    evaluate_retrieved_ids,
    summarize_retrieval_metrics,
)


class FullMetricsSystemTest(unittest.TestCase):
    """完整指标体系契约测试。"""

    def test_business_metrics_cover_accuracy_success_abstention_and_hallucination(self) -> None:
        answers = [
            AnswerEvaluation("q1", "a", "a", True, difficulty="easy"),
            AnswerEvaluation("q2", "bad", "a", False, difficulty="hard", error_type="hallucination"),
        ]
        tasks = [TaskResult("q1", "success"), TaskResult("q2", "partial")]
        abstentions = [
            AbstentionEvaluation("q1", has_answer_in_kb=False, system_abstained=True),
            AbstentionEvaluation("q2", has_answer_in_kb=True, system_abstained=True),
        ]
        hallucinations = [
            HallucinationEvaluation("q1", "a", False),
            HallucinationEvaluation("q2", "bad", True, ["无依据陈述"]),
        ]

        metrics = summarize_business_metrics(answers, tasks, abstentions, hallucinations)

        self.assertEqual(metrics.answer_accuracy, 0.5)
        self.assertEqual(calculate_task_success_rate(tasks), 0.75)
        self.assertEqual(calculate_abstention_accuracy(abstentions), 0.5)
        self.assertEqual(calculate_hallucination_rate(hallucinations), 0.5)
        self.assertEqual(metrics.error_type_distribution["hallucination"], 1)
        self.assertEqual(metrics.abstention_confusion_matrix.true_positive, 1)

    def test_retrieval_metrics_include_optional_multi_hop_and_deep_recall(self) -> None:
        metrics = evaluate_retrieved_ids(["a", "b", "c"], ["a", "c"])
        summary = summarize_retrieval_metrics([metrics])

        self.assertEqual(metrics["recall_at_20"], 1.0)
        self.assertEqual(metrics["recall_at_100"], 1.0)
        self.assertEqual(calculate_multi_hop_recall(["a", "b"], ["a", "c"], 2), 0.0)
        self.assertEqual(summary.multi_hop_recall, 1.0)

    def test_performance_metrics_calculate_latency_qps_and_cache(self) -> None:
        breakdowns = [
            LatencyBreakdown(query_embedding=10, dense_retrieval=20, llm_generation=100),
            LatencyBreakdown(total=300),
            LatencyBreakdown(total=900),
        ]

        metrics = summarize_performance_metrics(breakdowns, window_seconds=3, cache_hits=1)

        self.assertEqual(calculate_qps(30, 3), 10.0)
        self.assertEqual(calculate_cache_hit_rate(2, 4), 0.5)
        self.assertEqual(calculate_percentile([1, 2, 3], 50), 2.0)
        self.assertEqual(metrics.qps, 1.0)
        self.assertGreater(metrics.latency_p95, metrics.latency_p50)
        self.assertEqual(metrics.stage_avg_ms["query_embedding"], 10 / 3)

    def test_cost_and_resource_metrics(self) -> None:
        costs = [
            calculate_query_cost(embedding_cost=0.001, retrieval_cost=0.002),
            calculate_query_cost(llm_cost=0.01),
        ]
        resources = [
            ResourceUtilization(cpu_usage=0.5, memory_usage=0.7, gpu_usage=0.6),
            ResourceUtilization(cpu_usage=0.9, memory_usage=0.8, gpu_usage=None),
        ]

        cost_metrics = summarize_cost_metrics(costs)
        resource_metrics = summarize_resource_metrics(resources)

        self.assertEqual(cost_metrics.query_count, 2)
        self.assertEqual(cost_metrics.total_cost, 0.013000000000000001)
        self.assertEqual(resource_metrics.max_cpu_usage, 0.9)
        self.assertEqual(resource_metrics.avg_gpu_usage, 0.6)

    def test_dataset_validation_uses_existing_evaluation_files(self) -> None:
        qa_rows = json.loads((ROOT / "rag_corpus" / "evaluation" / "qa_pairs.json").read_text(encoding="utf-8"))
        retrieval_rows = json.loads(
            (ROOT / "rag_corpus" / "evaluation" / "retrieval_test.json").read_text(encoding="utf-8")
        )

        qa_result = validate_qa_dataset(qa_rows)
        retrieval_result = validate_retrieval_dataset(retrieval_rows)
        examples = load_qa_dataset(ROOT / "rag_corpus" / "evaluation" / "qa_pairs.json")

        self.assertTrue(qa_result.valid)
        self.assertTrue(retrieval_result.valid)
        self.assertGreaterEqual(len(examples), 20)
        self.assertIn("产品问答", qa_result.profile.category_distribution)

    def test_monitoring_alerts_and_prometheus_export(self) -> None:
        business = summarize_business_metrics(
            [AnswerEvaluation("q", "bad", "good", False, error_type="wrong")],
            [TaskResult("q", "failed")],
            [AbstentionEvaluation("q", has_answer_in_kb=False, system_abstained=False)],
            [HallucinationEvaluation("q", "bad", True)],
        )
        performance = summarize_performance_metrics([LatencyBreakdown(total=2500)], 1, cache_hits=0)
        cost = summarize_cost_metrics([calculate_query_cost(llm_cost=0.2)])
        resource = summarize_resource_metrics([ResourceUtilization(cpu_usage=0.5, memory_usage=0.95)])
        snapshot = MonitoringSnapshot(
            business=business,
            performance=performance,
            cost=cost,
            resource=resource,
        )

        alerts = evaluate_alerts(snapshot, RAGMetricTargets())
        prometheus_text = export_prometheus_metrics(snapshot)

        self.assertTrue(any(alert.name == "HallucinationRateHigh" for alert in alerts))
        self.assertTrue(any(alert.name == "LatencyP95High" for alert in alerts))
        self.assertIn("rag_business_answer_accuracy", prometheus_text)


if __name__ == "__main__":
    unittest.main()
