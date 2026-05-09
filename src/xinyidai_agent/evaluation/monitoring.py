"""RAG 指标监控与告警。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from xinyidai_agent.evaluation.business_metrics import BusinessMetrics
from xinyidai_agent.evaluation.cost_metrics import CostMetrics, ResourceMetrics
from xinyidai_agent.evaluation.performance_metrics import PerformanceMetrics
from xinyidai_agent.evaluation.retrieval_metrics import RetrievalMetrics


AlertSeverity = Literal["warning", "critical"]


@dataclass(frozen=True, slots=True)
class RAGMetricTargets:
    """信易贷 RAG 指标目标值。"""

    answer_accuracy: float = 0.85
    task_success_rate: float = 0.80
    abstention_accuracy: float = 0.90
    hallucination_rate: float = 0.05
    recall_at_5: float = 0.80
    mrr: float = 0.70
    ndcg_at_5: float = 0.80
    precision_at_5: float = 0.60
    latency_p95_ms: float = 1000.0
    latency_p99_ms: float = 2000.0
    qps: float = 10.0
    cache_hit_rate: float = 0.20
    cost_per_query: float = 0.01
    memory_usage: float = 0.80


@dataclass(frozen=True, slots=True)
class AlertEvent:
    """告警事件。"""

    name: str
    severity: AlertSeverity
    message: str
    actual: float
    target: float


@dataclass(frozen=True, slots=True)
class MonitoringSnapshot:
    """一次完整 RAG 指标快照。"""

    retrieval: RetrievalMetrics | None = None
    business: BusinessMetrics | None = None
    performance: PerformanceMetrics | None = None
    cost: CostMetrics | None = None
    resource: ResourceMetrics | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为 JSON 友好结构。"""
        payload = {}
        for name in ("retrieval", "business", "performance", "cost", "resource"):
            value = getattr(self, name)
            payload[name] = asdict(value) if value is not None else None
        payload["extra"] = dict(self.extra)
        return payload


def evaluate_alerts(
    snapshot: MonitoringSnapshot,
    targets: RAGMetricTargets | None = None,
) -> list[AlertEvent]:
    """根据目标值评估告警。"""
    target = targets or RAGMetricTargets()
    alerts: list[AlertEvent] = []

    if snapshot.business is not None:
        _min_alert(alerts, "AnswerAccuracyLow", snapshot.business.answer_accuracy, target.answer_accuracy)
        _min_alert(alerts, "TaskSuccessRateLow", snapshot.business.task_success_rate, target.task_success_rate)
        _min_alert(alerts, "AbstentionAccuracyLow", snapshot.business.abstention_accuracy, target.abstention_accuracy)
        _max_alert(alerts, "HallucinationRateHigh", snapshot.business.hallucination_rate, target.hallucination_rate)

    if snapshot.retrieval is not None:
        _min_alert(alerts, "RecallAt5Low", snapshot.retrieval.recall_at_5, target.recall_at_5)
        _min_alert(alerts, "MRRLow", snapshot.retrieval.mrr, target.mrr)
        _min_alert(alerts, "NDCGAt5Low", snapshot.retrieval.ndcg_at_5, target.ndcg_at_5)
        _min_alert(alerts, "PrecisionAt5Low", snapshot.retrieval.precision_at_5, target.precision_at_5)

    if snapshot.performance is not None:
        _max_alert(alerts, "LatencyP95High", snapshot.performance.latency_p95, target.latency_p95_ms)
        _max_alert(alerts, "LatencyP99High", snapshot.performance.latency_p99, target.latency_p99_ms)
        _min_alert(alerts, "QPSLow", snapshot.performance.qps, target.qps)
        _min_alert(alerts, "CacheHitRateLow", snapshot.performance.cache_hit_rate, target.cache_hit_rate)

    if snapshot.cost is not None:
        _max_alert(alerts, "CostPerQueryHigh", snapshot.cost.avg_cost_per_query, target.cost_per_query)

    if snapshot.resource is not None:
        _max_alert(alerts, "MemoryUsageHigh", snapshot.resource.max_memory_usage, target.memory_usage)

    return alerts


def export_prometheus_metrics(snapshot: MonitoringSnapshot, prefix: str = "rag") -> str:
    """导出 Prometheus 文本格式指标。"""
    lines: list[str] = []
    flattened = _flatten(snapshot.to_dict())
    for key, value in sorted(flattened.items()):
        if isinstance(value, int | float):
            metric_name = f"{prefix}_{key}".replace(".", "_")
            lines.append(f"{metric_name} {float(value)}")
    return "\n".join(lines) + ("\n" if lines else "")


def _min_alert(alerts: list[AlertEvent], name: str, actual: float, target: float) -> None:
    """低于目标值时告警。"""
    if actual < target:
        alerts.append(
            AlertEvent(
                name=name,
                severity="critical" if actual < target * 0.8 else "warning",
                message=f"{name}: actual {actual:.4f} < target {target:.4f}",
                actual=actual,
                target=target,
            )
        )


def _max_alert(alerts: list[AlertEvent], name: str, actual: float, target: float) -> None:
    """高于目标值时告警。"""
    if actual > target:
        alerts.append(
            AlertEvent(
                name=name,
                severity="critical" if actual > target * 1.2 else "warning",
                message=f"{name}: actual {actual:.4f} > target {target:.4f}",
                actual=actual,
                target=target,
            )
        )


def _flatten(value: dict, prefix: str = "") -> dict[str, float | int | str | None]:
    """扁平化嵌套字典。"""
    rows: dict[str, float | int | str | None] = {}
    for key, item in value.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            rows.update(_flatten(item, full_key))
        else:
            rows[full_key] = item
    return rows
