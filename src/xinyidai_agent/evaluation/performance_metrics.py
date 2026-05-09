"""RAG 性能指标。"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True, slots=True)
class LatencyBreakdown:
    """一次请求的延迟分解，单位毫秒。"""

    query_embedding: float = 0.0
    dense_retrieval: float = 0.0
    sparse_retrieval: float = 0.0
    fusion: float = 0.0
    reranking: float = 0.0
    llm_generation: float = 0.0
    total: float = 0.0

    def normalized_total(self) -> float:
        """返回总延迟；若 total 未传入，则按阶段求和。"""
        if self.total > 0:
            return self.total
        return (
            self.query_embedding
            + self.dense_retrieval
            + self.sparse_retrieval
            + self.fusion
            + self.reranking
            + self.llm_generation
        )


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """性能指标汇总。"""

    latency_p50: float
    latency_p95: float
    latency_p99: float
    avg_latency: float
    qps: float
    cache_hit_rate: float
    request_count: int
    stage_avg_ms: dict[str, float]


def calculate_percentile(values: list[float], percentile: float) -> float:
    """使用线性插值计算百分位。"""
    if not values:
        return 0.0
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)

    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def calculate_qps(total_requests: int, window_seconds: float) -> float:
    """计算 QPS。"""
    if total_requests <= 0 or window_seconds <= 0:
        return 0.0
    return total_requests / window_seconds


def calculate_cache_hit_rate(cache_hits: int, total_requests: int) -> float:
    """计算缓存命中率。"""
    if total_requests <= 0:
        return 0.0
    return cache_hits / total_requests


def summarize_performance_metrics(
    latency_breakdowns: list[LatencyBreakdown],
    window_seconds: float,
    cache_hits: int = 0,
    total_requests: int | None = None,
) -> PerformanceMetrics:
    """汇总性能指标。"""
    request_count = total_requests if total_requests is not None else len(latency_breakdowns)
    totals = [item.normalized_total() for item in latency_breakdowns]
    return PerformanceMetrics(
        latency_p50=calculate_percentile(totals, 50),
        latency_p95=calculate_percentile(totals, 95),
        latency_p99=calculate_percentile(totals, 99),
        avg_latency=mean(totals) if totals else 0.0,
        qps=calculate_qps(request_count, window_seconds),
        cache_hit_rate=calculate_cache_hit_rate(cache_hits, request_count),
        request_count=request_count,
        stage_avg_ms=_stage_averages(latency_breakdowns),
    )


def _stage_averages(latency_breakdowns: list[LatencyBreakdown]) -> dict[str, float]:
    """计算各阶段平均延迟。"""
    if not latency_breakdowns:
        return {
            "query_embedding": 0.0,
            "dense_retrieval": 0.0,
            "sparse_retrieval": 0.0,
            "fusion": 0.0,
            "reranking": 0.0,
            "llm_generation": 0.0,
        }
    fields = [
        "query_embedding",
        "dense_retrieval",
        "sparse_retrieval",
        "fusion",
        "reranking",
        "llm_generation",
    ]
    return {
        field: mean(getattr(item, field) for item in latency_breakdowns)
        for field in fields
    }
