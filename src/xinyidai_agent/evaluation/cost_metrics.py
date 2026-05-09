"""RAG 成本与资源指标。"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True, slots=True)
class QueryCost:
    """单次查询成本，单位人民币元。"""

    embedding_cost: float = 0.0
    retrieval_cost: float = 0.0
    reranking_cost: float = 0.0
    llm_cost: float = 0.0
    total_cost: float = 0.0

    def normalized_total(self) -> float:
        """返回总成本；若 total_cost 未传入，则按阶段求和。"""
        if self.total_cost > 0:
            return self.total_cost
        return self.embedding_cost + self.retrieval_cost + self.reranking_cost + self.llm_cost


@dataclass(frozen=True, slots=True)
class ResourceUtilization:
    """资源利用率。"""

    cpu_usage: float
    memory_usage: float
    gpu_usage: float | None = None


@dataclass(frozen=True, slots=True)
class CostMetrics:
    """成本指标汇总。"""

    avg_cost_per_query: float
    max_cost_per_query: float
    total_cost: float
    query_count: int
    avg_embedding_cost: float
    avg_retrieval_cost: float
    avg_reranking_cost: float
    avg_llm_cost: float


@dataclass(frozen=True, slots=True)
class ResourceMetrics:
    """资源利用率汇总。"""

    avg_cpu_usage: float
    max_cpu_usage: float
    avg_memory_usage: float
    max_memory_usage: float
    avg_gpu_usage: float | None
    max_gpu_usage: float | None
    sample_count: int


def calculate_query_cost(
    embedding_cost: float = 0.0,
    retrieval_cost: float = 0.0,
    reranking_cost: float = 0.0,
    llm_cost: float = 0.0,
) -> QueryCost:
    """计算单次查询成本。"""
    total = embedding_cost + retrieval_cost + reranking_cost + llm_cost
    return QueryCost(
        embedding_cost=embedding_cost,
        retrieval_cost=retrieval_cost,
        reranking_cost=reranking_cost,
        llm_cost=llm_cost,
        total_cost=total,
    )


def summarize_cost_metrics(costs: list[QueryCost]) -> CostMetrics:
    """汇总查询成本。"""
    if not costs:
        return CostMetrics(
            avg_cost_per_query=0.0,
            max_cost_per_query=0.0,
            total_cost=0.0,
            query_count=0,
            avg_embedding_cost=0.0,
            avg_retrieval_cost=0.0,
            avg_reranking_cost=0.0,
            avg_llm_cost=0.0,
        )

    totals = [item.normalized_total() for item in costs]
    return CostMetrics(
        avg_cost_per_query=mean(totals),
        max_cost_per_query=max(totals),
        total_cost=sum(totals),
        query_count=len(costs),
        avg_embedding_cost=mean(item.embedding_cost for item in costs),
        avg_retrieval_cost=mean(item.retrieval_cost for item in costs),
        avg_reranking_cost=mean(item.reranking_cost for item in costs),
        avg_llm_cost=mean(item.llm_cost for item in costs),
    )


def summarize_resource_metrics(samples: list[ResourceUtilization]) -> ResourceMetrics:
    """汇总资源利用率。"""
    if not samples:
        return ResourceMetrics(
            avg_cpu_usage=0.0,
            max_cpu_usage=0.0,
            avg_memory_usage=0.0,
            max_memory_usage=0.0,
            avg_gpu_usage=None,
            max_gpu_usage=None,
            sample_count=0,
        )

    gpu_samples = [item.gpu_usage for item in samples if item.gpu_usage is not None]
    return ResourceMetrics(
        avg_cpu_usage=mean(item.cpu_usage for item in samples),
        max_cpu_usage=max(item.cpu_usage for item in samples),
        avg_memory_usage=mean(item.memory_usage for item in samples),
        max_memory_usage=max(item.memory_usage for item in samples),
        avg_gpu_usage=mean(gpu_samples) if gpu_samples else None,
        max_gpu_usage=max(gpu_samples) if gpu_samples else None,
        sample_count=len(samples),
    )
