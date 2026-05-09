"""检索质量评测指标。"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """检索指标汇总结果。"""

    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_5: float
    precision_at_5: float
    recall_at_20: float = 0.0
    recall_at_100: float = 0.0
    multi_hop_recall: float = 0.0
    query_count: int = 0


def calculate_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """计算 Recall@K。"""
    if k <= 0 or not relevant_ids:
        return 0.0

    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(retrieved_k & relevant_set)
    return hits / len(relevant_set)


def calculate_precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """计算 Precision@K。"""
    if k <= 0:
        return 0.0

    retrieved_k = retrieved_ids[:k]
    if not retrieved_k:
        return 0.0

    relevant_set = set(relevant_ids)
    hits = len(set(retrieved_k) & relevant_set)
    return hits / k


def calculate_multi_hop_recall(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """计算多跳召回。

    多跳问题要求所有相关证据都在 TopK 内出现，因此该指标是严格的 0/1 判定。
    """
    if k <= 0 or not relevant_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    return 1.0 if set(relevant_ids).issubset(retrieved_k) else 0.0


def calculate_mrr(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """计算 Mean Reciprocal Rank。"""
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def calculate_ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """计算二值相关性的 NDCG@K。"""
    if k <= 0 or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / log2(rank + 1)

    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def summarize_retrieval_metrics(
    per_query_results: list[dict[str, float]],
) -> RetrievalMetrics:
    """汇总每个查询的检索指标。"""
    query_count = len(per_query_results)
    if query_count == 0:
        return RetrievalMetrics(
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            recall_at_20=0.0,
            recall_at_100=0.0,
            mrr=0.0,
            ndcg_at_5=0.0,
            precision_at_5=0.0,
            multi_hop_recall=0.0,
            query_count=0,
        )

    return RetrievalMetrics(
        recall_at_1=_mean(per_query_results, "recall_at_1"),
        recall_at_3=_mean(per_query_results, "recall_at_3"),
        recall_at_5=_mean(per_query_results, "recall_at_5"),
        recall_at_10=_mean(per_query_results, "recall_at_10"),
        recall_at_20=_mean(per_query_results, "recall_at_20"),
        recall_at_100=_mean(per_query_results, "recall_at_100"),
        mrr=_mean(per_query_results, "mrr"),
        ndcg_at_5=_mean(per_query_results, "ndcg_at_5"),
        precision_at_5=_mean(per_query_results, "precision_at_5"),
        multi_hop_recall=_mean(per_query_results, "multi_hop_recall"),
        query_count=query_count,
    )


def evaluate_retrieved_ids(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> dict[str, float]:
    """计算单个查询的完整检索指标。"""
    return {
        "recall_at_1": calculate_recall_at_k(retrieved_ids, relevant_ids, 1),
        "recall_at_3": calculate_recall_at_k(retrieved_ids, relevant_ids, 3),
        "recall_at_5": calculate_recall_at_k(retrieved_ids, relevant_ids, 5),
        "recall_at_10": calculate_recall_at_k(retrieved_ids, relevant_ids, 10),
        "recall_at_20": calculate_recall_at_k(retrieved_ids, relevant_ids, 20),
        "recall_at_100": calculate_recall_at_k(retrieved_ids, relevant_ids, 100),
        "mrr": calculate_mrr(retrieved_ids, relevant_ids),
        "ndcg_at_5": calculate_ndcg_at_k(retrieved_ids, relevant_ids, 5),
        "precision_at_5": calculate_precision_at_k(retrieved_ids, relevant_ids, 5),
        "multi_hop_recall": calculate_multi_hop_recall(retrieved_ids, relevant_ids, 5),
    }


def _mean(rows: list[dict[str, float]], key: str) -> float:
    """计算指定字段均值。"""
    return sum(row.get(key, 0.0) for row in rows) / len(rows)
