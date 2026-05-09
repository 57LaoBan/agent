"""RAG P0 评测脚本。"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.evaluation import RAGEvaluator  # noqa: E402
from xinyidai_agent.evaluation.dataset import validate_qa_dataset, validate_retrieval_dataset  # noqa: E402
from xinyidai_agent.evaluation.monitoring import (  # noqa: E402
    MonitoringSnapshot,
    evaluate_alerts,
    export_prometheus_metrics,
)
from xinyidai_agent.rag.embedder import BGEEmbedder  # noqa: E402
from xinyidai_agent.rag.reranker import CrossEncoderReranker  # noqa: E402
from xinyidai_agent.rag.retriever import build_hybrid_retriever  # noqa: E402
from xinyidai_agent.rag.storage import PgVectorStore  # noqa: E402


async def main() -> None:
    """加载评测集并运行 P0 指标 dry-run。"""
    parser = argparse.ArgumentParser(description="运行 RAG P0 评测")
    parser.add_argument("--evaluation-dir", default="rag_corpus/evaluation")
    parser.add_argument("--output", default="rag_corpus/evaluation/results.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dsn", default=os.getenv("RAG_PGVECTOR_DSN"), help="真实检索评测 PostgreSQL DSN")
    parser.add_argument("--live", action="store_true", help="使用真实 pgvector + BGE + Reranker 评测")
    parser.add_argument("--device", default="cpu", help="真实评测时模型运行设备")
    parser.add_argument("--no-rerank", action="store_true", help="真实评测时关闭 Cross-Encoder 重排")
    args = parser.parse_args()

    evaluation_dir = ROOT / args.evaluation_dir
    retrieval_cases = _read_json(evaluation_dir / "retrieval_test.json")
    qa_pairs = _read_json(evaluation_dir / "qa_pairs.json")
    retrieval_validation = validate_retrieval_dataset(retrieval_cases)
    qa_validation = validate_qa_dataset(qa_pairs)
    if not retrieval_validation.valid or not qa_validation.valid:
        raise SystemExit(
            "评测集校验失败："
            f"retrieval_errors={retrieval_validation.errors}; qa_errors={qa_validation.errors}"
        )

    store = None
    retriever = None
    mode = "dry_run_expected_rank"
    if args.live:
        if not args.dsn:
            raise SystemExit("真实评测需要传入 --dsn 或设置 RAG_PGVECTOR_DSN。")
        embedder = BGEEmbedder(device=args.device)
        store = PgVectorStore(args.dsn, embedding_dimension=embedder.dimension)
        await store.initialize()
        hybrid_retriever = build_hybrid_retriever(embedder, store)
        reranker = None if args.no_rerank else CrossEncoderReranker(device=args.device)
        retriever = _LiveRAGEvaluationRetriever(hybrid_retriever, reranker=reranker)
        mode = "live_pgvector_hybrid"

    try:
        evaluator = RAGEvaluator(retriever=retriever)
        retrieval_details = await evaluator.evaluate_retrieval_details(retrieval_cases, top_k=args.top_k)
        retrieval_metrics = await evaluator.evaluate_retrieval(retrieval_cases, top_k=args.top_k)
        e2e_result = await evaluator.evaluate_end_to_end(qa_pairs, top_k=args.top_k)
    finally:
        if store is not None:
            await store.close()
    snapshot = MonitoringSnapshot(
        retrieval=retrieval_metrics,
        business=None,
        performance=None,
        cost=None,
        resource=None,
        extra={
            "qa_pair_count": float(len(qa_pairs)),
            "retrieval_case_count": float(len(retrieval_cases)),
        },
    )
    alerts = evaluate_alerts(snapshot)
    prometheus_text = export_prometheus_metrics(snapshot)

    result = {
        "mode": mode,
        "retrieval_metrics": asdict(retrieval_metrics),
        "answer_metrics": asdict(e2e_result.answer_metrics),
        "dataset_validation": {
            "qa": {
                "valid": qa_validation.valid,
                "errors": qa_validation.errors,
                "warnings": qa_validation.warnings,
                "profile": asdict(qa_validation.profile),
            },
            "retrieval": {
                "valid": retrieval_validation.valid,
                "errors": retrieval_validation.errors,
                "warnings": retrieval_validation.warnings,
                "profile": asdict(retrieval_validation.profile),
            },
        },
        "alerts": [asdict(alert) for alert in alerts],
        "prometheus_metrics": prometheus_text,
        "retrieval_case_count": len(retrieval_cases),
        "qa_pair_count": len(qa_pairs),
        "retrieval_details": retrieval_details,
    }
    output_path = ROOT / args.output
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("RAG P0 评测完成")
    print(f"检索用例: {len(retrieval_cases)}")
    print(f"问答用例: {len(qa_pairs)}")
    print(f"Recall@5: {retrieval_metrics.recall_at_5:.3f}")
    print(f"MRR: {retrieval_metrics.mrr:.3f}")
    print(f"NDCG@5: {retrieval_metrics.ndcg_at_5:.3f}")
    print(f"结果文件: {output_path}")


def _read_json(path: Path) -> list[dict]:
    """读取 JSON 评测数据。"""
    return json.loads(path.read_text(encoding="utf-8"))


class _LiveRAGEvaluationRetriever:
    """评测脚本内的异步真实检索适配器。"""

    def __init__(self, hybrid_retriever: Any, reranker: Any | None = None) -> None:
        """初始化真实检索链路。"""
        self._hybrid_retriever = hybrid_retriever
        self._reranker = reranker

    async def retrieve(self, query: str, top_k: int) -> list[Any]:
        """执行混合检索和可选重排序。"""
        candidates = await self._hybrid_retriever.retrieve(query, top_k=max(top_k * 4, top_k), mode="hybrid")
        if self._reranker is None:
            return candidates[:top_k]
        return await asyncio.to_thread(self._reranker.rerank, query, candidates, top_k)


if __name__ == "__main__":
    asyncio.run(main())
