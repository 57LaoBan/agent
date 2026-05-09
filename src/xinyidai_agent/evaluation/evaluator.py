"""RAG 评测主流程。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
from typing import Any

from xinyidai_agent.evaluation.answer_metrics import (
    AnswerMetrics,
    calculate_citation_accuracy,
    calculate_reference_overlap,
    evaluate_completeness,
    evaluate_faithfulness,
    evaluate_relevance,
    summarize_answer_metrics,
)
from xinyidai_agent.evaluation.retrieval_metrics import (
    RetrievalMetrics,
    evaluate_retrieved_ids,
    summarize_retrieval_metrics,
)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """完整评测结果。"""

    retrieval_metrics: RetrievalMetrics
    answer_metrics: AnswerMetrics
    per_query_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化字典。"""
        return {
            "retrieval_metrics": asdict(self.retrieval_metrics),
            "answer_metrics": asdict(self.answer_metrics),
            "per_query_results": self.per_query_results,
        }


class RAGEvaluator:
    """RAG 系统评测器。"""

    def __init__(
        self,
        retriever: Any,
        answer_generator: Any | None = None,
        llm_client: Any | None = None,
    ) -> None:
        """初始化评测器。"""
        self._retriever = retriever
        self._answer_generator = answer_generator
        self._llm_client = llm_client

    async def evaluate_retrieval(
        self,
        test_cases: list[dict[str, Any]],
        top_k: int = 10,
    ) -> RetrievalMetrics:
        """评测检索质量。"""
        per_query_results = await self.evaluate_retrieval_details(test_cases, top_k=top_k)
        return summarize_retrieval_metrics(
            [row["metrics"] for row in per_query_results],
        )

    async def evaluate_retrieval_details(
        self,
        test_cases: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """评测检索质量并返回每条查询明细。"""
        rows: list[dict[str, Any]] = []
        for case in test_cases:
            query = str(case["query"])
            relevant_ids = _case_relevant_ids(case)
            retrieved = await self._retrieve(query, top_k=top_k, case=case)
            retrieved_ids = [_extract_result_id(item) for item in retrieved]
            metrics = evaluate_retrieved_ids(retrieved_ids, relevant_ids)
            rows.append(
                {
                    "id": case.get("id"),
                    "query": query,
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": relevant_ids,
                    "metrics": metrics,
                }
            )
        return rows

    async def evaluate_end_to_end(
        self,
        qa_pairs: list[dict[str, Any]],
        top_k: int = 10,
    ) -> EvaluationResult:
        """端到端评测检索与答案质量。"""
        per_query_results: list[dict[str, Any]] = []
        retrieval_rows: list[dict[str, float]] = []
        answer_rows: list[dict[str, float]] = []

        for case in qa_pairs:
            question = str(case["question"])
            relevant_ids = _case_relevant_ids(case)
            retrieved = await self._retrieve(question, top_k=top_k, case=case)
            retrieved_ids = [_extract_result_id(item) for item in retrieved]
            retrieval_metrics = evaluate_retrieved_ids(retrieved_ids, relevant_ids)
            retrieval_rows.append(retrieval_metrics)

            answer = await self._generate_answer(question, retrieved, case)
            answer_metrics = await self._evaluate_answer(case, answer, retrieved, relevant_ids)
            answer_rows.append(answer_metrics)

            per_query_results.append(
                {
                    "id": case.get("id"),
                    "question": question,
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": relevant_ids,
                    "answer": answer,
                    "retrieval_metrics": retrieval_metrics,
                    "answer_metrics": answer_metrics,
                }
            )

        return EvaluationResult(
            retrieval_metrics=summarize_retrieval_metrics(retrieval_rows),
            answer_metrics=summarize_answer_metrics(answer_rows),
            per_query_results=per_query_results,
        )

    async def _retrieve(self, query: str, top_k: int, case: dict[str, Any]) -> list[Any]:
        """调用真实检索器或兼容测试用 oracle 检索器。"""
        if self._retriever is None:
            return [
                {"chunk_id": chunk_id, "content": "", "metadata": {"source": "expected_rank"}}
                for chunk_id in case.get("expected_rank", [])[:top_k]
            ]

        if hasattr(self._retriever, "retrieve"):
            result = self._retriever.retrieve(query, top_k)
        else:
            result = self._retriever(query, top_k)

        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, tuple):
            result = result[0]
        return list(result or [])

    async def _generate_answer(
        self,
        question: str,
        retrieved: list[Any],
        case: dict[str, Any],
    ) -> str:
        """生成答案；未注入生成器时使用参考答案作为离线基线答案。"""
        if self._answer_generator is None:
            return str(case.get("expected_answer", ""))

        if hasattr(self._answer_generator, "generate"):
            result = self._answer_generator.generate(question, retrieved)
        else:
            result = self._answer_generator(question, retrieved)
        if inspect.isawaitable(result):
            result = await result
        return str(result)

    async def _evaluate_answer(
        self,
        case: dict[str, Any],
        answer: str,
        retrieved: list[Any],
        relevant_ids: list[str],
    ) -> dict[str, float]:
        """评估单条答案质量。"""
        question = str(case["question"])
        expected_answer = str(case.get("expected_answer", ""))
        cited_ids = _case_cited_ids(case, answer)
        citation_accuracy = calculate_citation_accuracy(answer, cited_ids, relevant_ids)

        if self._llm_client is None:
            completeness = calculate_reference_overlap(answer, expected_answer)
            answer_accuracy = 1.0 if completeness >= 0.75 else completeness
            return {
                "faithfulness": 1.0 if answer else 0.0,
                "relevance": calculate_reference_overlap(question + answer, question),
                "completeness": completeness,
                "citation_accuracy": citation_accuracy,
                "answer_accuracy": answer_accuracy,
                "hallucination_rate": 0.0 if answer_accuracy >= 0.75 else 1.0 - answer_accuracy,
            }

        docs = [_extract_result_text(item) for item in retrieved]
        faithfulness = await evaluate_faithfulness(question, answer, docs, self._llm_client)
        relevance = await evaluate_relevance(question, answer, self._llm_client)
        completeness = await evaluate_completeness(
            question,
            answer,
            expected_answer,
            self._llm_client,
        )
        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "completeness": completeness,
            "citation_accuracy": citation_accuracy,
            "answer_accuracy": completeness,
            "hallucination_rate": 1.0 - faithfulness,
        }


def _case_relevant_ids(case: dict[str, Any]) -> list[str]:
    """从不同数据集字段中读取相关 chunk ID。"""
    return list(
        case.get("relevant_chunks")
        or case.get("ground_truth_chunks")
        or case.get("relevant_chunk_ids")
        or []
    )


def _case_cited_ids(case: dict[str, Any], _answer: str) -> list[str]:
    """读取用例显式引用 ID。"""
    return list(case.get("cited_source_ids") or case.get("ground_truth_chunks") or [])


def _extract_result_id(item: Any) -> str:
    """从检索结果对象中提取 chunk ID。"""
    if isinstance(item, dict):
        metadata = item.get("metadata") or {}
        return str(
            item.get("chunk_id")
            or item.get("source_id")
            or item.get("id")
            or metadata.get("chunk_id")
            or metadata.get("source_id")
            or ""
        )
    metadata = getattr(item, "metadata", {}) or {}
    return str(
        getattr(item, "chunk_id", None)
        or getattr(item, "source_id", None)
        or getattr(item, "id", None)
        or metadata.get("chunk_id")
        or metadata.get("source_id")
        or ""
    )


def _extract_result_text(item: Any) -> str:
    """从检索结果对象中提取正文。"""
    if isinstance(item, dict):
        return str(item.get("content") or item.get("text") or "")
    return str(getattr(item, "content", None) or getattr(item, "text", None) or "")
