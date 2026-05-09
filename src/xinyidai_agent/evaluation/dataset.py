"""RAG 评测数据集模型与校验。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    """评测样例。"""

    id: str
    category: str
    difficulty: str
    question: str
    expected_answer: str
    relevant_chunk_ids: list[str]
    relevant_doc_ids: list[str]
    requires_multi_doc: bool
    requires_reasoning: bool
    has_answer: bool


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """数据集画像。"""

    total_count: int
    category_distribution: dict[str, int]
    difficulty_distribution: dict[str, int]
    multi_doc_count: int
    reasoning_count: int
    no_answer_count: int


@dataclass(frozen=True, slots=True)
class DatasetValidationResult:
    """数据集校验结果。"""

    valid: bool
    errors: list[str]
    warnings: list[str]
    profile: DatasetProfile


def load_qa_dataset(path: Path) -> list[EvaluationExample]:
    """读取并规范化问答评测集。"""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [normalise_qa_example(row) for row in rows]


def normalise_qa_example(row: dict[str, Any]) -> EvaluationExample:
    """把 JSON 行规范化为评测样例。"""
    metadata = row.get("metadata") or {}
    return EvaluationExample(
        id=str(row["id"]),
        category=str(row["category"]),
        difficulty=str(row["difficulty"]),
        question=str(row["question"]),
        expected_answer=str(row.get("expected_answer", "")),
        relevant_chunk_ids=list(row.get("ground_truth_chunks") or row.get("relevant_chunk_ids") or []),
        relevant_doc_ids=list(row.get("ground_truth_docs") or row.get("relevant_doc_ids") or []),
        requires_multi_doc=bool(metadata.get("requires_multi_doc", False)),
        requires_reasoning=bool(metadata.get("requires_reasoning", False)),
        has_answer=bool(metadata.get("has_answer", True)),
    )


def profile_dataset(examples: list[EvaluationExample]) -> DatasetProfile:
    """生成数据集画像。"""
    category_counts = Counter(item.category for item in examples)
    difficulty_counts = Counter(item.difficulty for item in examples)
    return DatasetProfile(
        total_count=len(examples),
        category_distribution=dict(category_counts),
        difficulty_distribution=dict(difficulty_counts),
        multi_doc_count=sum(1 for item in examples if item.requires_multi_doc),
        reasoning_count=sum(1 for item in examples if item.requires_reasoning),
        no_answer_count=sum(1 for item in examples if not item.has_answer),
    )


def validate_qa_dataset(
    rows: list[dict[str, Any]],
    min_size: int = 20,
) -> DatasetValidationResult:
    """校验问答评测集结构和覆盖。"""
    errors: list[str] = []
    warnings: list[str] = []
    examples: list[EvaluationExample] = []

    for index, row in enumerate(rows):
        try:
            example = normalise_qa_example(row)
        except KeyError as exc:
            errors.append(f"第 {index} 条缺少字段：{exc}")
            continue
        examples.append(example)
        if not example.relevant_chunk_ids:
            errors.append(f"{example.id} 缺少 relevant_chunk_ids/ground_truth_chunks")
        if not example.relevant_doc_ids:
            errors.append(f"{example.id} 缺少 relevant_doc_ids/ground_truth_docs")

    profile = profile_dataset(examples)
    if profile.total_count < min_size:
        errors.append(f"问答集数量不足：{profile.total_count} < {min_size}")
    for difficulty in ("easy", "medium", "hard"):
        if profile.difficulty_distribution.get(difficulty, 0) == 0:
            errors.append(f"缺少难度分层：{difficulty}")
    if profile.multi_doc_count == 0:
        warnings.append("缺少多文档样例")
    if profile.reasoning_count == 0:
        warnings.append("缺少推理样例")

    return DatasetValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        profile=profile,
    )


def validate_retrieval_dataset(
    rows: list[dict[str, Any]],
    min_size: int = 30,
) -> DatasetValidationResult:
    """校验检索评测集结构和覆盖。"""
    errors: list[str] = []
    warnings: list[str] = []
    examples: list[EvaluationExample] = []

    for index, row in enumerate(rows):
        try:
            examples.append(
                EvaluationExample(
                    id=str(row["id"]),
                    category=str((row.get("metadata") or {}).get("scenario", "检索")),
                    difficulty=str((row.get("metadata") or {}).get("ambiguity", "unknown")),
                    question=str(row["query"]),
                    expected_answer="",
                    relevant_chunk_ids=list(row.get("relevant_chunks") or []),
                    relevant_doc_ids=list(row.get("relevant_docs") or []),
                    requires_multi_doc=len(row.get("relevant_docs") or []) > 1,
                    requires_reasoning=(row.get("metadata") or {}).get("query_type") == "reasoning",
                    has_answer=True,
                )
            )
        except KeyError as exc:
            errors.append(f"第 {index} 条缺少字段：{exc}")
            continue
        if not row.get("expected_rank"):
            errors.append(f"{row.get('id', index)} 缺少 expected_rank")
        if not row.get("irrelevant_docs"):
            warnings.append(f"{row.get('id', index)} 缺少 irrelevant_docs")

    profile = profile_dataset(examples)
    if profile.total_count < min_size:
        errors.append(f"检索集数量不足：{profile.total_count} < {min_size}")

    return DatasetValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        profile=profile,
    )
