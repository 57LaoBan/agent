"""RAG 业务指标。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Literal


AnswerErrorType = Literal["hallucination", "incomplete", "irrelevant", "wrong", "unsupported"]
TaskFeedback = Literal["success", "failed", "partial"]


@dataclass(frozen=True, slots=True)
class AnswerEvaluation:
    """答案人工或 LLM Judge 评估结果。"""

    query: str
    generated_answer: str
    expected_answer: str
    is_correct: bool
    difficulty: str = "unknown"
    error_type: AnswerErrorType | None = None


@dataclass(frozen=True, slots=True)
class TaskResult:
    """用户任务结果。"""

    query: str
    user_feedback: TaskFeedback
    follow_up_count: int = 0


@dataclass(frozen=True, slots=True)
class AbstentionEvaluation:
    """拒答评估结果。"""

    query: str
    has_answer_in_kb: bool
    system_abstained: bool

    @property
    def is_correct(self) -> bool:
        """判断拒答或回答决策是否正确。"""
        return self.has_answer_in_kb != self.system_abstained


@dataclass(frozen=True, slots=True)
class HallucinationEvaluation:
    """幻觉评估结果。"""

    query: str
    generated_answer: str
    has_hallucination: bool
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AbstentionConfusionMatrix:
    """拒答混淆矩阵。"""

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


@dataclass(frozen=True, slots=True)
class BusinessMetrics:
    """业务指标汇总。"""

    answer_accuracy: float
    task_success_rate: float
    abstention_accuracy: float
    hallucination_rate: float
    answer_accuracy_by_difficulty: dict[str, float]
    error_type_distribution: dict[str, int]
    abstention_confusion_matrix: AbstentionConfusionMatrix
    evaluated_answer_count: int
    task_count: int
    abstention_count: int
    hallucination_count: int


def calculate_answer_accuracy(evaluations: list[AnswerEvaluation]) -> float:
    """计算答案准确率。"""
    if not evaluations:
        return 0.0
    return sum(1 for item in evaluations if item.is_correct) / len(evaluations)


def calculate_answer_accuracy_by_difficulty(
    evaluations: list[AnswerEvaluation],
) -> dict[str, float]:
    """按难度计算答案准确率。"""
    grouped: dict[str, list[AnswerEvaluation]] = defaultdict(list)
    for item in evaluations:
        grouped[item.difficulty].append(item)
    return {difficulty: calculate_answer_accuracy(rows) for difficulty, rows in grouped.items()}


def calculate_task_success_rate(results: list[TaskResult], partial_credit: float = 0.5) -> float:
    """计算任务成功率，partial 默认记 0.5 分。"""
    if not results:
        return 0.0
    score = 0.0
    for item in results:
        if item.user_feedback == "success":
            score += 1.0
        elif item.user_feedback == "partial":
            score += partial_credit
    return score / len(results)


def calculate_abstention_accuracy(evaluations: list[AbstentionEvaluation]) -> float:
    """计算拒答准确率。"""
    if not evaluations:
        return 0.0
    return sum(1 for item in evaluations if item.is_correct) / len(evaluations)


def calculate_abstention_confusion_matrix(
    evaluations: list[AbstentionEvaluation],
) -> AbstentionConfusionMatrix:
    """计算拒答混淆矩阵。"""
    true_positive = true_negative = false_positive = false_negative = 0
    for item in evaluations:
        if not item.has_answer_in_kb and item.system_abstained:
            true_positive += 1
        elif item.has_answer_in_kb and not item.system_abstained:
            true_negative += 1
        elif item.has_answer_in_kb and item.system_abstained:
            false_positive += 1
        else:
            false_negative += 1
    return AbstentionConfusionMatrix(
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def calculate_hallucination_rate(evaluations: list[HallucinationEvaluation]) -> float:
    """计算幻觉率。"""
    if not evaluations:
        return 0.0
    return sum(1 for item in evaluations if item.has_hallucination) / len(evaluations)


def summarize_business_metrics(
    answer_evaluations: list[AnswerEvaluation] | None = None,
    task_results: list[TaskResult] | None = None,
    abstention_evaluations: list[AbstentionEvaluation] | None = None,
    hallucination_evaluations: list[HallucinationEvaluation] | None = None,
) -> BusinessMetrics:
    """汇总业务指标。"""
    answers = answer_evaluations or []
    tasks = task_results or []
    abstentions = abstention_evaluations or []
    hallucinations = hallucination_evaluations or []
    error_counts = Counter(item.error_type for item in answers if item.error_type)

    return BusinessMetrics(
        answer_accuracy=calculate_answer_accuracy(answers),
        task_success_rate=calculate_task_success_rate(tasks),
        abstention_accuracy=calculate_abstention_accuracy(abstentions),
        hallucination_rate=calculate_hallucination_rate(hallucinations),
        answer_accuracy_by_difficulty=calculate_answer_accuracy_by_difficulty(answers),
        error_type_distribution=dict(error_counts),
        abstention_confusion_matrix=calculate_abstention_confusion_matrix(abstentions),
        evaluated_answer_count=len(answers),
        task_count=len(tasks),
        abstention_count=len(abstentions),
        hallucination_count=len(hallucinations),
    )
