"""答案质量评测指标。"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import re
from typing import Any


@dataclass(frozen=True, slots=True)
class AnswerMetrics:
    """答案质量指标结果。"""

    faithfulness: float
    relevance: float
    completeness: float
    citation_accuracy: float
    answer_accuracy: float = 0.0
    hallucination_rate: float = 0.0
    evaluated_count: int = 0


async def evaluate_faithfulness(
    question: str,
    answer: str,
    retrieved_docs: list[str],
    llm_client: Any,
) -> float:
    """使用 LLM-as-Judge 评估答案是否忠实于检索文档。"""
    prompt = f"""
请判断以下答案是否完全基于提供的文档，不包含编造或推测内容。

问题：{question}

答案：{answer}

文档：
{chr(10).join(retrieved_docs)}

请只输出 JSON：
{{"analysis": "逐句分析", "score": 0.0到1.0之间的小数}}
"""
    return await _judge_score(prompt, llm_client)


async def evaluate_relevance(
    question: str,
    answer: str,
    llm_client: Any,
) -> float:
    """使用 LLM-as-Judge 评估答案是否回应了问题。"""
    prompt = f"""
请判断答案是否直接回答了问题。

问题：{question}
答案：{answer}

请只输出 JSON：
{{"analysis": "简要说明", "score": 0.0到1.0之间的小数}}
"""
    return await _judge_score(prompt, llm_client)


async def evaluate_completeness(
    question: str,
    answer: str,
    expected_answer: str,
    llm_client: Any,
) -> float:
    """使用 LLM-as-Judge 评估答案相对参考答案的完整性。"""
    prompt = f"""
请判断生成答案相对参考答案是否完整覆盖关键要点。

问题：{question}
参考答案：{expected_answer}
生成答案：{answer}

请只输出 JSON：
{{"analysis": "缺失或覆盖情况", "score": 0.0到1.0之间的小数}}
"""
    return await _judge_score(prompt, llm_client)


def calculate_citation_accuracy(answer: str, cited_source_ids: list[str], relevant_ids: list[str]) -> float:
    """计算引用准确率。

    若调用方没有显式传入引用 ID，会从答案中的 `[chunk_id]` 或 `【chunk_id】` 形态提取。
    """
    citations = cited_source_ids or extract_citation_ids(answer)
    if not citations:
        return 0.0
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    return len(set(citations) & relevant_set) / len(set(citations))


def calculate_reference_overlap(answer: str, expected_answer: str) -> float:
    """基于字符 bigram 的参考答案覆盖率，用作无 LLM 的离线基线指标。"""
    expected_units = _char_bigrams(expected_answer)
    if not expected_units:
        return 0.0
    answer_units = _char_bigrams(answer)
    return len(answer_units & expected_units) / len(expected_units)


def extract_citation_ids(answer: str) -> list[str]:
    """从答案文本中提取引用 ID。"""
    ids = re.findall(r"[\[【]([A-Za-z0-9_.\-/]+)[\]】]", answer)
    return [item for item in ids if item]


def summarize_answer_metrics(rows: list[dict[str, float]]) -> AnswerMetrics:
    """汇总答案质量指标。"""
    if not rows:
        return AnswerMetrics(
            faithfulness=0.0,
            relevance=0.0,
            completeness=0.0,
            citation_accuracy=0.0,
            answer_accuracy=0.0,
            hallucination_rate=0.0,
            evaluated_count=0,
        )

    return AnswerMetrics(
        faithfulness=_mean(rows, "faithfulness"),
        relevance=_mean(rows, "relevance"),
        completeness=_mean(rows, "completeness"),
        citation_accuracy=_mean(rows, "citation_accuracy"),
        answer_accuracy=_mean(rows, "answer_accuracy"),
        hallucination_rate=_mean(rows, "hallucination_rate"),
        evaluated_count=len(rows),
    )


async def _judge_score(prompt: str, llm_client: Any) -> float:
    """调用不同形态的 LLM 客户端并解析 score 字段。"""
    if llm_client is None:
        raise ValueError("LLM-as-Judge 指标需要显式传入 llm_client")

    raw = await _call_llm_client(prompt, llm_client)
    return _parse_score(raw)


async def _call_llm_client(prompt: str, llm_client: Any) -> str:
    """兼容 callable、ainvoke、invoke、chat.completions.create 等常见客户端。"""
    if callable(llm_client):
        result = llm_client(prompt)
    elif hasattr(llm_client, "ainvoke"):
        result = llm_client.ainvoke(prompt)
    elif hasattr(llm_client, "invoke"):
        result = llm_client.invoke(prompt)
    elif hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
        result = llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
    else:
        raise TypeError("不支持的 llm_client 形态")

    if inspect.isawaitable(result):
        result = await result

    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return str(result.content)
    if hasattr(result, "choices"):
        return str(result.choices[0].message.content)
    return str(result)


def _parse_score(raw: str) -> float:
    """从 LLM 输出中解析 0-1 分数。"""
    text = raw.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"无法从 LLM Judge 输出中解析 JSON：{raw}") from None
        payload = json.loads(match.group(0))

    score = float(payload["score"])
    return max(0.0, min(1.0, score))


def _char_bigrams(text: str) -> set[str]:
    """提取中文友好的字符 bigram。"""
    normalized = re.sub(r"\s+", "", text)
    if len(normalized) <= 1:
        return {normalized} if normalized else set()
    return {normalized[index : index + 2] for index in range(len(normalized) - 1)}


def _mean(rows: list[dict[str, float]], key: str) -> float:
    """计算均值。"""
    return sum(row.get(key, 0.0) for row in rows) / len(rows)
