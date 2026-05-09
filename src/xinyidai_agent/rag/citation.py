"""RAG 引用生成器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from xinyidai_agent.protocol import SourceDocument
from xinyidai_agent.rag.retriever import RetrievalResult


@dataclass(frozen=True, slots=True)
class Citation:
    """面向模型提示词和前端展示的引用信息。"""

    index: int
    source_id: str
    title: str
    snippet: str
    score: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
    url: str | None = None


class CitationGenerator:
    """从检索结果生成引用、上下文和工具证据。"""

    def __init__(self, max_snippet_chars: int = 240, max_context_chars: int = 6000) -> None:
        """初始化引用截断参数。"""
        if max_snippet_chars <= 0:
            raise ValueError("max_snippet_chars 必须大于 0")
        if max_context_chars <= 0:
            raise ValueError("max_context_chars 必须大于 0")
        self._max_snippet_chars = max_snippet_chars
        self._max_context_chars = max_context_chars

    def generate(self, results: Iterable[RetrievalResult | SourceDocument]) -> list[Citation]:
        """从检索结果生成有序引用列表。"""
        citations: list[Citation] = []
        for index, result in enumerate(results, start=1):
            source = _to_source_document(result)
            citations.append(
                Citation(
                    index=index,
                    source_id=source.source_id,
                    title=source.title,
                    snippet=self._snippet(source.content),
                    score=source.score,
                    metadata=dict(source.metadata),
                    url=source.url,
                )
            )
        return citations

    def format_context(self, results: Iterable[RetrievalResult | SourceDocument]) -> str:
        """格式化为可直接放入 Prompt 的引用上下文。"""
        blocks: list[str] = []
        used_chars = 0
        for citation in self.generate(results):
            score_text = "" if citation.score is None else f" score={citation.score:.4f}"
            block = f"[{citation.index}] {citation.title}{score_text}\n{citation.snippet}"
            if used_chars + len(block) > self._max_context_chars:
                break
            blocks.append(block)
            used_chars += len(block)
        return "\n\n".join(blocks)

    def to_source_documents(self, results: Iterable[RetrievalResult | SourceDocument]) -> list[SourceDocument]:
        """转换为工具层 SourceDocument 列表。"""
        return [_to_source_document(result) for result in results]

    def _snippet(self, content: str) -> str:
        """生成稳定长度的引用摘要。"""
        normalized = " ".join(str(content or "").split())
        if len(normalized) <= self._max_snippet_chars:
            return normalized
        return normalized[: self._max_snippet_chars - 3].rstrip() + "..."


def _to_source_document(result: RetrievalResult | SourceDocument) -> SourceDocument:
    """统一检索结果和工具证据模型。"""
    if isinstance(result, SourceDocument):
        return result
    return result.to_source_document()
