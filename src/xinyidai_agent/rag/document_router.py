"""文档类型识别与路由。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from xinyidai_agent.rag.chunkers.base import Chunk


class DocumentChunker(Protocol):
    """分块器协议。"""

    def chunk(self, content: str, doc_id: str) -> list[Chunk]:
        """分块文档。"""


class PDFDocumentChunker(Protocol):
    """PDF 分块器协议。"""

    def chunk(self, file_path: str, doc_id: str) -> list[Chunk]:
        """按文件路径分块 PDF。"""


class DocumentRouter:
    """文档类型识别与路由器。

    根据文件后缀选择 Markdown、PDF、表格或纯文本分块策略；实例本身不保存请求态，
    可以被批处理任务并发复用。
    """

    _markdown_suffixes = frozenset({".md", ".markdown"})
    _pdf_suffixes = frozenset({".pdf"})
    _text_suffixes = frozenset({".txt", ".text"})
    _table_suffixes = frozenset({".html", ".htm", ".table", ".csv"})

    def __init__(
        self,
        markdown_chunker: DocumentChunker,
        pdf_chunker: PDFDocumentChunker,
        table_chunker: DocumentChunker,
        text_chunker: DocumentChunker,
    ) -> None:
        """初始化文档路由器。"""
        self._markdown_chunker = markdown_chunker
        self._pdf_chunker = pdf_chunker
        self._table_chunker = table_chunker
        self._text_chunker = text_chunker

    def route_and_chunk(self, file_path: str) -> list[Chunk]:
        """识别文档类型并选择对应分块器。"""
        path = Path(file_path)
        suffix = path.suffix.lower()
        doc_id = path.stem

        if suffix in self._pdf_suffixes:
            return self._pdf_chunker.chunk(str(path), doc_id)

        if not path.exists():
            raise FileNotFoundError(f"文档不存在：{file_path}")

        content = path.read_text(encoding="utf-8")
        if suffix in self._markdown_suffixes:
            return self._markdown_chunker.chunk(content, doc_id)
        if suffix in self._text_suffixes:
            return self._text_chunker.chunk(content, doc_id)
        if suffix in self._table_suffixes:
            return self._table_chunker.chunk(content, doc_id)

        raise ValueError(f"不支持的文件类型：{suffix}")
