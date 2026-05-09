"""PDF 布局感知分块器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xinyidai_agent.rag.chunkers.base import Chunk, SentenceBoundaryChunkerMixin
from xinyidai_agent.rag.chunkers.table_chunker import TableChunker


class PDFLayoutAwareChunker(SentenceBoundaryChunkerMixin):
    """PDF 布局感知分块器。

    使用 Unstructured.io 提取标题、正文、列表与表格；表格保持独立块，正文再走句子边界分块。
    """

    _noise_categories = frozenset({"Header", "Footer", "PageNumber"})
    _text_categories = frozenset({"Title", "NarrativeText", "ListItem", "Text"})

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        min_chunk_size: int = 100,
        table_chunker: TableChunker | None = None,
    ) -> None:
        """初始化 PDF 分块器。"""
        self._validate_chunk_params(chunk_size, overlap, min_chunk_size)
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._min_chunk_size = min_chunk_size
        self._table_chunker = table_chunker or TableChunker()

    def chunk(self, file_path: str, doc_id: str) -> list[Chunk]:
        """使用 Unstructured.io 分块 PDF 文件。"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF 文件不存在：{file_path}")

        elements = self._partition_pdf(str(path))
        chunks: list[Chunk] = []

        for element_index, element in enumerate(elements):
            category = getattr(element, "category", "")
            if category in self._noise_categories:
                continue

            if category == "Table":
                table_chunk = self._chunk_table(element, doc_id, len(chunks), element_index)
                if table_chunk:
                    chunks.append(table_chunk)
                continue

            if category in self._text_categories:
                chunks.extend(self._chunk_text_element(element, doc_id, len(chunks), element_index))

        return chunks

    def _partition_pdf(self, file_path: str) -> list[Any]:
        """调用 Unstructured.io 的 PDF 解析能力，缺依赖时给出明确安装错误。"""
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as exc:
            raise RuntimeError(
                "PDFLayoutAwareChunker 需要安装 unstructured[pdf] 后才能解析 PDF。"
            ) from exc

        return list(
            partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True,
                extract_images_in_pdf=False,
            )
        )

    def _chunk_table(
        self,
        table_element: Any,
        doc_id: str,
        chunk_index: int,
        element_index: int,
    ) -> Chunk | None:
        """把 PDF 表格元素序列化为独立表格 Chunk。"""
        metadata = getattr(table_element, "metadata", None)
        table_html = getattr(metadata, "text_as_html", None)
        if not table_html:
            table_text = getattr(table_element, "text", "") or ""
            table_chunks = self._table_chunker.chunk(table_text, doc_id)
            content = table_chunks[0].content if table_chunks else ""
            table_format = "text"
        else:
            content = self._table_chunker.serialize_table_html(table_html)
            table_format = "html"

        if not content.strip():
            return None

        return Chunk(
            chunk_id=f"{doc_id}_table_{chunk_index}",
            doc_id=doc_id,
            content=content,
            start_char=0,
            end_char=len(content),
            metadata={
                "type": "table",
                "format": table_format,
                "page_number": getattr(metadata, "page_number", None),
                "element_index": element_index,
                "chunk_index": chunk_index,
                "char_count": len(content),
            },
        )

    def _chunk_text_element(
        self,
        element: Any,
        doc_id: str,
        start_index: int,
        element_index: int,
    ) -> list[Chunk]:
        """把 PDF 文本类元素按句子边界分块。"""
        text = (getattr(element, "text", "") or "").strip()
        if not text:
            return []

        metadata = getattr(element, "metadata", None)
        page_number = getattr(metadata, "page_number", None)
        spans = self._split_sentence_spans(text)
        return self._build_sentence_chunks(
            spans=spans,
            doc_id=doc_id,
            chunk_id_prefix="text",
            start_index=start_index,
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            min_chunk_size=self._min_chunk_size,
            metadata_factory=lambda chunk_index, _first, _end: {
                "type": "pdf_text",
                "page_number": page_number,
                "element_index": element_index,
                "chunk_index": chunk_index,
            },
        )
