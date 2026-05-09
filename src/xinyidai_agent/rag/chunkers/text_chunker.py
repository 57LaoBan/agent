"""纯文本分块器。"""

from __future__ import annotations

from xinyidai_agent.rag.chunkers.base import Chunk, SentenceBoundaryChunkerMixin


class TextChunker(SentenceBoundaryChunkerMixin):
    """纯文本分块器。

    核心策略：按句子边界分块，合并到目标大小附近，并保留尾部重叠降低边界丢失。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        min_chunk_size: int = 100,
    ) -> None:
        """初始化纯文本分块器。"""
        self._validate_chunk_params(chunk_size, overlap, min_chunk_size)
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._min_chunk_size = min_chunk_size

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        """按句子边界分块纯文本。"""
        spans = self._split_sentence_spans(text)
        return self._build_sentence_chunks(
            spans=spans,
            doc_id=doc_id,
            chunk_id_prefix="chunk",
            start_index=0,
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            min_chunk_size=self._min_chunk_size,
            metadata_factory=lambda chunk_index, _first, _end: {
                "type": "text",
                "chunk_index": chunk_index,
            },
        )
