"""Markdown 结构化分块器。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from xinyidai_agent.rag.chunkers.base import Chunk, SentenceBoundaryChunkerMixin


@dataclass(frozen=True, slots=True)
class MarkdownSection:
    """Markdown 章节。"""

    header: str
    level: int
    content: str
    header_path: str
    start_char: int
    end_char: int


class MarkdownStructuredChunker(SentenceBoundaryChunkerMixin):
    """Markdown 结构化分块器。

    先按标题层级拆 section，再在 section 内按句子边界分块，元数据保留标题路径。
    """

    _heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    _fence_pattern = re.compile(r"^\s*(```+|~~~+)")

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        min_chunk_size: int = 100,
    ) -> None:
        """初始化 Markdown 分块器。"""
        self._validate_chunk_params(chunk_size, overlap, min_chunk_size)
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._min_chunk_size = min_chunk_size

    def chunk(self, markdown_text: str, doc_id: str) -> list[Chunk]:
        """分块 Markdown 文档。"""
        chunks: list[Chunk] = []
        for section in self._split_by_headers(markdown_text):
            section_chunks = self._chunk_section(section, doc_id, len(chunks))
            chunks.extend(section_chunks)
        return chunks

    def _split_by_headers(self, text: str) -> list[MarkdownSection]:
        """按 Markdown 标题层级分割章节，并忽略代码块内的伪标题。"""
        sections: list[MarkdownSection] = []
        header_stack: list[tuple[int, str]] = []
        current_lines: list[str] = []
        current_start = 0
        offset = 0
        in_fence = False

        for line in text.splitlines(keepends=True):
            stripped_line = line.rstrip("\r\n")
            if self._fence_pattern.match(stripped_line):
                in_fence = not in_fence

            match = None if in_fence else self._heading_pattern.match(stripped_line)
            if match:
                self._append_section(sections, header_stack, current_lines, current_start, offset)

                level = len(match.group(1))
                header = match.group(2).strip()
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                header_stack.append((level, header))

                current_lines = []
                current_start = offset + len(line)
            else:
                current_lines.append(line)

            offset += len(line)

        self._append_section(sections, header_stack, current_lines, current_start, len(text))
        return sections

    def _append_section(
        self,
        sections: list[MarkdownSection],
        header_stack: list[tuple[int, str]],
        lines: list[str],
        start_char: int,
        end_char: int,
    ) -> None:
        """把累计的 section 内容追加到结果列表。"""
        content = "".join(lines).strip()
        if not content:
            return

        header = header_stack[-1][1] if header_stack else ""
        level = header_stack[-1][0] if header_stack else 0
        sections.append(
            MarkdownSection(
                header=header,
                level=level,
                content=content,
                header_path=self._build_header_path(header_stack),
                start_char=start_char + len("".join(lines)) - len("".join(lines).lstrip()),
                end_char=end_char,
            )
        )

    def _build_header_path(self, header_stack: list[tuple[int, str]]) -> str:
        """构建标题路径。"""
        if not header_stack:
            return "/"
        return "/" + "/".join(header for _, header in header_stack) + "/"

    def _chunk_section(
        self,
        section: MarkdownSection,
        doc_id: str,
        start_index: int,
    ) -> list[Chunk]:
        """对单个 Markdown section 进行句子边界分块。"""
        spans = self._split_sentence_spans(section.content, base_offset=section.start_char)
        return self._build_sentence_chunks(
            spans=spans,
            doc_id=doc_id,
            chunk_id_prefix="chunk",
            start_index=start_index,
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            min_chunk_size=self._min_chunk_size,
            metadata_factory=lambda chunk_index, _first, _end: {
                "type": "markdown_section",
                "header": section.header,
                "header_path": section.header_path,
                "level": section.level,
                "chunk_index": chunk_index,
            },
        )
