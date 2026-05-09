"""表格专用分块器。"""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
import re

from xinyidai_agent.rag.chunkers.base import Chunk


@dataclass(frozen=True, slots=True)
class _ParsedCell:
    """HTML 表格单元格。"""

    text: str
    is_header: bool
    colspan: int = 1
    rowspan: int = 1


class _HTMLTableParser(HTMLParser):
    """基于标准库的 HTML 表格解析器，避免为基础序列化引入额外运行时依赖。"""

    def __init__(self) -> None:
        """初始化解析状态。"""
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_ParsedCell]] = []
        self._in_table = False
        self._in_row = False
        self._current_row: list[_ParsedCell] = []
        self._current_cell_tag: str | None = None
        self._current_cell_attrs: dict[str, str] = {}
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """处理开始标签，只采集 table/tr/td/th 内的信息。"""
        if tag == "table":
            self._in_table = True
        elif tag == "tr" and self._in_table:
            self._in_row = True
            self._current_row = []
        elif tag in {"td", "th"} and self._in_row:
            self._current_cell_tag = tag
            self._current_cell_attrs = {key: value or "" for key, value in attrs}
            self._current_text = []

    def handle_data(self, data: str) -> None:
        """采集单元格文本。"""
        if self._current_cell_tag:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        """处理结束标签并落单元格或行。"""
        if tag in {"td", "th"} and self._current_cell_tag == tag:
            text = " ".join("".join(self._current_text).split())
            self._current_row.append(
                _ParsedCell(
                    text=unescape(text),
                    is_header=tag == "th",
                    colspan=self._positive_int(self._current_cell_attrs.get("colspan"), 1),
                    rowspan=self._positive_int(self._current_cell_attrs.get("rowspan"), 1),
                )
            )
            self._current_cell_tag = None
            self._current_cell_attrs = {}
            self._current_text = []
        elif tag == "tr" and self._in_row:
            if self._current_row:
                self.rows.append(self._current_row)
            self._in_row = False
            self._current_row = []
        elif tag == "table":
            self._in_table = False

    def _positive_int(self, value: str | None, default: int) -> int:
        """读取 HTML rowspan/colspan，非法值按 1 处理。"""
        try:
            parsed = int(value or default)
        except ValueError:
            return default
        return max(parsed, 1)


class TableChunker:
    """表格分块器。

    表格作为独立语义单元序列化，输出中显式保留列头和每一行的字段关系。
    """

    _markdown_separator = re.compile(r"^:?-{3,}:?$")

    def chunk(self, table_text: str, doc_id: str) -> list[Chunk]:
        """把 HTML 或 Markdown 表格转换为一个独立 Chunk。"""
        if "<table" in table_text.lower():
            content = self.serialize_table_html(table_text)
            table_format = "html"
        else:
            content = self.serialize_table_markdown(table_text)
            table_format = "markdown"

        if not content.strip():
            return []

        return [
            Chunk(
                chunk_id=f"{doc_id}_table_0",
                doc_id=doc_id,
                content=content,
                start_char=0,
                end_char=len(table_text),
                metadata={
                    "type": "table",
                    "format": table_format,
                    "chunk_index": 0,
                    "char_count": len(content),
                },
            )
        ]

    def serialize_table_html(self, table_html: str) -> str:
        """序列化 HTML 表格为便于 LLM 理解的文本。"""
        parser = _HTMLTableParser()
        parser.feed(table_html)
        rows = self._expand_html_rows(parser.rows)
        if not rows:
            return ""

        first_row_is_header = any(cell.is_header for cell in parser.rows[0])
        headers = rows[0] if first_row_is_header else self._default_headers(len(rows[0]))
        data_rows = rows[1:] if first_row_is_header else rows
        return self._serialize_with_labels(headers, data_rows)

    def serialize_table_markdown(self, table_md: str) -> str:
        """序列化 Markdown 表格为便于 LLM 理解的文本。"""
        lines = [line.strip() for line in table_md.strip().splitlines() if line.strip()]
        if len(lines) < 2:
            return table_md.strip()

        headers = self._split_markdown_row(lines[0])
        separator_cells = self._split_markdown_row(lines[1])
        if not headers or not separator_cells or not all(
            self._markdown_separator.match(cell) for cell in separator_cells
        ):
            return table_md.strip()

        rows = [self._split_markdown_row(line) for line in lines[2:]]
        return self._serialize_with_labels(headers, rows)

    def _expand_html_rows(self, rows: list[list[_ParsedCell]]) -> list[list[str]]:
        """展开 HTML 行列合并，保证序列化时列数量稳定。"""
        expanded_rows: list[list[str]] = []
        rowspans: dict[int, list[tuple[str, int]]] = {}

        for row_index, row in enumerate(rows):
            expanded: list[str] = []
            column_index = 0

            while column_index in rowspans:
                text, remaining = rowspans[column_index].pop(0)
                expanded.append(text)
                if remaining > 1:
                    rowspans.setdefault(column_index, []).append((text, remaining - 1))
                if not rowspans[column_index]:
                    rowspans.pop(column_index)
                column_index += 1

            for cell in row:
                while column_index in rowspans:
                    text, remaining = rowspans[column_index].pop(0)
                    expanded.append(text)
                    if remaining > 1:
                        rowspans.setdefault(column_index, []).append((text, remaining - 1))
                    if not rowspans[column_index]:
                        rowspans.pop(column_index)
                    column_index += 1

                for _ in range(cell.colspan):
                    expanded.append(cell.text)
                    if cell.rowspan > 1:
                        rowspans.setdefault(column_index, []).append((cell.text, cell.rowspan - 1))
                    column_index += 1

            expanded_rows.append(expanded)
            _ = row_index

        return expanded_rows

    def _serialize_with_labels(self, headers: list[str], rows: list[list[str]]) -> str:
        """序列化为带字段标签的文本格式。"""
        max_columns = max([len(headers), *(len(row) for row in rows)] or [0])
        normalized_headers = self._normalize_headers(headers, max_columns)

        lines = [
            f"表格：{len(rows)} 行 x {len(normalized_headers)} 列",
            f"列头：{', '.join(normalized_headers)}",
        ]
        for row_index, row in enumerate(rows, start=1):
            normalized_row = self._normalize_row(row, len(normalized_headers))
            row_text = ", ".join(
                f"{header}: {value}" for header, value in zip(normalized_headers, normalized_row)
            )
            lines.append(f"第 {row_index} 行：{row_text}")

        return "\n".join(lines)

    def _split_markdown_row(self, line: str) -> list[str]:
        """拆分 Markdown 表格行，支持转义竖线。"""
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|") and not stripped.endswith(r"\|"):
            stripped = stripped[:-1]

        cells: list[str] = []
        current: list[str] = []
        escaped = False
        for char in stripped:
            if escaped:
                current.append(char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "|":
                cells.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        cells.append("".join(current).strip())
        return cells

    def _normalize_headers(self, headers: list[str], columns: int) -> list[str]:
        """补齐空表头，保证每列都有稳定字段名。"""
        normalized = [header or f"列{index + 1}" for index, header in enumerate(headers[:columns])]
        while len(normalized) < columns:
            normalized.append(f"列{len(normalized) + 1}")
        return normalized

    def _normalize_row(self, row: list[str], columns: int) -> list[str]:
        """补齐或截断数据行到表头列数。"""
        normalized = row[:columns]
        while len(normalized) < columns:
            normalized.append("")
        return normalized

    def _default_headers(self, columns: int) -> list[str]:
        """为无表头表格生成默认列名。"""
        return [f"列{index + 1}" for index in range(columns)]
