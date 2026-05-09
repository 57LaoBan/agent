"""分块器基类和通用文本切分能力。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True, slots=True)
class Chunk:
    """文档分块。

    分块对象是后续向量化、引用生成和 trace 的统一输入，字段保持不可变，
    避免并发入库或批处理时被下游代码意外改写。
    """

    chunk_id: str
    doc_id: str
    content: str
    start_char: int
    end_char: int
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TextSpan:
    """带原文偏移的文本片段。"""

    text: str
    start_char: int
    end_char: int


MetadataFactory = Callable[[int, TextSpan, int], dict[str, Any]]


class SentenceBoundaryChunkerMixin:
    """按句子边界构造分块的通用实现。

    该 mixin 不保存可变状态，Markdown、PDF、TXT 分块器可在并发任务中安全复用。
    """

    _sentence_end_chars = frozenset("。！？!?；;\n\r")

    def _validate_chunk_params(
        self,
        chunk_size: int,
        overlap: int,
        min_chunk_size: int,
    ) -> None:
        """校验分块参数，提前阻断会导致死循环或空结果的配置。"""
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if overlap < 0:
            raise ValueError("overlap 不能小于 0")
        if overlap >= chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size 不能小于 0")

    def _split_sentence_spans(self, text: str, base_offset: int = 0) -> list[TextSpan]:
        """按中英文句末符号和换行切出带偏移的句子片段。"""
        spans: list[TextSpan] = []
        segment_start = 0

        for index, char in enumerate(text):
            if char in self._sentence_end_chars:
                self._append_clean_span(spans, text, segment_start, index + 1, base_offset)
                segment_start = index + 1

        self._append_clean_span(spans, text, segment_start, len(text), base_offset)
        return spans

    def _split_long_span(self, span: TextSpan, chunk_size: int) -> list[TextSpan]:
        """将超长句子按字符上限硬切，避免单块远超目标大小。"""
        if len(span.text) <= chunk_size:
            return [span]

        spans: list[TextSpan] = []
        cursor = 0
        while cursor < len(span.text):
            next_cursor = min(cursor + chunk_size, len(span.text))
            piece = span.text[cursor:next_cursor].strip()
            if piece:
                leading_spaces = len(span.text[cursor:next_cursor]) - len(
                    span.text[cursor:next_cursor].lstrip()
                )
                piece_start = span.start_char + cursor + leading_spaces
                spans.append(
                    TextSpan(
                        text=piece,
                        start_char=piece_start,
                        end_char=piece_start + len(piece),
                    )
                )
            cursor = next_cursor
        return spans

    def _build_sentence_chunks(
        self,
        *,
        spans: Iterable[TextSpan],
        doc_id: str,
        chunk_id_prefix: str,
        start_index: int,
        chunk_size: int,
        overlap: int,
        min_chunk_size: int,
        metadata_factory: MetadataFactory,
        separator: str = "",
    ) -> list[Chunk]:
        """将句子片段合并为目标大小附近的 Chunk 列表。"""
        chunks: list[Chunk] = []
        current: list[TextSpan] = []
        current_size = 0

        normalized_spans: list[TextSpan] = []
        for span in spans:
            normalized_spans.extend(self._split_long_span(span, chunk_size))

        for span in normalized_spans:
            separator_size = len(separator) if current else 0
            if current and current_size + separator_size + len(span.text) > chunk_size:
                emitted = self._emit_chunk(
                    current=current,
                    chunks=chunks,
                    doc_id=doc_id,
                    chunk_id_prefix=chunk_id_prefix,
                    start_index=start_index,
                    min_chunk_size=min_chunk_size,
                    metadata_factory=metadata_factory,
                    separator=separator,
                )
                current = self._select_overlap(current, overlap) if emitted else current
                current_size = self._spans_size(current, separator)

            current.append(span)
            current_size = self._spans_size(current, separator)

        if current:
            self._emit_chunk(
                current=current,
                chunks=chunks,
                doc_id=doc_id,
                chunk_id_prefix=chunk_id_prefix,
                start_index=start_index,
                min_chunk_size=min_chunk_size,
                metadata_factory=metadata_factory,
                separator=separator,
                force_when_empty=True,
            )

        return chunks

    def _append_clean_span(
        self,
        spans: list[TextSpan],
        text: str,
        start: int,
        end: int,
        base_offset: int,
    ) -> None:
        """追加去除首尾空白后的片段，并保留其在原文中的真实偏移。"""
        raw = text[start:end]
        if not raw.strip():
            return

        leading_spaces = len(raw) - len(raw.lstrip())
        clean_text = raw.strip()
        clean_start = base_offset + start + leading_spaces
        spans.append(TextSpan(text=clean_text, start_char=clean_start, end_char=clean_start + len(clean_text)))

    def _emit_chunk(
        self,
        *,
        current: list[TextSpan],
        chunks: list[Chunk],
        doc_id: str,
        chunk_id_prefix: str,
        start_index: int,
        min_chunk_size: int,
        metadata_factory: MetadataFactory,
        separator: str,
        force_when_empty: bool = False,
    ) -> bool:
        """把当前句子窗口落为 Chunk，过小尾块优先并入前一个块。"""
        chunk_text = separator.join(span.text for span in current).strip()
        if not chunk_text:
            return False

        if len(chunk_text) < min_chunk_size and chunks and not force_when_empty:
            return False

        if len(chunk_text) < min_chunk_size and chunks and force_when_empty:
            previous = chunks.pop()
            # 小尾块常常带有上一块的 overlap 句子，合并时先剔除已在上一块末尾出现的前缀。
            merge_spans = list(current)
            while merge_spans and previous.content.endswith(merge_spans[0].text):
                merge_spans.pop(0)

            tail_text = separator.join(span.text for span in merge_spans).strip()
            if not tail_text:
                chunks.append(previous)
                return True

            merged_text = separator.join([previous.content, tail_text]).strip()
            merged_metadata = dict(previous.metadata)
            merged_metadata["merged_small_tail"] = True
            chunks.append(
                Chunk(
                    chunk_id=previous.chunk_id,
                    doc_id=previous.doc_id,
                    content=merged_text,
                    start_char=previous.start_char,
                    end_char=merge_spans[-1].end_char,
                    metadata=merged_metadata,
                )
            )
            return True

        chunk_index = start_index + len(chunks)
        first_span = current[0]
        last_span = current[-1]
        metadata = metadata_factory(chunk_index, first_span, last_span.end_char)
        metadata.setdefault("chunk_index", chunk_index)
        metadata.setdefault("char_count", len(chunk_text))

        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_{chunk_id_prefix}_{chunk_index}",
                doc_id=doc_id,
                content=chunk_text,
                start_char=first_span.start_char,
                end_char=last_span.end_char,
                metadata=metadata,
            )
        )
        return True

    def _select_overlap(self, spans: list[TextSpan], overlap: int) -> list[TextSpan]:
        """选择尾部若干句子作为重叠窗口，按字符数近似控制。"""
        if overlap <= 0:
            return []

        selected: list[TextSpan] = []
        total = 0
        for span in reversed(spans):
            selected.append(span)
            total += len(span.text)
            if total >= overlap:
                break
        return list(reversed(selected))

    def _spans_size(self, spans: list[TextSpan], separator: str) -> int:
        """计算一组片段合并后的字符长度。"""
        if not spans:
            return 0
        return sum(len(span.text) for span in spans) + len(separator) * (len(spans) - 1)
