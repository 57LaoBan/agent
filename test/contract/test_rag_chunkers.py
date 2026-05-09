from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.rag import EmptyRetriever  # noqa: E402
from xinyidai_agent.rag.chunkers.base import Chunk  # noqa: E402
from xinyidai_agent.rag.chunkers.markdown_chunker import MarkdownStructuredChunker  # noqa: E402
from xinyidai_agent.rag.chunkers.pdf_chunker import PDFLayoutAwareChunker  # noqa: E402
from xinyidai_agent.rag.chunkers.table_chunker import TableChunker  # noqa: E402
from xinyidai_agent.rag.chunkers.text_chunker import TextChunker  # noqa: E402
from xinyidai_agent.rag.document_router import DocumentRouter  # noqa: E402


class RAGChunkerTest(unittest.TestCase):
    """RAG 分块器契约测试。"""

    def test_package_entry_keeps_legacy_empty_retriever_import(self) -> None:
        _, trace = EmptyRetriever().retrieve("信易贷产品", 3)

        self.assertEqual(trace.retriever_type, "empty")
        self.assertTrue(trace.mock)

    def test_text_chunker_uses_sentence_boundary_overlap_and_offsets(self) -> None:
        text = "第一句说明小微企业融资。第二句说明税贷准入条件。第三句说明授权流程。"
        chunker = TextChunker(chunk_size=24, overlap=12, min_chunk_size=1)

        chunks = chunker.chunk(text, "policy")

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[0].content.endswith("第二句说明税贷准入条件。"))
        self.assertTrue(chunks[1].content.startswith("第二句说明税贷准入条件。"))
        self.assertEqual(chunks[0].start_char, 0)
        self.assertEqual(text[chunks[1].start_char : chunks[1].end_char], chunks[1].content)
        self.assertEqual(chunks[0].metadata["type"], "text")

    def test_text_chunker_merges_small_tail_without_duplicate_overlap(self) -> None:
        text = "第一句内容很长。第二句内容很长。短句。"
        chunker = TextChunker(chunk_size=16, overlap=8, min_chunk_size=15)

        chunks = chunker.chunk(text, "policy")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content.count("第二句内容很长。"), 1)
        self.assertIn("短句。", chunks[0].content)
        self.assertTrue(chunks[0].metadata["merged_small_tail"])

    def test_markdown_chunker_preserves_header_path_and_ignores_fenced_heading(self) -> None:
        markdown = """# 产品说明
总览说明第一句。总览说明第二句。
## 准入条件
企业成立满一年。纳税记录良好。
```text
# 代码块内不是标题
```
仍属于准入条件章节。
"""
        chunker = MarkdownStructuredChunker(chunk_size=80, overlap=0, min_chunk_size=1)

        chunks = chunker.chunk(markdown, "credit_doc")

        self.assertEqual(
            [chunk.metadata["header_path"] for chunk in chunks],
            ["/产品说明/", "/产品说明/准入条件/"],
        )
        self.assertIn("代码块内不是标题", chunks[1].content)
        self.assertEqual(chunks[1].metadata["header"], "准入条件")
        self.assertEqual(chunks[1].metadata["level"], 2)

    def test_table_chunker_serializes_markdown_table_with_column_labels(self) -> None:
        table = """| 产品名称 | 利率 | 额度 |
| --- | --- | --- |
| 小微税贷 | 4.5% | 500万 |
| 发票贷 | 5.0% | 300万 |
"""

        serialized = TableChunker().serialize_table_markdown(table)

        self.assertIn("表格：2 行 x 3 列", serialized)
        self.assertIn("列头：产品名称, 利率, 额度", serialized)
        self.assertIn("第 1 行：产品名称: 小微税贷, 利率: 4.5%, 额度: 500万", serialized)

    def test_table_chunker_serializes_html_table_with_colspan(self) -> None:
        html = """
        <table>
          <tr><th>产品</th><th colspan="2">额度</th></tr>
          <tr><td>小微税贷</td><td>最高</td><td>500万</td></tr>
        </table>
        """

        serialized = TableChunker().serialize_table_html(html)

        self.assertIn("表格：1 行 x 3 列", serialized)
        self.assertIn("列头：产品, 额度, 额度", serialized)
        self.assertIn("第 1 行：产品: 小微税贷, 额度: 最高, 额度: 500万", serialized)

    def test_document_router_routes_markdown_text_table_and_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            markdown_chunker = RecordingChunker("markdown")
            text_chunker = RecordingChunker("text")
            table_chunker = RecordingChunker("table")
            pdf_chunker = RecordingPDFChunker()
            router = DocumentRouter(markdown_chunker, pdf_chunker, table_chunker, text_chunker)

            md_path = tmp_path / "credit.md"
            txt_path = tmp_path / "notice.txt"
            html_path = tmp_path / "products.html"
            pdf_path = tmp_path / "policy.pdf"
            md_path.write_text("# 标题\n正文", encoding="utf-8")
            txt_path.write_text("纯文本", encoding="utf-8")
            html_path.write_text("<table><tr><td>A</td></tr></table>", encoding="utf-8")
            pdf_path.write_bytes(b"%PDF-1.4")

            self.assertEqual(router.route_and_chunk(str(md_path))[0].metadata["type"], "markdown")
            self.assertEqual(router.route_and_chunk(str(txt_path))[0].metadata["type"], "text")
            self.assertEqual(router.route_and_chunk(str(html_path))[0].metadata["type"], "table")
            self.assertEqual(router.route_and_chunk(str(pdf_path))[0].metadata["type"], "pdf")
            self.assertEqual(pdf_chunker.calls, [(str(pdf_path), "policy")])

    def test_pdf_chunker_splits_text_and_keeps_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "policy.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")
            elements = [
                FakeElement("Header", "页眉不应进入分块", FakeMetadata(page_number=1)),
                FakeElement(
                    "NarrativeText",
                    "第一句说明政策。第二句说明准入。第三句说明流程。",
                    FakeMetadata(page_number=1),
                ),
                FakeElement(
                    "Table",
                    metadata=FakeMetadata(
                        page_number=2,
                        text_as_html="<table><tr><th>产品</th><th>额度</th></tr>"
                        "<tr><td>小微税贷</td><td>500万</td></tr></table>",
                    ),
                ),
            ]
            chunker = PDFLayoutAwareChunker(chunk_size=17, overlap=0, min_chunk_size=1)
            with patch.object(chunker, "_partition_pdf", return_value=elements):
                chunks = chunker.chunk(str(pdf_path), "policy_pdf")

            self.assertEqual([chunk.metadata["type"] for chunk in chunks], ["pdf_text", "pdf_text", "table"])
            self.assertEqual(chunks[0].metadata["page_number"], 1)
            self.assertEqual(chunks[2].metadata["page_number"], 2)
            self.assertNotIn("页眉不应进入分块", "".join(chunk.content for chunk in chunks))
            self.assertIn("产品: 小微税贷", chunks[2].content)


@dataclass
class FakeMetadata:
    """测试用 PDF 元数据。"""

    page_number: int | None = None
    text_as_html: str | None = None


@dataclass
class FakeElement:
    """测试用 PDF 元素。"""

    category: str
    text: str = ""
    metadata: FakeMetadata | None = None


class RecordingChunker:
    """记录入参的测试分块器。"""

    def __init__(self, label: str) -> None:
        """初始化记录标签。"""
        self.label = label
        self.calls: list[tuple[str, str]] = []

    def chunk(self, content: str, doc_id: str) -> list[Chunk]:
        """记录内容并返回测试 Chunk。"""
        self.calls.append((content, doc_id))
        return [
            Chunk(
                chunk_id=f"{doc_id}_{self.label}_0",
                doc_id=doc_id,
                content=content,
                start_char=0,
                end_char=len(content),
                metadata={"type": self.label},
            )
        ]


class RecordingPDFChunker:
    """记录 PDF 文件路径的测试分块器。"""

    def __init__(self) -> None:
        """初始化调用记录。"""
        self.calls: list[tuple[str, str]] = []

    def chunk(self, file_path: str, doc_id: str) -> list[Chunk]:
        """记录文件路径并返回测试 Chunk。"""
        self.calls.append((file_path, doc_id))
        return [
            Chunk(
                chunk_id=f"{doc_id}_pdf_0",
                doc_id=doc_id,
                content=file_path,
                start_char=0,
                end_char=len(file_path),
                metadata={"type": "pdf"},
            )
        ]


if __name__ == "__main__":
    unittest.main()
