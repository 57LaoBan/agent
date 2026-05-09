from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.rag import Document, DocumentLoader  # noqa: E402


class DocumentLoaderTest(unittest.TestCase):
    """文档加载器契约测试。"""

    def test_package_entry_exports_document_loader_models(self) -> None:
        """包入口应导出 Document 和 DocumentLoader，便于后续入库脚本复用。"""
        self.assertIs(DocumentLoader, DocumentLoader)
        self.assertEqual(Document.__name__, "Document")

    def test_load_markdown_file_uses_h1_title_and_metadata(self) -> None:
        """Markdown 文件应按 UTF-8 读取，并优先使用一级标题作为文档标题。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "credit_products.md"
            path.write_text("# 信易贷产品说明\n正文内容。", encoding="utf-8")

            document = DocumentLoader().load_from_file(path)

        self.assertEqual(document.doc_id, "credit_products")
        self.assertEqual(document.title, "信易贷产品说明")
        self.assertEqual(document.content, "# 信易贷产品说明\n正文内容。")
        self.assertTrue(document.source_path.endswith("credit_products.md"))
        self.assertEqual(document.metadata["file_type"], ".md")
        self.assertEqual(document.metadata["encoding"], "utf-8")
        self.assertEqual(document.metadata["loader"], "text")
        self.assertGreater(document.metadata["file_size"], 0)

    def test_load_from_directory_is_recursive_and_stably_sorted(self) -> None:
        """目录加载应递归匹配 pattern，并按路径稳定排序输出。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            nested = root / "nested"
            nested.mkdir()
            (nested / "b.md").write_text("# B\n正文。", encoding="utf-8")
            (root / "a.md").write_text("# A\n正文。", encoding="utf-8")
            (root / "ignored.txt").write_text("忽略。", encoding="utf-8")

            documents = DocumentLoader().load_from_directory(root, pattern="**/*.md")

        self.assertEqual([document.doc_id for document in documents], ["a", "b"])
        self.assertEqual([document.title for document in documents], ["A", "B"])

    def test_load_text_file_uses_file_stem_title(self) -> None:
        """非 Markdown 文本文件应使用文件名作为标题。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "notice.txt"
            path.write_text("普通文本。", encoding="utf-8")

            document = DocumentLoader().load_from_file(path)

        self.assertEqual(document.doc_id, "notice")
        self.assertEqual(document.title, "notice")
        self.assertEqual(document.content, "普通文本。")
        self.assertEqual(document.metadata["file_type"], ".txt")

    def test_load_pdf_file_merges_pdf_metadata(self) -> None:
        """PDF 文件应调用 PDF 抽取链路，并保留页码、元素数等元数据。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "policy.pdf"
            path.write_bytes(b"%PDF-1.4")
            loader = DocumentLoader()

            with patch.object(
                loader,
                "_load_pdf_content",
                return_value=("第一页内容\n\n第二页内容", {"page_numbers": [1, 2], "page_count": 2}),
            ):
                document = loader.load_from_file(path)

        self.assertEqual(document.doc_id, "policy")
        self.assertEqual(document.title, "policy")
        self.assertEqual(document.content, "第一页内容\n\n第二页内容")
        self.assertEqual(document.metadata["file_type"], ".pdf")
        self.assertEqual(document.metadata["loader"], "unstructured_pdf")
        self.assertEqual(document.metadata["page_numbers"], [1, 2])
        self.assertEqual(document.metadata["page_count"], 2)

    def test_load_from_directory_rejects_non_directory(self) -> None:
        """目录加载入口应拒绝文件路径，避免误把单文件当目录扫描。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "single.md"
            path.write_text("# 单文件", encoding="utf-8")

            with self.assertRaises(NotADirectoryError):
                DocumentLoader().load_from_directory(path)

    def test_load_from_file_rejects_unsupported_suffix(self) -> None:
        """加载器应拒绝未声明支持的文件类型。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "archive.zip"
            path.write_bytes(b"zip")

            with self.assertRaises(ValueError):
                DocumentLoader().load_from_file(path)


if __name__ == "__main__":
    unittest.main()
