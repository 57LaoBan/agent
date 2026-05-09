"""文档加载器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """文档模型。

    文档加载器只负责把磁盘文件转换为统一文档对象；后续分块、向量化和入库
    使用该模型中的稳定字段与元数据，不在加载阶段写入任何外部状态。
    """

    doc_id: str
    title: str
    content: str
    source_path: str
    metadata: dict[str, Any]


class DocumentLoader:
    """文档加载器。

    支持从目录或单文件加载 Markdown、TXT、HTML/表格和 PDF。PDF 文本抽取使用
    Unstructured.io，依赖在调用 PDF 路径时懒加载，避免影响普通文本加载链路。
    """

    _text_suffixes = frozenset({".md", ".markdown", ".txt", ".text", ".html", ".htm", ".csv"})
    _pdf_suffixes = frozenset({".pdf"})
    _supported_suffixes = _text_suffixes | _pdf_suffixes
    _pdf_noise_categories = frozenset({"Header", "Footer", "PageNumber"})

    def load_from_directory(
        self,
        directory: Path,
        pattern: str = "**/*.md",
    ) -> list[Document]:
        """从目录加载文档。

        Args:
            directory: 文档目录。
            pattern: pathlib glob 匹配模式，默认递归加载 Markdown。

        Returns:
            按路径稳定排序后的文档列表。
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"文档目录不存在：{directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"路径不是文档目录：{directory}")

        documents: list[Document] = []
        for file_path in sorted(directory.glob(pattern), key=lambda path: path.as_posix()):
            if file_path.is_file():
                documents.append(self.load_from_file(file_path))
        return documents

    def load_from_file(self, file_path: Path) -> Document:
        """从单个文件加载文档。"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文档不存在：{path}")
        if not path.is_file():
            raise IsADirectoryError(f"路径不是文件：{path}")

        suffix = path.suffix.lower()
        if suffix not in self._supported_suffixes:
            raise ValueError(f"不支持的文档类型：{suffix}")

        stat = path.stat()
        if suffix in self._pdf_suffixes:
            content, pdf_metadata = self._load_pdf_content(path)
            metadata = {
                "file_type": suffix,
                "file_size": stat.st_size,
                "loader": "unstructured_pdf",
                **pdf_metadata,
            }
        else:
            content = path.read_text(encoding="utf-8")
            metadata = {
                "file_type": suffix,
                "file_size": stat.st_size,
                "encoding": "utf-8",
                "loader": "text",
            }

        return Document(
            doc_id=self._build_doc_id(path),
            title=self._build_title(path, content),
            content=content,
            source_path=str(path.resolve()),
            metadata=metadata,
        )

    def _load_pdf_content(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """使用 Unstructured.io 抽取 PDF 文本和页码元数据。"""
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as exc:
            raise RuntimeError("加载 PDF 文档需要安装 unstructured[pdf] 和 unstructured-inference。") from exc

        elements = partition_pdf(
            filename=str(file_path),
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=False,
        )

        text_parts: list[str] = []
        page_numbers: set[int] = set()
        element_count = 0
        for element in elements:
            category = getattr(element, "category", "")
            if category in self._pdf_noise_categories:
                continue

            text = (getattr(element, "text", "") or "").strip()
            if not text:
                continue

            metadata = getattr(element, "metadata", None)
            page_number = getattr(metadata, "page_number", None)
            if isinstance(page_number, int):
                page_numbers.add(page_number)

            text_parts.append(text)
            element_count += 1

        return "\n\n".join(text_parts), {
            "page_numbers": sorted(page_numbers),
            "page_count": len(page_numbers),
            "element_count": element_count,
        }

    def _build_doc_id(self, file_path: Path) -> str:
        """基于文件名生成稳定文档 ID。"""
        return file_path.stem

    def _build_title(self, file_path: Path, content: str) -> str:
        """优先使用 Markdown 一级标题作为标题，否则使用文件名。"""
        if file_path.suffix.lower() in {".md", ".markdown"}:
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("# "):
                    return stripped[2:].strip() or file_path.stem
        return file_path.stem
