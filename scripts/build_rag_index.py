"""构建 RAG pgvector 索引。

支持通过 --table-prefix 在同一 PostgreSQL 实例下与 baseline 索引隔离，
并把当前批次的 chunker/embedder 版本写入 chunk metadata，便于检索时回溯。
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.config import load_env_file  # noqa: E402
from xinyidai_agent.rag.chunkers import (  # noqa: E402
    MarkdownStructuredChunker,
    PDFLayoutAwareChunker,
    TableChunker,
    TextChunker,
)
from xinyidai_agent.rag.document_loader import Document, DocumentLoader  # noqa: E402
from xinyidai_agent.rag.document_router import DocumentRouter  # noqa: E402
from xinyidai_agent.rag.embedder import BGEEmbedder  # noqa: E402
from xinyidai_agent.rag.storage import PgVectorStore  # noqa: E402


# 当前离线入库版本号；写入 chunk.metadata，便于评测、灰度与回滚定位。
INDEX_VERSION = "v2"
EMBEDDER_NAME = "bge-m3"
CHUNKER_NAME = "production_router"


async def main() -> None:
    """命令行入口：加载文档、分块、向量化并入库。"""
    parser = argparse.ArgumentParser(description="构建信易贷 RAG pgvector 索引")
    parser.add_argument(
        "--docs-dir",
        default=os.getenv("RAG_DOCS_DIR", "rag_corpus/knowledge"),
        help="知识库文档目录",
    )
    parser.add_argument("--pattern", default="**/*.md", help="pathlib glob 匹配模式")
    parser.add_argument(
        "--dsn",
        default=os.getenv("RAG_PGVECTOR_DSN") or os.getenv("SESSION_DB_DSN"),
        help="PostgreSQL DSN，缺省读 RAG_PGVECTOR_DSN / SESSION_DB_DSN",
    )
    parser.add_argument(
        "--table-prefix",
        default=os.getenv("RAG_TABLE_PREFIX", "rag_v2_"),
        help="表命名空间前缀，缺省 rag_v2_，与 baseline (public.rag_chunks) 隔离",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="向量化和入库批大小")
    parser.add_argument("--chunk-size", type=int, default=512, help="分块目标字符数")
    parser.add_argument("--overlap", type=int, default=50, help="分块重叠字符数")
    parser.add_argument("--min-chunk-size", type=int, default=100, help="最小分块字符数")
    parser.add_argument("--device", default=os.getenv("RAG_EMBEDDER_DEVICE", "cpu"), help="向量化模型设备")
    parser.add_argument("--dry-run", action="store_true", help="只统计文档和分块，不写入数据库")
    parser.add_argument(
        "--env-file",
        default=str(ROOT / "config" / ".env"),
        help="额外加载的 dotenv 文件路径",
    )
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    # 解析后再次回填，让 env 中的默认值在命令行未传时生效。
    if not args.dsn:
        args.dsn = os.getenv("RAG_PGVECTOR_DSN") or os.getenv("SESSION_DB_DSN")
    if args.table_prefix is None:
        args.table_prefix = os.getenv("RAG_TABLE_PREFIX", "rag_v2_")

    docs_dir = (ROOT / args.docs_dir).resolve()
    loader = DocumentLoader()
    router = _build_router(args.chunk_size, args.overlap, args.min_chunk_size)
    documents = loader.load_from_directory(docs_dir, pattern=args.pattern)
    indexed_documents = _build_index_documents(
        documents,
        router,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_chunk_size=args.min_chunk_size,
    )
    chunk_count = sum(len(item.chunks) for item in indexed_documents)

    print(f"文档目录: {docs_dir}")
    print(f"匹配模式: {args.pattern}")
    print(f"文档数: {len(indexed_documents)}")
    print(f"分块数: {chunk_count}")
    print(f"表前缀: {args.table_prefix or '(空，写入 rag_documents/rag_chunks)'}")

    if args.dry_run:
        print("dry-run 已完成，未写入数据库。")
        return
    if not args.dsn:
        raise SystemExit("缺少 PostgreSQL DSN，请传入 --dsn 或在 .env 中配置 RAG_PGVECTOR_DSN。")

    embedder = BGEEmbedder(device=args.device)
    store = PgVectorStore(
        args.dsn,
        embedding_dimension=embedder.dimension,
        table_prefix=args.table_prefix,
    )
    await store.initialize()
    try:
        await _write_index(indexed_documents, embedder, store, batch_size=args.batch_size)
        doc_count = await store.count_documents()
        chunk_total = await store.count_chunks()
        print(f"入库完成：{store.documents_table}={doc_count} 行，{store.chunks_table}={chunk_total} 行")
    finally:
        await store.close()

    print("RAG 索引构建完成。")


class IndexedDocument:
    """离线入库用的文档和分块集合。"""

    def __init__(self, document: Document, chunks: list) -> None:
        """初始化索引文档。"""
        self.document = document
        self.chunks = chunks


def _build_router(chunk_size: int, overlap: int, min_chunk_size: int) -> DocumentRouter:
    """构建与生产配置一致的文档路由器。"""
    table_chunker = TableChunker()
    return DocumentRouter(
        markdown_chunker=MarkdownStructuredChunker(chunk_size, overlap, min_chunk_size),
        pdf_chunker=PDFLayoutAwareChunker(chunk_size, overlap, min_chunk_size, table_chunker=table_chunker),
        table_chunker=table_chunker,
        text_chunker=TextChunker(chunk_size, overlap, min_chunk_size),
    )


def _build_index_documents(
    documents: Iterable[Document],
    router: DocumentRouter,
    *,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> list[IndexedDocument]:
    """加载每个文档的结构化分块，并把版本与切片配置写入 chunk metadata。"""
    indexed: list[IndexedDocument] = []
    for document in documents:
        chunks = router.route_and_chunk(document.source_path)
        for chunk in chunks:
            chunk.metadata.setdefault("doc_title", document.title)
            chunk.metadata.setdefault("source_path", document.source_path)
            chunk.metadata["index_version"] = INDEX_VERSION
            chunk.metadata["embedder"] = EMBEDDER_NAME
            chunk.metadata["chunker"] = CHUNKER_NAME
            chunk.metadata["chunk_size"] = chunk_size
            chunk.metadata["overlap"] = overlap
            chunk.metadata["min_chunk_size"] = min_chunk_size
        indexed.append(IndexedDocument(document=document, chunks=chunks))
    return indexed


async def _write_index(
    indexed_documents: list[IndexedDocument],
    embedder: BGEEmbedder,
    store: PgVectorStore,
    batch_size: int,
) -> None:
    """批量写入文档、分块和向量。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    for indexed in indexed_documents:
        document = indexed.document
        document_metadata = {
            **document.metadata,
            "index_version": INDEX_VERSION,
            "embedder": EMBEDDER_NAME,
            "chunker": CHUNKER_NAME,
        }
        await store.insert_document(
            doc_id=document.doc_id,
            title=document.title,
            content=document.content,
            source_path=document.source_path,
            metadata=document_metadata,
        )

        for chunk_batch in _batched(indexed.chunks, batch_size):
            texts = [chunk.content for chunk in chunk_batch]
            embeddings = embedder.embed_documents(texts, batch_size=batch_size, show_progress=False)
            rows = []
            for chunk, embedding in zip(chunk_batch, embeddings):
                rows.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "content": chunk.content,
                        "embedding": embedding,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "metadata": chunk.metadata,
                    }
                )
            await store.insert_chunks_batch(rows)


def _batched(items: list, batch_size: int) -> Iterable[list]:
    """按固定大小切分列表。"""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


if __name__ == "__main__":
    # Windows 上 psycopg 异步驱动不兼容默认 ProactorEventLoop，需切到 Selector 策略。
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
