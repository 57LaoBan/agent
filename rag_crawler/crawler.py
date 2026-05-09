"""
最小可行 RAG 语料爬虫。

用途：从免费公开网站采集企业信用、银行贷款、监管公告、招投标等信息，
生成可被 RAG 系统直接消费的 JSONL chunk 语料。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import yaml
from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:  # pragma: no cover - 仅在未安装可选依赖时兜底
    trafilatura = None


class MinimalRAGCrawler:
    """最小 RAG 爬虫。"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path).resolve()
        with self.config_path.open(encoding="utf-8") as f:
            self.config: dict[str, Any] = yaml.safe_load(f) or {}

        configured_output = Path(self.config["output_dir"])
        if configured_output.is_absolute():
            self.output_dir = configured_output
        else:
            self.output_dir = (self.config_path.parent / configured_output).resolve()

        self.raw_dir = self.output_dir / "raw"
        self.cleaned_dir = self.output_dir / "cleaned"  # 新增：清洗后的 Markdown
        self.chunks_dir = self.output_dir / "chunks"
        self.db_dir = self.output_dir / "db"
        self.logs_dir = self.output_dir / "logs"

        for directory in [self.raw_dir, self.cleaned_dir, self.chunks_dir, self.db_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.db_path = self.db_dir / "corpus.db"
        self._init_logging()
        self._init_db()

        self.stats = {
            "new_docs": 0,
            "skipped_duplicates": 0,
            "failed_urls": 0,
            "new_chunks": 0,
        }

    def _init_logging(self) -> None:
        log_file = self.logs_dir / f"crawler_{datetime.now().strftime('%Y%m%d')}.log"
        self.logger = logging.getLogger(f"rag_crawler.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self.logger.propagate = False

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source TEXT,
                    source_url TEXT UNIQUE,
                    title TEXT,
                    category TEXT,
                    content_hash TEXT,
                    crawl_time TEXT,
                    status TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    chunk_index INTEGER,
                    text TEXT,
                    text_hash TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_url ON documents(source_url)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON documents(content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")

    def _is_duplicate(self, url: str, content_hash: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE source_url = ? OR content_hash = ? LIMIT 1",
                (url, content_hash),
            ).fetchone()
        return row is not None

    def _save_document(self, doc: dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                (doc_id, source, source_url, title, category, content_hash, crawl_time, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc["doc_id"],
                    doc["source"],
                    doc["source_url"],
                    doc["title"],
                    doc["category"],
                    doc["content_hash"],
                    doc["crawl_time"],
                    "success",
                ),
            )

    def _save_chunks_to_db(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, doc_id, chunk_index, text, text_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk["id"],
                        chunk["doc_id"],
                        chunk["metadata"]["chunk_index"],
                        chunk["text"],
                        hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()[:16],
                    )
                    for chunk in chunks
                ],
            )

    async def fetch_html(self, url: str, timeout: int = 30) -> str | None:
        """获取 HTML。"""
        try:
            # 根据配置决定是否使用代理
            use_proxy = self.config.get("use_proxy", False)
            if use_proxy:
                proxy_url = self.config.get("proxy_url")
                client_kwargs = {
                    "timeout": timeout,
                    "follow_redirects": True,
                    "verify": False,
                }
                if proxy_url:
                    client_kwargs["proxy"] = proxy_url
            else:
                # 不使用代理：设置空的环境变量覆盖
                import os
                old_http_proxy = os.environ.get("HTTP_PROXY")
                old_https_proxy = os.environ.get("HTTPS_PROXY")
                os.environ["HTTP_PROXY"] = ""
                os.environ["HTTPS_PROXY"] = ""

                client_kwargs = {
                    "timeout": timeout,
                    "follow_redirects": True,
                    "verify": False,
                }

            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 CompanyRAGCrawler/0.1"
                        )
                    },
                )
                response.raise_for_status()

                # 恢复环境变量
                if not use_proxy:
                    if old_http_proxy:
                        os.environ["HTTP_PROXY"] = old_http_proxy
                    if old_https_proxy:
                        os.environ["HTTPS_PROXY"] = old_https_proxy

                return response.text
        except Exception as exc:  # noqa: BLE001 - 采集任务需记录并继续
            self.logger.error("Failed to fetch %s: %s", url, exc)
            self.stats["failed_urls"] += 1
            return None

    def extract_links(self, html: str, base_url: str, selector: str) -> list[str]:
        """从列表页提取详情页链接。"""
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        seen: set[str] = set()

        for anchor in soup.select(selector):
            href = anchor.get("href")
            if not href:
                continue
            full_url = urljoin(base_url, href.strip())
            if full_url not in seen:
                seen.add(full_url)
                links.append(full_url)

        return links

    def extract_content(
        self,
        html: str,
        url: str,
        source_config: dict[str, Any],
    ) -> dict[str, str] | None:
        """提取标题和正文。优先使用配置选择器，失败时再用 trafilatura。"""
        soup = BeautifulSoup(html, "html.parser")
        title_selector = source_config.get("title_selector", "h1")
        content_selector = source_config.get("content_selector")

        title_elem = soup.select_one(title_selector)
        title = title_elem.get_text(" ", strip=True) if title_elem else ""

        content = ""
        if content_selector:
            content_elem = soup.select_one(content_selector)
            if content_elem:
                for noise in content_elem.select("script, style, nav, footer, header, aside"):
                    noise.decompose()
                content = content_elem.get_text("\n", strip=True)

        if (not content or len(content) < 20) and trafilatura is not None:
            extracted = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )
            content = extracted.strip() if extracted else content

        if not title:
            title = self._fallback_title(soup)

        content = self._normalize_text(content)
        if not content:
            self.logger.warning("No content extracted from %s", url)
            return None

        return {"title": title, "content": content}

    @staticmethod
    def _fallback_title(soup: BeautifulSoup) -> str:
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(" ", strip=True)
        return "Untitled"

    @staticmethod
    def _normalize_text(text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)

    def chunk_text(self, text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
        """按字符切分文本，适配中文最小方案。"""
        chunk_size = int(chunk_size or self.config.get("chunk_size", 800))
        overlap = int(overlap if overlap is not None else self.config.get("chunk_overlap", 100))
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

        normalized = self._normalize_text(text)
        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = start + chunk_size
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(normalized):
                break
            start = end - overlap
        return chunks

    async def crawl_source(self, source_config: dict[str, Any]) -> list[dict[str, Any]]:
        """爬取单个数据源。"""
        source_name = source_config["name"]
        self.logger.info("开始爬取: %s", source_name)

        all_chunks: list[dict[str, Any]] = []
        list_html = await self.fetch_html(source_config["list_url"])
        if not list_html:
            return all_chunks

        detail_urls = self.extract_links(
            list_html,
            source_config["list_url"],
            source_config["list_selector"],
        )[: int(source_config.get("max_pages", 10))]
        self.logger.info("找到 %s 个详情页链接", len(detail_urls))

        delay = float(self.config.get("request_delay_seconds", 2))
        today = datetime.now().strftime("%Y%m%d")

        for index, url in enumerate(detail_urls, 1):
            self.logger.info("[%s/%s] 爬取: %s", index, len(detail_urls), url)
            html = await self.fetch_html(url)
            if not html:
                continue

            content_hash = hashlib.sha256(html.encode("utf-8")).hexdigest()[:16]
            if self._is_duplicate(url, content_hash):
                self.logger.info("跳过重复: %s", url)
                self.stats["skipped_duplicates"] += 1
                continue

            extracted = self.extract_content(html, url, source_config)
            if not extracted:
                self.stats["failed_urls"] += 1
                continue

            doc_id = f"{source_name}_{today}_{index:03d}"
            raw_file = self.raw_dir / f"{doc_id}.html"
            raw_file.write_text(html, encoding="utf-8")

            crawl_time = datetime.now().isoformat()

            # 保存为 Markdown
            md_file = self.cleaned_dir / f"{doc_id}.md"
            md_content = self._generate_markdown(
                doc_id=doc_id,
                source=source_name,
                url=url,
                title=extracted["title"],
                category=source_config["category"],
                crawl_time=crawl_time,
                content=extracted["content"]
            )
            md_file.write_text(md_content, encoding="utf-8")

            # 根据配置决定是否切片
            enable_chunking = self.config.get("enable_chunking", True)
            doc_chunks: list[dict[str, Any]] = []

            if enable_chunking:
                text_chunks = self.chunk_text(extracted["content"])

                for chunk_index, chunk in enumerate(text_chunks):
                    doc_chunks.append(
                        {
                            "id": f"{doc_id}-chunk-{chunk_index:03d}",
                            "doc_id": doc_id,
                            "source": source_name,
                            "source_url": url,
                            "title": extracted["title"],
                            "category": source_config["category"],
                            "crawl_time": crawl_time,
                            "text": chunk,
                            "metadata": {
                                "chunk_index": chunk_index,
                                "total_chunks": len(text_chunks),
                            },
                        }
                    )

                self._save_chunks_to_db(doc_chunks)
                all_chunks.extend(doc_chunks)
                self.stats["new_chunks"] += len(doc_chunks)

            self._save_document(
                {
                    "doc_id": doc_id,
                    "source": source_name,
                    "source_url": url,
                    "title": extracted["title"],
                    "category": source_config["category"],
                    "content_hash": content_hash,
                    "crawl_time": crawl_time,
                }
            )

            self.stats["new_docs"] += 1

            if delay > 0:
                await asyncio.sleep(delay)

        return all_chunks

    def _generate_markdown(
        self,
        doc_id: str,
        source: str,
        url: str,
        title: str,
        category: str,
        crawl_time: str,
        content: str
    ) -> str:
        """生成 Markdown 格式文档"""
        return f"""---
doc_id: {doc_id}
source: {source}
source_url: {url}
title: {title}
category: {category}
crawl_time: {crawl_time}
---

# {title}

{content}
"""

    async def crawl_all(self) -> list[dict[str, Any]]:
        """爬取所有启用的数据源。"""
        all_chunks: list[dict[str, Any]] = []
        for source_config in self.config.get("sources", []):
            if not source_config.get("enabled", True):
                continue
            chunks = await self.crawl_source(source_config)
            all_chunks.extend(chunks)
        return all_chunks

    def save_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """保存 chunks 到每日 JSONL，并追加到全量 JSONL。"""
        today = datetime.now().strftime("%Y%m%d")
        daily_file = self.chunks_dir / f"daily_{today}.jsonl"
        all_file = self.chunks_dir / "all_documents.jsonl"

        with daily_file.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        with all_file.open("a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        self.logger.info("保存到: %s", daily_file)
        self.logger.info("追加到: %s", all_file)

    def print_report(self) -> None:
        """打印采集报告。"""
        report = f"""
============================================================
RAG 语料采集报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
============================================================

采集统计：
  - 新增文档：{self.stats['new_docs']} 篇
  - 新增 chunks：{self.stats['new_chunks']} 条
  - 跳过重复：{self.stats['skipped_duplicates']} 条
  - 失败 URL：{self.stats['failed_urls']} 个

输出目录：
  - 原始 HTML：{self.raw_dir}
  - Chunks：{self.chunks_dir}
  - 数据库：{self.db_path}
  - 日志：{self.logs_dir}

============================================================
"""
        print(report)
        self.logger.info(report)


async def main(config_path: str) -> None:
    crawler = MinimalRAGCrawler(config_path)
    print("开始采集 RAG 语料...")
    chunks = await crawler.crawl_all()
    if chunks:
        crawler.save_chunks(chunks)
    crawler.print_report()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="最小可行 RAG 语料爬虫")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="配置文件路径，默认使用 rag_crawler/config.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.config))
