import asyncio
import json
import sqlite3
from pathlib import Path

import yaml

from rag_crawler.crawler import MinimalRAGCrawler


def write_config(tmp_path: Path) -> Path:
    config = {
        "output_dir": str(tmp_path / "rag_corpus"),
        "request_delay_seconds": 0,
        "chunk_size": 20,
        "chunk_overlap": 5,
        "sources": [
            {
                "name": "demo_source",
                "enabled": True,
                "category": "banking_policy",
                "list_url": "https://example.com/list/index.html",
                "list_selector": ".list li a",
                "title_selector": "h1",
                "content_selector": ".content",
                "max_pages": 2,
            }
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def test_crawler_initializes_output_directories_and_sqlite_schema(tmp_path):
    crawler = MinimalRAGCrawler(str(write_config(tmp_path)))

    assert crawler.raw_dir.exists()
    assert crawler.chunks_dir.exists()
    assert crawler.db_path.exists()
    assert crawler.logs_dir.exists()

    with sqlite3.connect(crawler.db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    assert {"documents", "chunks"}.issubset(tables)


def test_extract_links_resolves_relative_urls_and_limits_to_selector(tmp_path):
    crawler = MinimalRAGCrawler(str(write_config(tmp_path)))
    html = """
    <ul class="list">
      <li><a href="/detail/a.html">A</a></li>
      <li><a href="detail/b.html">B</a></li>
    </ul>
    <a href="/ignored.html">ignored</a>
    """

    links = crawler.extract_links(html, "https://example.com/list/index.html", ".list li a")

    assert links == [
        "https://example.com/detail/a.html",
        "https://example.com/list/detail/b.html",
    ]


def test_extract_content_uses_configured_selector_and_title(tmp_path):
    crawler = MinimalRAGCrawler(str(write_config(tmp_path)))
    html = """
    <html><body>
      <h1>贷款政策公告</h1>
      <div class="content">
        <p>第一段：支持小微企业融资。</p>
        <p>第二段：优化银行贷款流程。</p>
      </div>
      <nav>导航不应进入正文</nav>
    </body></html>
    """

    extracted = crawler.extract_content(
        html,
        "https://example.com/detail/a.html",
        {
            "title_selector": "h1",
            "content_selector": ".content",
        },
    )

    assert extracted == {
        "title": "贷款政策公告",
        "content": "第一段：支持小微企业融资。\n第二段：优化银行贷款流程。",
    }


def test_crawl_source_saves_raw_html_chunks_and_dedupes_second_run(tmp_path):
    crawler = MinimalRAGCrawler(str(write_config(tmp_path)))

    pages = {
        "https://example.com/list/index.html": """
            <ul class="list">
              <li><a href="/detail/a.html">A</a></li>
            </ul>
        """,
        "https://example.com/detail/a.html": """
            <html><body>
              <h1>金融服务通知</h1>
              <div class="content">支持小微企业融资，优化银行贷款服务，降低融资成本。</div>
            </body></html>
        """,
    }

    async def fake_fetch(url: str, timeout: int = 30):
        return pages[url]

    crawler.fetch_html = fake_fetch
    source_config = crawler.config["sources"][0]

    first_chunks = asyncio.run(crawler.crawl_source(source_config))
    crawler.save_chunks(first_chunks)
    second_chunks = asyncio.run(crawler.crawl_source(source_config))

    assert crawler.stats["new_docs"] == 1
    assert crawler.stats["skipped_duplicates"] == 1
    assert second_chunks == []
    assert len(first_chunks) >= 1
    assert list(crawler.raw_dir.glob("demo_source_*.html"))

    daily_file = next(crawler.chunks_dir.glob("daily_*.jsonl"))
    rows = [json.loads(line) for line in daily_file.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["source"] == "demo_source"
    assert rows[0]["category"] == "banking_policy"
    assert rows[0]["source_url"] == "https://example.com/detail/a.html"
    assert rows[0]["metadata"]["chunk_index"] == 0

    with sqlite3.connect(crawler.db_path) as conn:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    assert doc_count == 1
    assert chunk_count == len(first_chunks)
