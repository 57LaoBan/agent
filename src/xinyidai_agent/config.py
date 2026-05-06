from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


def load_env_file(path: Path) -> None:
    """读取简单 key=value 配置，不额外引入 python-dotenv 依赖。"""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class AgentConfig:
    llm_base_url: str = DEFAULT_BASE_URL
    llm_api_key: str = ""
    llm_model: str = DEFAULT_MODEL
    request_timeout_seconds: float = 60.0

    @staticmethod
    def default_env_path() -> Path:
        return Path(__file__).resolve().parents[2] / "config" / ".env"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        api_key = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
        model = os.getenv("LLM_MODEL") or os.getenv("DASHSCOPE_MODEL", DEFAULT_MODEL)

        return cls(
            llm_base_url=base_url,
            llm_api_key=api_key,
            llm_model=model,
            request_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        )


@dataclass(frozen=True)
class SessionStorageConfig:
    """会话持久化存储配置。"""

    backend: str = "postgres"
    dsn: str = "postgresql://xinyidai:xinyidai123@localhost:5432/xinyidai"

    @classmethod
    def from_env(cls) -> "SessionStorageConfig":
        """从环境变量读取会话存储配置，并兼容 localhost:5432 形式的输入。"""
        backend = os.getenv("SESSION_STORAGE_BACKEND", "postgres").strip().lower()
        raw_dsn = (
            os.getenv("SESSION_DB_DSN")
            or os.getenv("SESSION_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or ""
        ).strip()
        if raw_dsn:
            dsn = _normalize_postgres_dsn(raw_dsn)
        else:
            dsn = _build_default_postgres_dsn()
        return cls(backend=backend, dsn=dsn)


def _build_default_postgres_dsn() -> str:
    """根据本地 Docker 默认参数拼出 PostgreSQL DSN。"""
    host = os.getenv("SESSION_DB_HOST", "localhost")
    port = os.getenv("SESSION_DB_PORT", "5432")
    database = os.getenv("SESSION_DB_NAME", "xinyidai")
    user = os.getenv("SESSION_DB_USER", "xinyidai")
    password = os.getenv("SESSION_DB_PASSWORD", "xinyidai123")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def _normalize_postgres_dsn(raw_dsn: str) -> str:
    """把 http://localhost:5432/ 这类输入转换为 psycopg 可用的 PostgreSQL DSN。"""
    if raw_dsn.startswith(("postgresql://", "postgres://")):
        return raw_dsn
    if raw_dsn.startswith(("http://", "https://")):
        parsed = urlparse(raw_dsn)
        host = parsed.hostname or os.getenv("SESSION_DB_HOST", "localhost")
        port = parsed.port or int(os.getenv("SESSION_DB_PORT", "5432"))
        database = parsed.path.strip("/") or os.getenv("SESSION_DB_NAME", "xinyidai")
        user = os.getenv("SESSION_DB_USER", "xinyidai")
        password = os.getenv("SESSION_DB_PASSWORD", "xinyidai123")
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return raw_dsn
