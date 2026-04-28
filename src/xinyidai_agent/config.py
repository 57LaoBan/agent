from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
