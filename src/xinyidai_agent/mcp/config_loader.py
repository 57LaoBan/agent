"""MCP 配置文件加载器。"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

import yaml

from xinyidai_agent.mcp.protocol import McpServersConfig


_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _substitute_env(value: Any) -> Any:
    """递归替换字符串中的 ${ENV_VAR}，未定义变量保留原字面量。"""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), match.group(0)), value)
    if isinstance(value, list):
        return [_substitute_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _substitute_env(item) for key, item in value.items()}
    return value


def load_mcp_servers_config(path: Path | str) -> McpServersConfig:
    """读取并校验 MCP YAML 配置；文件缺失时返回空配置。"""
    config_path = Path(path)
    if not config_path.exists():
        return McpServersConfig(servers=[])

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return McpServersConfig.model_validate(_substitute_env(raw))


def default_config_path() -> Path:
    """返回项目根目录下的默认 MCP 配置路径。"""
    return Path(__file__).resolve().parents[3] / "config" / "mcp_servers.yaml"
