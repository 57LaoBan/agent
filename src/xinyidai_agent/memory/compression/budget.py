"""token 预算估算，基于 tiktoken。"""

from __future__ import annotations

from typing import Any

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False
    _ENC = None


def estimate_tokens(text: str) -> int:
    """估算文本 token 数。tiktoken 不可用时回退到字符数 / 2。"""
    if _HAS_TIKTOKEN and _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 2)


def estimate_state_tokens(short_summary: str, recent_turns: list[dict[str, Any]]) -> int:
    """估算 STM 部分（摘要 + 最近轮次）占用的 token。"""
    total = estimate_tokens(short_summary)
    for turn in recent_turns:
        total += estimate_tokens(str(turn))
    return total
