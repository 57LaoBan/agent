"""结构化折叠 reducer，零 LLM 成本。

从 session.py 中的 _fold_turns / _append_recent_turn 逻辑迁出，
作为默认压缩策略。
"""

from __future__ import annotations

from typing import Any


def fold_overflow_turns(
    recent_turns: list[dict[str, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], str]:
    """保留最近 limit 轮，溢出部分折叠为业务事实摘要。

    Returns:
        (保留的轮次窗口, 溢出部分的折叠摘要)
    """
    if len(recent_turns) <= limit:
        return recent_turns, ""

    overflow = recent_turns[:-limit]
    window = recent_turns[-limit:]
    folded = _fold_turns(overflow)
    return window, folded


def _fold_turns(turns: list[dict[str, Any]]) -> str:
    """把溢出的轮次压缩成短摘要（业务事实）。"""
    business_facts: list[str] = []
    for turn in turns:
        scene = turn.get("scene")
        tool_name = turn.get("tool_name")
        business_status = turn.get("business_status")
        if scene or tool_name or business_status:
            business_facts.append(
                ",".join(str(value) for value in [scene, tool_name, business_status] if value)
            )
    return " | ".join(business_facts[-5:])
