"""高风险动作确认流（从根目录 pending_actions.py 迁入）。

兼容桥接：根目录 pending_actions.py 仍可用。
"""

# ruff: noqa: F401
from xinyidai_agent.pending_actions import PendingActionBuilder

__all__ = ["PendingActionBuilder"]
