"""业务状态机 reducer（从根目录 runtime_state.py 迁入）。

兼容桥接：根目录 runtime_state.py 仍可用，re-export 本模块。
"""

# ruff: noqa: F401
from xinyidai_agent.runtime_state import (
    RuntimeStateReducer,
    build_pending_action_hash,
)

__all__ = ["RuntimeStateReducer", "build_pending_action_hash"]
