"""ReAct observation 容器（从根目录 react_observation.py 迁入）。

兼容桥接：根目录 react_observation.py 仍可用。
"""

# ruff: noqa: F401
from xinyidai_agent.react_observation import ReactObservation, ToolSpecBrief

__all__ = ["ReactObservation", "ToolSpecBrief"]
