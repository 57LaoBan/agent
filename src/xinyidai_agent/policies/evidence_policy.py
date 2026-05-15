"""证据边界策略（从根目录 evidence_policy.py 迁入）。

兼容桥接：根目录 evidence_policy.py 仍可用。
"""

# ruff: noqa: F401
from xinyidai_agent.evidence_policy import EvidenceDecision, EvidencePolicy

__all__ = ["EvidenceDecision", "EvidencePolicy"]
