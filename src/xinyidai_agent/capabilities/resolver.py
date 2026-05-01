from __future__ import annotations

from dataclasses import dataclass

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.capabilities.catalog import default_capability_catalog
from xinyidai_agent.protocol import RouteDecision, RouteScene


@dataclass(frozen=True)
class CapabilityResolution:
    policy: CapabilityPolicy | None
    source: str
    reason: str
    ambiguous_candidates: list[str]


class CapabilityResolver:
    """从候选路由解析系统允许的业务能力。

    模型负责判断 scene；resolver 只按 scene 在后端预定义能力池中收口到允许的业务能力。
    """

    def __init__(self, catalog: list[CapabilityPolicy] | None = None) -> None:
        self._catalog = catalog or default_capability_catalog()

    def resolve(self, route: RouteDecision) -> CapabilityResolution:
        candidates = self._by_scene(route.scene)
        if not candidates:
            return CapabilityResolution(
                policy=self._unknown_policy(),
                source="fallback",
                reason=f"场景 {route.scene} 没有匹配的业务能力，进入追问。",
                ambiguous_candidates=[],
            )

        if len(candidates) == 1:
            return CapabilityResolution(
                policy=candidates[0],
                source="scene_default",
                reason="该场景只有一个候选业务能力，按场景默认能力处理。",
                ambiguous_candidates=[],
            )

        return CapabilityResolution(
            policy=None,
            source="ambiguous",
            reason="该场景存在多个候选业务能力，需要用户进一步确认。",
            ambiguous_candidates=[policy.capability_id for policy in candidates],
        )

    def _by_scene(self, scene: RouteScene) -> list[CapabilityPolicy]:
        return [policy for policy in self._catalog if policy.scene == scene]

    def _unknown_policy(self) -> CapabilityPolicy:
        for policy in self._catalog:
            if policy.capability_id == "unknown.clarify":
                return policy
        return CapabilityPolicy(
            capability_id="unknown.clarify",
            scene="UNKNOWN",
            standard_intent="UNKNOWN",
            description="无法确认用户业务意图时追问。",
        )
