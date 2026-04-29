from __future__ import annotations

from dataclasses import dataclass

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.capabilities.catalog import default_capability_catalog
from xinyidai_agent.protocol import ChatRequest, RouteDecision, RouteScene


@dataclass(frozen=True)
class CapabilityResolution:
    policy: CapabilityPolicy | None
    source: str
    reason: str
    ambiguous_candidates: list[str]


class CapabilityResolver:
    """从候选路由解析系统允许的业务能力。

    模型提供 scene/raw_intent/slots 语义信号；resolver 只在后端预定义能力池中选择。
    """

    def __init__(self, catalog: list[CapabilityPolicy] | None = None) -> None:
        self._catalog = catalog or default_capability_catalog()

    def resolve(self, request: ChatRequest, route: RouteDecision) -> CapabilityResolution:
        candidates = self._by_scene(route.scene)
        if not candidates:
            return CapabilityResolution(
                policy=self._unknown_policy(),
                source="fallback",
                reason=f"场景 {route.scene} 没有匹配的业务能力，进入追问。",
                ambiguous_candidates=[],
            )

        raw_intent = (route.raw_intent or route.intent or "").strip()
        alias_matches = [
            policy
            for policy in candidates
            if raw_intent and raw_intent in policy.raw_intent_aliases
        ]
        keyword_matches = [
            policy
            for policy in candidates
            if any(keyword in request.user_message for keyword in policy.keywords)
        ]

        matched = self._unique(alias_matches + keyword_matches)
        if len(matched) == 1:
            return CapabilityResolution(
                policy=matched[0],
                source="resolved",
                reason="根据模型原始意图或用户原文命中唯一业务能力。",
                ambiguous_candidates=[],
            )

        if len(matched) > 1:
            return CapabilityResolution(
                policy=None,
                source="ambiguous",
                reason="命中多个业务能力，需要追问用户确认。",
                ambiguous_candidates=[policy.capability_id for policy in matched],
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
            reason="该场景存在多个候选业务能力，但模型原始意图和关键词未能唯一命中。",
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

    def _unique(self, policies: list[CapabilityPolicy]) -> list[CapabilityPolicy]:
        seen: set[str] = set()
        result: list[CapabilityPolicy] = []
        for policy in policies:
            if policy.capability_id in seen:
                continue
            seen.add(policy.capability_id)
            result.append(policy)
        return result
