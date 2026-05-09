from __future__ import annotations

from dataclasses import dataclass

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.capabilities.catalog import default_capability_catalog
from xinyidai_agent.protocol import RouteDecision, RouteScene


@dataclass(frozen=True)
class CapabilityResolution:
    """capability 解析结果。"""

    policy: CapabilityPolicy | None
    source: str
    reason: str
    ambiguous_candidates: list[str]


class CapabilityResolver:
    """从模型路由结果解析到 capability。

    设计原则：模型只在 catalog 预设的 standard_intent 枚举里选一个，
    resolver 按 (standard_intent, scene) 在 catalog 中精确查找；
    不再用别名表/关键词模糊匹配（那是补丁，无法穷尽模型自由文本）。
    """

    def __init__(self, catalog: list[CapabilityPolicy] | None = None) -> None:
        """初始化时按 standard_intent 建索引，避免每次线性扫描。"""
        self._catalog = catalog or default_capability_catalog()
        self._by_standard_intent: dict[str, list[CapabilityPolicy]] = {}
        for policy in self._catalog:
            self._by_standard_intent.setdefault(policy.standard_intent, []).append(policy)

    def resolve(self, route: RouteDecision) -> CapabilityResolution:
        """按 standard_intent 精确派生 capability。

        匹配优先级：
        1. standard_intent + scene 同时命中且唯一 → 采用；
        2. standard_intent 唯一命中（不要求 scene 一致）→ 采用并记录 source=intent_only；
           同 standard_intent 多 capability 时按 scene 二次过滤；
        3. 都未命中 → 进入 unknown.clarify 追问。
        """
        standard_intent = (route.intent or "").strip()

        candidates = self._by_standard_intent.get(standard_intent, [])
        if not candidates:
            return CapabilityResolution(
                policy=self._unknown_policy(),
                source="fallback",
                reason=(
                    f"standard_intent={standard_intent!r} 不在 catalog 预设范围内，"
                    "进入追问。"
                ),
                ambiguous_candidates=[],
            )

        if len(candidates) == 1:
            policy = candidates[0]
            source = "scene_default" if route.raw_intent and route.raw_intent != standard_intent else "standard_intent"
            if policy.scene != route.scene:
                # scene 与 standard_intent 不一致时按 standard_intent 为准并记录差异，
                # 模型偶尔会把 scene 选错，但 standard_intent 已经强约束就足够。
                return CapabilityResolution(
                    policy=policy,
                    source="standard_intent_only",
                    reason=(
                        f"standard_intent={standard_intent!r} 唯一命中 "
                        f"{policy.capability_id}，模型给出的 scene={route.scene} "
                        f"已被覆写为 {policy.scene}。"
                    ),
                    ambiguous_candidates=[],
                )
            return CapabilityResolution(
                policy=policy,
                source=source,
                reason=(
                    f"standard_intent={standard_intent!r} + scene={route.scene} "
                    f"在 catalog 中唯一命中 {policy.capability_id}。"
                ),
                ambiguous_candidates=[],
            )

        scene_filtered = [policy for policy in candidates if policy.scene == route.scene]
        if len(scene_filtered) == 1:
            return CapabilityResolution(
                policy=scene_filtered[0],
                source="standard_intent_scene",
                reason=(
                    f"standard_intent={standard_intent!r} 在 catalog 多 capability 中"
                    f"按 scene={route.scene} 唯一命中 {scene_filtered[0].capability_id}。"
                ),
                ambiguous_candidates=[],
            )

        return CapabilityResolution(
            policy=None,
            source="ambiguous",
            reason=(
                f"standard_intent={standard_intent!r} 在 catalog 中映射到多个 capability，"
                f"scene={route.scene} 仍无法唯一区分，需要追问用户。"
            ),
            ambiguous_candidates=[policy.capability_id for policy in candidates],
        )

    def _by_scene(self, scene: RouteScene) -> list[CapabilityPolicy]:
        """按 scene 列出所有候选；保留旧接口供其它模块查询。"""
        return [policy for policy in self._catalog if policy.scene == scene]

    def _unknown_policy(self) -> CapabilityPolicy:
        """返回 unknown.clarify 兜底策略。"""
        for policy in self._catalog:
            if policy.capability_id == "unknown.clarify":
                return policy
        return CapabilityPolicy(
            capability_id="unknown.clarify",
            scene="UNKNOWN",
            standard_intent="UNKNOWN",
            description="无法确认用户业务意图时追问。",
        )
