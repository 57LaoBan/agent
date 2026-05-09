from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from xinyidai_agent.llm import ChatModel
from xinyidai_agent.protocol import ChatRequest, RouteDecision
from xinyidai_agent.router.model_router import ModelIntentRouter
from xinyidai_agent.router.policy import RoutePolicy
from xinyidai_agent.router.rules import RuleBasedRouter, default_knowledge_route
from xinyidai_agent.router.rules_guard import RulesGuard
from xinyidai_agent.router.scene_direct import SceneDirectDispatcher


class PreRouteOutcome(BaseModel):
    """L0 + L1 短路输出，由 runtime 决定直接返回或继续路由。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["guard_reject", "scene_direct", "continue"]
    final_answer: str | None = None
    stop_reason: str | None = None
    handler_name: str | None = None


class ControlledIntentRouter:
    """本地守卫、模型判别、策略收口的业务路由器。"""

    def __init__(
        self,
        model: ChatModel | None = None,
        rules: RuleBasedRouter | None = None,
        policy: RoutePolicy | None = None,
        guard: RulesGuard | None = None,
        scene_direct: SceneDirectDispatcher | None = None,
    ) -> None:
        self._rules = rules or RuleBasedRouter()
        self._model_router = ModelIntentRouter(model) if model else None
        self._policy = policy or RoutePolicy()
        self._guard = guard or RulesGuard()
        self._scene_direct = scene_direct

    def pre_route(
        self,
        request: ChatRequest,
        *,
        has_pending_action: bool,
    ) -> PreRouteOutcome:
        """执行 L0 + L1 前置短路。"""
        guard_result = self._guard.guard(request, has_pending_action=has_pending_action)
        if guard_result.kind in {"reject_empty", "reject_invalid"}:
            return PreRouteOutcome(
                kind="guard_reject",
                final_answer=guard_result.user_visible_message,
                stop_reason="rules_guard_rejected",
                handler_name=guard_result.reason,
            )
        if guard_result.kind == "continuation_confirm":
            return PreRouteOutcome(kind="continue", handler_name="continuation_confirm")
        if self._scene_direct is not None:
            hit = self._scene_direct.dispatch(request)
            if hit is not None:
                return PreRouteOutcome(
                    kind="scene_direct",
                    final_answer=hit.final_answer,
                    stop_reason=hit.stop_reason,
                    handler_name=hit.handler_name,
                )
        return PreRouteOutcome(kind="continue")

    def route(self, request: ChatRequest) -> RouteDecision:
        rule_route = self._rules.match(request)
        if rule_route is not None:
            return self._policy.apply(request, rule_route)

        if self._model_router is not None:
            return self._policy.apply(request, self._model_router.route(request))

        return self._policy.apply(request, default_knowledge_route(request))
