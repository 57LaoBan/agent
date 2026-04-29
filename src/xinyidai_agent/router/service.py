from __future__ import annotations

from xinyidai_agent.llm import ChatModel
from xinyidai_agent.protocol import ChatRequest, RouteDecision
from xinyidai_agent.router.model_router import ModelIntentRouter
from xinyidai_agent.router.policy import RoutePolicy
from xinyidai_agent.router.rules import RuleBasedRouter, default_knowledge_route


class ControlledIntentRouter:
    """规则优先、模型补充、策略收口的业务路由器。"""

    def __init__(
        self,
        model: ChatModel | None = None,
        rules: RuleBasedRouter | None = None,
        policy: RoutePolicy | None = None,
    ) -> None:
        self._rules = rules or RuleBasedRouter()
        self._model_router = ModelIntentRouter(model) if model else None
        self._policy = policy or RoutePolicy()

    def route(self, request: ChatRequest) -> RouteDecision:
        rule_route = self._rules.match(request)
        if rule_route is not None:
            return self._policy.apply(request, rule_route)

        if self._model_router is not None:
            return self._policy.apply(request, self._model_router.route(request))

        return self._policy.apply(request, default_knowledge_route(request))
