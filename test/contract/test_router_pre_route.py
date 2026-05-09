from __future__ import annotations

from xinyidai_agent.capabilities.catalog import CapabilityCatalog, default_capability_catalog
from xinyidai_agent.protocol import ChatRequest
from xinyidai_agent.router.scene_direct import SceneDirectDispatcher
from xinyidai_agent.router.service import ControlledIntentRouter


class CountingModel:
    """统计模型调用次数的测试模型。"""

    def __init__(self) -> None:
        """初始化调用计数。"""
        self.calls = 0

    def complete(self, messages, response_format=None) -> str:
        """如果 pre_route 生效，该方法不应被调用。"""
        self.calls += 1
        return '{"scene":"UNKNOWN","standard_intent":"UNKNOWN","confidence":0.1,"route_reason":"x"}'


def _router(model: CountingModel | None = None) -> ControlledIntentRouter:
    """构造带 L1 直达分发器的路由器。"""
    catalog = CapabilityCatalog(default_capability_catalog())
    return ControlledIntentRouter(
        model=model,
        scene_direct=SceneDirectDispatcher(catalog),
    )


def test_empty_message_rejected_by_guard() -> None:
    """空消息由 L0 直接拒绝。"""
    outcome = _router().pre_route(ChatRequest(user_message="   "), has_pending_action=False)

    assert outcome.kind == "guard_reject"
    assert outcome.stop_reason == "rules_guard_rejected"
    assert "问题" in (outcome.final_answer or "")


def test_pure_punctuation_rejected_by_guard() -> None:
    """纯标点由 L0 直接拒绝。"""
    outcome = _router().pre_route(ChatRequest(user_message="？？！！"), has_pending_action=False)

    assert outcome.kind == "guard_reject"
    assert outcome.handler_name == "no_chinese_or_letter"


def test_confirmation_reply_passes_to_continue() -> None:
    """有 pending_action 时确认回复继续交给 runtime 处理。"""
    outcome = _router().pre_route(ChatRequest(user_message="确认"), has_pending_action=True)

    assert outcome.kind == "continue"
    assert outcome.handler_name == "continuation_confirm"


def test_introduction_hits_scene_direct() -> None:
    """身份介绍命中 L1 直达且不调用模型。"""
    model = CountingModel()
    outcome = _router(model).pre_route(ChatRequest(user_message="你是谁？"), has_pending_action=False)

    assert outcome.kind == "scene_direct"
    assert outcome.handler_name == "introduction"
    assert "信易贷智能助手" in (outcome.final_answer or "")
    assert model.calls == 0


def test_capability_list_hits_scene_direct() -> None:
    """能力清单命中 L1 直达。"""
    outcome = _router().pre_route(ChatRequest(user_message="你支持哪些业务"), has_pending_action=False)

    assert outcome.kind == "scene_direct"
    assert outcome.handler_name == "capability_list"
    assert "业务能力" in (outcome.final_answer or "")


def test_greeting_hits_scene_direct() -> None:
    """问候命中 L1 直达。"""
    outcome = _router().pre_route(ChatRequest(user_message="你好"), has_pending_action=False)

    assert outcome.kind == "scene_direct"
    assert outcome.handler_name == "greeting"


def test_thanks_hits_scene_direct() -> None:
    """感谢命中 L1 直达。"""
    outcome = _router().pre_route(ChatRequest(user_message="谢谢"), has_pending_action=False)

    assert outcome.kind == "scene_direct"
    assert outcome.handler_name == "thanks"
