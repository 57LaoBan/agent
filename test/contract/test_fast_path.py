from __future__ import annotations

from dataclasses import replace

import pytest

from xinyidai_agent.capabilities.catalog import default_capability_catalog
from xinyidai_agent.protocol import ChatRequest, RetrievalTrace, RouteDecision, SessionStateSnapshot, SourceDocument
from xinyidai_agent.runtime.fast_path import FastPathRunner
from xinyidai_agent.tools.rag_search import RagSearchTool
from xinyidai_agent.tools.registry import ToolRegistry, default_tool_registry


class CountingAnswerModel:
    """统计模型调用次数并返回固定答案。"""

    def __init__(self) -> None:
        """初始化调用计数。"""
        self.calls = 0

    def complete(self, messages, response_format=None) -> str:
        """生成工具结果答案。"""
        self.calls += 1
        return "根据知识库证据，小微企业可申请信贷支持。"


class SourceRetriever:
    """返回固定 RAG 证据的检索器。"""

    def retrieve(self, query: str, top_k: int):
        """返回一条证据和检索 trace。"""
        sources = [
            SourceDocument(
                source_id="policy-1",
                title="信易贷政策",
                content="小微企业可申请信贷支持。",
                score=0.9,
            )
        ]
        trace = RetrievalTrace(
            query=query,
            top_k=top_k,
            results_count=1,
            retriever_type="static",
            rerank_applied=False,
        )
        return sources, trace


def _capability(capability_id: str):
    """按 ID 读取 capability。"""
    return {policy.capability_id: policy for policy in default_capability_catalog()}[capability_id]


def _session() -> SessionStateSnapshot:
    """构造最小会话快照。"""
    return SessionStateSnapshot(session_id="s-1", updated_at="2026-05-09T00:00:00Z")


def _route(capability_id: str, filled_slots: dict | None = None) -> RouteDecision:
    """根据 capability 构造路由结果。"""
    capability = _capability(capability_id)
    return RouteDecision(
        scene=capability.scene,
        intent=capability.standard_intent,
        confidence=0.95,
        filled_slots=filled_slots or {},
        allowed_tools=capability.allowed_tools,
        allowed_tool_categories=capability.allowed_tool_categories,
        risk_level=capability.risk_level,
        route_reason="测试路由。",
        should_call_tool=bool(capability.allowed_tools),
    )


def test_knowledge_qa_fast_path_succeeds_with_sources() -> None:
    """知识问答有证据时 fast_path 成功并附带引用。"""
    model = CountingAnswerModel()
    registry = ToolRegistry([RagSearchTool(retriever=SourceRetriever())])
    runner = FastPathRunner(registry, model)

    outcome = runner.run(
        ChatRequest(user_message="信易贷适合哪些企业？", top_k=3),
        _route("knowledge.policy.read"),
        _capability("knowledge.policy.read"),
        _session(),
    )

    assert outcome.succeeded
    assert outcome.observation is not None
    assert outcome.observation.kind == "success"
    assert outcome.sources[0].source_id == "policy-1"
    assert "参考来源：信易贷政策" in (outcome.final_answer or "")


def test_knowledge_qa_fast_path_fallback_on_empty() -> None:
    """知识问答无证据时 fast_path 回退 ReAct 多轮。"""
    runner = FastPathRunner(default_tool_registry(), CountingAnswerModel())

    outcome = runner.run(
        ChatRequest(user_message="不存在的政策？"),
        _route("knowledge.policy.read"),
        _capability("knowledge.policy.read"),
        _session(),
    )

    assert not outcome.succeeded
    assert outcome.fallback_reason == "observation_kind_empty"


def test_application_status_fast_path_requires_company_name() -> None:
    """申请状态快速通道必须有 company_name。"""
    runner = FastPathRunner(default_tool_registry(), CountingAnswerModel())

    outcome = runner.run(
        ChatRequest(user_message="查进度"),
        _route("application.status.read"),
        _capability("application.status.read"),
        _session(),
    )

    assert not outcome.succeeded
    assert outcome.fallback_reason == "not_fast_path_eligible"


def test_data_query_capability_not_eligible() -> None:
    """普通额度查询 capability 不允许 fast_path。"""
    runner = FastPathRunner(default_tool_registry(), CountingAnswerModel())

    outcome = runner.run(
        ChatRequest(user_message="查额度"),
        _route("credit.limit.read", {"company_name": "杭州示例科技有限公司"}),
        _capability("credit.limit.read"),
        _session(),
    )

    assert not outcome.succeeded
    assert outcome.fallback_reason == "not_fast_path_eligible"


def test_fast_path_argument_mapping_per_capability() -> None:
    """每个 fast_path capability 都有显式参数映射。"""
    runner = FastPathRunner(default_tool_registry(), CountingAnswerModel())

    knowledge_args = runner._build_arguments(
        _capability("knowledge.policy.read"),
        _route("knowledge.policy.read"),
        ChatRequest(user_message="准入条件", top_k=2),
    )
    status_args = runner._build_arguments(
        _capability("application.status.read"),
        _route("application.status.read", {"company_name": "杭州示例科技有限公司"}),
        ChatRequest(user_message="查进度"),
    )

    assert knowledge_args == {"query": "准入条件", "top_k": 2}
    assert status_args == {"company_name": "杭州示例科技有限公司"}


def test_fast_path_does_not_call_model_for_arguments() -> None:
    """fast_path 构造参数时不调用 LLM。"""
    model = CountingAnswerModel()
    runner = FastPathRunner(default_tool_registry(), model)
    disabled = replace(_capability("credit.limit.read"), fast_path_eligible=True)

    with pytest.raises(ValueError):
        runner._build_arguments(
            disabled,
            _route("credit.limit.read", {"company_name": "杭州示例科技有限公司"}),
            ChatRequest(user_message="查额度"),
        )

    assert model.calls == 0
