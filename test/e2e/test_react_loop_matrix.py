from __future__ import annotations

import pytest

from xinyidai_agent.capabilities.catalog import CapabilityCatalog, default_capability_catalog
from xinyidai_agent.memory import MemoryManager
from xinyidai_agent.protocol import (
    ChatRequest,
    RetrievalTrace,
    RouteDecision,
    SourceDocument,
    ToolCall,
    ToolResult,
)
from xinyidai_agent.router import ControlledIntentRouter
from xinyidai_agent.router.rules_guard import RulesGuard
from xinyidai_agent.router.scene_direct import SceneDirectDispatcher
from xinyidai_agent.runtime import ControlledAgentLoop
from xinyidai_agent.tools.base import SlotSpec, ToolExecution, ToolSpec
from xinyidai_agent.tools.mock_credit import MockCreditAmountTool
from xinyidai_agent.tools.rag_search import RagSearchTool
from xinyidai_agent.tools.registry import ToolRegistry


class MatrixAnswerModel:
    """端到端矩阵使用的确定性模型。"""

    def complete(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | None = None,
    ) -> str:
        """系统记忆工具返回空操作，普通回答基于工具上下文生成。"""
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        if "update_session_state" in system:
            return '{"tool_name":"update_session_state","arguments":{"operations":[]}}'
        if response_format is not None:
            return '{"thought":"matrix","action":{"type":"handoff","message":"需要人工处理"},"confidence":0.5}'
        if "50" in user:
            return "根据工具结果，该企业当前可用授信额度为50万元。"
        if "sources" in user or "证据" in user:
            return "根据检索证据，信易贷支持符合条件的小微企业办理融资相关业务。"
        return "已处理。"


class MatrixRetriever:
    """返回固定证据的检索器。"""

    def retrieve(self, query: str, top_k: int):
        """返回与 query 对齐的静态证据和 trace。"""
        sources = [
            SourceDocument(
                source_id="matrix-policy-1",
                title="信易贷准入规则",
                content="信易贷面向符合条件的小微企业提供融资撮合、政策查询和授权相关服务。",
                source_type="policy",
                score=0.91,
            )
        ]
        trace = RetrievalTrace(
            query=query,
            top_k=top_k,
            results_count=len(sources),
            retriever_type="matrix",
            index_version="test",
            steps=[{"name": "matrix_retrieve"}],
        )
        return sources, trace


class CreateAuthorizationLinkTool:
    """授权链接创建工具声明，用于验证确认前不执行状态变更。"""

    name = "create_authorization_link"
    category = "authorization"
    risk_level = "link_create"
    description = "生成企业授权链接。"
    requires_confirmation = True

    def spec(self) -> ToolSpec:
        """返回需要确认的工具规格。"""
        return ToolSpec(
            name=self.name,
            category=self.category,
            risk_level=self.risk_level,
            description=self.description,
            requires_confirmation=self.requires_confirmation,
            input_slots=[SlotSpec("company_name", "string")],
            output_slots=[SlotSpec("authorization_id", "string")],
            is_read_only=False,
            is_idempotent=False,
            is_concurrency_safe=True,
            cost_class="medium",
            max_duration_ms=3000,
            input_schema={"company_name": "企业名称"},
        )

    def execute(self, request: ChatRequest, route: RouteDecision, tool_call: ToolCall) -> ToolExecution:
        """确认前不应执行；若执行也只返回内部 ID，不生成 URL。"""
        return ToolExecution(
            result=ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=self.name,
                tool_category=self.category,
                status="success",
                business_status="OK",
                output={"authorization_id": "AUTH-TEST"},
                terminal=True,
            )
        )


class MatrixRouter:
    """覆盖 Phase 9 查询矩阵的确定性路由器。"""

    def __init__(self) -> None:
        """初始化 L0/L1 短路委托。"""
        catalog = CapabilityCatalog(default_capability_catalog())
        self._pre = ControlledIntentRouter(
            model=None,
            guard=RulesGuard(),
            scene_direct=SceneDirectDispatcher(catalog),
        )

    def pre_route(self, request: ChatRequest, *, has_pending_action: bool):
        """复用生产 L0/L1 规则。"""
        return self._pre.pre_route(request, has_pending_action=has_pending_action)

    def route(self, request: ChatRequest) -> RouteDecision:
        """根据测试 query 返回固定业务路由。"""
        message = request.user_message
        if "授权" in message:
            return RouteDecision(
                scene="AUTHORIZATION",
                intent="CREATE_AUTHORIZATION_LINK",
                capability_id="authorization.link.create",
                confirmation_required=True,
                confidence=0.94,
                required_slots=["company_name"],
                filled_slots={"company_name": "ABC 公司"},
                allowed_tools=["create_authorization_link"],
                allowed_tool_categories=["authorization"],
                risk_level="link_create",
                route_reason="测试授权链接确认流。",
                route_source="matrix",
                should_call_tool=True,
            )
        if "授信" in message or "能贷" in message:
            return RouteDecision(
                scene="DATA_QUERY",
                intent="CREDIT_LIMIT_QUERY",
                capability_id="credit.limit.read",
                confidence=0.93,
                required_slots=["company_name"],
                filled_slots={"company_name": "ABC 公司"},
                allowed_tools=["query_credit_amount"],
                allowed_tool_categories=["data_query"],
                risk_level="read_only",
                route_reason="测试授信额度查询。",
                route_source="matrix",
                should_call_tool=True,
            )
        if "利率" in message:
            return RouteDecision(
                scene="DATA_QUERY",
                intent="PRODUCT_TERMS_QUERY",
                capability_id="product.terms.read",
                confidence=0.9,
                allowed_tools=["query_product_terms"],
                allowed_tool_categories=["data_query"],
                risk_level="read_only",
                route_reason="测试未接入产品参数工具。",
                route_source="matrix",
                should_call_tool=True,
            )
        return RouteDecision(
            scene="KNOWLEDGE_QA",
            intent="POLICY_OR_PRODUCT_QA",
            capability_id="knowledge.policy.read",
            confidence=0.92,
            allowed_tools=["rag_search"],
            allowed_tool_categories=["knowledge"],
            risk_level="read_only",
            route_reason="测试知识问答。",
            route_source="matrix",
            should_call_tool=True,
        )


def build_matrix_agent() -> ControlledAgentLoop:
    """构建端到端矩阵 Agent。"""
    registry = ToolRegistry(
        [
            MockCreditAmountTool(),
            RagSearchTool(retriever=MatrixRetriever()),
            CreateAuthorizationLinkTool(),
        ]
    )
    return ControlledAgentLoop(
        model=MatrixAnswerModel(),
        router=MatrixRouter(),
        tool_registry=registry,
        memory_manager=MemoryManager(),
    )


@pytest.fixture()
def agent() -> ControlledAgentLoop:
    """提供端到端矩阵 Agent fixture。"""
    return build_matrix_agent()


CASES: list[tuple[str, str, str | tuple[str, ...], bool, str]] = [
    ("empty", "", "rules_guard_rejected", False, "L0"),
    ("punct", "。。。。", "rules_guard_rejected", False, "L0"),
    ("intro", "你是谁", "answered_without_tool", False, "L1"),
    ("cap_list", "你能办什么业务", "answered_without_tool", False, "L1"),
    ("greeting", "你好", "answered_without_tool", False, "L1"),
    ("thanks", "谢谢", "answered_without_tool", False, "L1"),
    ("policy_qa", "信易贷的准入条件是什么", "completed", True, "L3-fast"),
    ("risk_tags", "高风险标签都包括哪些情况", "completed", True, "L3-fast"),
    ("credit_amt", "查一下 ABC 公司的授信额度", "completed", False, "L3-react"),
    ("rate", "小微税贷的利率是多少", ("answered_without_tool", "missing_slots", "completed"), False, "L3-react"),
    ("auth_link", "帮我给 ABC 公司生成授权链接", "waiting_confirmation", False, "L3-react"),
]


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_react_loop_matrix(case, agent: ControlledAgentLoop) -> None:
    """覆盖 L0/L1/L3-fast/L3-react 的端到端查询矩阵。"""
    case_id, message, expected, has_sources, expected_path = case
    response = agent.answer(ChatRequest(user_message=message, session_id=f"e2e_{case_id}"))

    if isinstance(expected, tuple):
        assert response.stop_reason in expected, f"{case_id}: stop_reason={response.stop_reason}"
    else:
        assert response.stop_reason == expected, f"{case_id}: stop_reason={response.stop_reason}"

    if has_sources:
        assert len(response.sources) >= 1, f"{case_id}: 缺少证据"

    event_types = [event.event_type for event in response.events]
    assert "final_answer" in event_types
    assert event_types[-1] == "turn_finished"

    if expected_path == "L0":
        assert "route_decision" not in event_types
    if expected_path == "L1":
        assert "route_decision" not in event_types
    if expected_path == "L3-fast":
        assert response.tool_trace and response.tool_trace[0].tool_name == "rag_search"
    if case_id == "rate":
        fabricated = any(token in response.answer for token in ("3.5%", "4.0%", "4.5%", "5.0%", "年化"))
        disclaimer = any(token in response.answer for token in ("未上线", "暂未", "无法获取", "请咨询", "联系"))
        if fabricated:
            assert disclaimer, f"rate 用例疑似幻觉数值：{response.answer}"
