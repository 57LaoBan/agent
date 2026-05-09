from __future__ import annotations

from dataclasses import dataclass

from xinyidai_agent.capabilities.base import CapabilityPolicy


@dataclass(frozen=True)
class CapabilityCatalog:
    """能力目录只读视图，供路由和直达分发器查询。"""

    policies: list[CapabilityPolicy]

    def user_visible_capabilities(self) -> list[CapabilityPolicy]:
        """返回可展示给用户的业务能力，排除闲聊和未知兜底能力。"""
        hidden = {"smalltalk.respond", "unknown.clarify"}
        return [policy for policy in self.policies if policy.capability_id not in hidden]


def default_capability_catalog() -> list[CapabilityPolicy]:
    """系统预设的业务能力目录。

    每个 capability 对应唯一的 standard_intent；模型只能在这些 standard_intent
    枚举里选一个，resolver 直接按 standard_intent 精确派生 capability。
    新增 capability 时同步在 protocol.StandardIntent 中加枚举值即可。
    """
    catalog = [
        CapabilityPolicy(
            capability_id="knowledge.policy.read",
            scene="KNOWLEDGE_QA",
            standard_intent="POLICY_OR_PRODUCT_QA",
            description=(
                "总结性 / 概念性 / 流程性的政策、产品、准入规则、操作流程、"
                "风险标签字典等知识问答；答案是文字描述、清单或步骤，不返回具体数值。"
            ),
            allowed_tools=["rag_search"],
            allowed_tool_categories=["knowledge"],
            risk_level="read_only",
            fast_path_eligible=True,
            requires_evidence=True,
            max_react_steps=2,
        ),
        CapabilityPolicy(
            capability_id="credit.limit.read",
            scene="DATA_QUERY",
            standard_intent="CREDIT_LIMIT_QUERY",
            description=(
                "查询某个企业的具体只读数据，如授信额度、可贷额度、当前风险标签等"
                "企业维度的字段值；需要 company_name。"
            ),
            required_slots=["company_name"],
            allowed_tools=["query_credit_amount"],
            allowed_tool_categories=["data_query"],
            risk_level="read_only",
            fast_path_eligible=False,
            requires_evidence=False,
            max_react_steps=4,
        ),
        CapabilityPolicy(
            capability_id="product.terms.read",
            scene="DATA_QUERY",
            standard_intent="PRODUCT_TERMS_QUERY",
            description=(
                "查询产品维度的具体参数：利率、额度区间、期限、费率等。"
                "对接业务方 Java 查询后端，工具尚未注册时由 ToolRegistry 给出降级提示。"
            ),
            allowed_tools=["query_product_terms"],
            allowed_tool_categories=["data_query"],
            risk_level="read_only",
            fast_path_eligible=False,
            requires_evidence=False,
            max_react_steps=4,
        ),
        CapabilityPolicy(
            capability_id="authorization.link.create",
            scene="AUTHORIZATION",
            standard_intent="CREATE_AUTHORIZATION_LINK",
            description="生成企业授权链接，用于企业授权信用数据查询；执行授权动作。",
            required_slots=["company_name"],
            allowed_tools=["create_authorization_link"],
            allowed_tool_categories=["authorization"],
            risk_level="link_create",
            confirmation_required=True,
            fast_path_eligible=False,
            requires_evidence=False,
            max_react_steps=4,
        ),
        CapabilityPolicy(
            capability_id="application.draft.create",
            scene="LOAN_APPLY",
            standard_intent="CREATE_APPLICATION",
            description="创建贷款申请草稿，后续需要企业授权后继续提交。",
            required_slots=["company_name", "product_name"],
            allowed_tools=[
                "search_product",
                "create_application",
                "create_authorization_link",
            ],
            allowed_tool_categories=["knowledge", "application", "authorization"],
            risk_level="state_create",
            confirmation_required=True,
            fast_path_eligible=False,
            requires_evidence=False,
            max_react_steps=6,
        ),
        CapabilityPolicy(
            capability_id="application.status.read",
            scene="APPLICATION_STATUS",
            standard_intent="APPLICATION_STATUS_QUERY",
            description="查询贷款申请办理状态。",
            required_slots=["company_name"],
            allowed_tools=["query_application_status"],
            allowed_tool_categories=["status"],
            risk_level="read_only",
            fast_path_eligible=True,
            requires_evidence=False,
            max_react_steps=2,
        ),
        CapabilityPolicy(
            capability_id="smalltalk.respond",
            scene="SMALLTALK",
            standard_intent="SMALLTALK",
            description="普通闲聊或助手能力说明，不开放业务工具。",
            risk_level="read_only",
            fast_path_eligible=False,
            requires_evidence=False,
            max_react_steps=1,
        ),
        CapabilityPolicy(
            capability_id="unknown.clarify",
            scene="UNKNOWN",
            standard_intent="UNKNOWN",
            description="无法确认用户业务意图时追问，禁止派生任何工具。",
            risk_level="read_only",
            fast_path_eligible=False,
            requires_evidence=False,
            max_react_steps=1,
        ),
    ]
    _validate_catalog(catalog)
    return catalog


def _validate_catalog(catalog: list[CapabilityPolicy]) -> None:
    """启动期校验 capability 目录，保证证据约束和步数配置可审计。"""
    for policy in catalog:
        if policy.max_react_steps < 1:
            raise ValueError(f"{policy.capability_id} max_react_steps 必须大于等于 1")
        if policy.requires_evidence and not policy.allowed_tools:
            raise ValueError(f"{policy.capability_id} requires_evidence=True 时必须声明 allowed_tools")
