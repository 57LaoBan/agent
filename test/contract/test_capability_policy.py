from __future__ import annotations

from dataclasses import replace

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.capabilities.catalog import default_capability_catalog


EXPECTED_REACT_FIELDS = {
    "knowledge.policy.read": (True, True, 2),
    "credit.limit.read": (False, False, 4),
    "product.terms.read": (False, False, 4),
    "authorization.link.create": (False, False, 4),
    "application.draft.create": (False, False, 6),
    "application.status.read": (True, False, 2),
    "smalltalk.respond": (False, False, 1),
    "unknown.clarify": (False, False, 1),
}


def _catalog_by_id() -> dict[str, CapabilityPolicy]:
    """按 capability_id 返回默认能力目录。"""
    return {policy.capability_id: policy for policy in default_capability_catalog()}


def test_all_capabilities_load_with_required_fields() -> None:
    """全部 8 个 capability 都能加载且 ReAct 字段逐项符合清单。"""
    catalog = _catalog_by_id()

    assert set(catalog) == set(EXPECTED_REACT_FIELDS)
    for capability_id, (fast_path, requires_evidence, max_steps) in EXPECTED_REACT_FIELDS.items():
        policy = catalog[capability_id]
        assert policy.fast_path_eligible is fast_path
        assert policy.requires_evidence is requires_evidence
        assert policy.max_react_steps == max_steps


def test_fast_path_eligible_when_slots_filled() -> None:
    """knowledge.policy.read 没有必填槽位，允许走快速通道。"""
    policy = _catalog_by_id()["knowledge.policy.read"]

    assert policy.is_fast_path_eligible_for_route({})


def test_fast_path_blocked_when_slots_missing() -> None:
    """application.status.read 缺 company_name 时不能走快速通道。"""
    policy = _catalog_by_id()["application.status.read"]

    assert not policy.is_fast_path_eligible_for_route({})
    assert policy.is_fast_path_eligible_for_route({"company_name": "杭州示例科技有限公司"})


def test_fast_path_blocked_when_capability_disabled() -> None:
    """credit.limit.read 显式关闭快速通道后永远返回 False。"""
    policy = _catalog_by_id()["credit.limit.read"]

    assert not policy.is_fast_path_eligible_for_route({"company_name": "杭州示例科技有限公司"})


def test_fast_path_blocked_when_multi_tools() -> None:
    """allowed_tools 长度大于 1 时不允许快速通道。"""
    policy = replace(
        _catalog_by_id()["knowledge.policy.read"],
        allowed_tools=["rag_search", "query_credit_amount"],
    )

    assert not policy.is_fast_path_eligible_for_route({})


def test_fast_path_blocked_when_not_read_only() -> None:
    """非只读能力不允许快速通道。"""
    policy = replace(_catalog_by_id()["knowledge.policy.read"], risk_level="state_create")

    assert not policy.is_fast_path_eligible_for_route({})


def test_capability_max_react_steps_positive() -> None:
    """max_react_steps 必须大于等于 1。"""
    assert all(policy.max_react_steps >= 1 for policy in default_capability_catalog())
