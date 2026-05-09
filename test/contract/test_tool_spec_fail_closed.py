from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from xinyidai_agent.tools.base import SlotSpec, ToolSpec
from xinyidai_agent.tools.registry import default_tool_registry


def test_tool_spec_defaults_are_fail_closed() -> None:
    """ToolSpec 默认运行属性必须全部偏拒绝。"""
    spec = ToolSpec(name="x", category="utility", risk_level="read_only", description="")

    assert spec.is_read_only is False
    assert spec.is_idempotent is False
    assert spec.is_concurrency_safe is False
    assert spec.cost_class == "medium"
    assert spec.max_duration_ms == 5000


def test_registered_tools_declare_metadata_explicitly() -> None:
    """当前默认注册工具都是只读、幂等、并发安全的显式声明。"""
    registry = default_tool_registry()
    specs = registry.specs_for_categories(["knowledge", "data_query"])

    assert {spec.name for spec in specs} == {"rag_search", "query_credit_amount"}
    for spec in specs:
        assert spec.is_read_only is True
        assert spec.is_idempotent is True
        assert spec.is_concurrency_safe is True
        assert spec.max_duration_ms > 0


def test_required_argument_names_filters_optional() -> None:
    """required_argument_names 只返回 required=True 的入参。"""
    spec = ToolSpec(
        name="x",
        category="utility",
        risk_level="read_only",
        description="",
        input_slots=[
            SlotSpec(name="company_name", value_type="string", required=True),
            SlotSpec(name="top_k", value_type="integer", required=False),
        ],
    )

    assert spec.required_argument_names() == ["company_name"]


def test_tool_spec_is_frozen() -> None:
    """ToolSpec 是 frozen dataclass，运行时不能被改写。"""
    spec = ToolSpec(name="x", category="utility", risk_level="read_only", description="")

    with pytest.raises(FrozenInstanceError):
        spec.is_read_only = True
