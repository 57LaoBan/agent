"""MCP 协议层契约测试：模型冻结、禁止额外字段、默认值稳定。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from xinyidai_agent.mcp.protocol import (
    McpServerConfig,
    McpServersConfig,
    McpToolDescriptor,
    build_qualified_name,
)


class TestMcpServerConfig:
    """MCP Server 配置模型的边界契约。"""

    def test_default_risk_level_is_read_only(self) -> None:
        """未显式声明风险时必须按只读工具处理。"""
        cfg = McpServerConfig(name="x", command="python")
        assert cfg.risk_level == "read_only"
        assert cfg.transport == "stdio"
        assert cfg.enabled is True

    def test_extra_fields_forbidden(self) -> None:
        """配置文件出现未知字段时必须失败，避免静默忽略治理参数。"""
        with pytest.raises(ValidationError):
            McpServerConfig(name="x", command="python", unknown="value")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        """配置模型必须不可变，避免运行期被外部代码改写。"""
        cfg = McpServerConfig(name="x", command="python")
        with pytest.raises(ValidationError):
            cfg.name = "y"  # type: ignore[misc]

    def test_server_name_rejects_shell_sensitive_characters(self) -> None:
        """server 名只允许安全字符，确保 qualified name 可控。"""
        with pytest.raises(ValidationError):
            McpServerConfig(name="bad-name", command="python")


class TestServersConfig:
    """整体配置文件模型契约。"""

    def test_empty_servers_default(self) -> None:
        """空配置必须可加载，便于应急关闭全部 MCP。"""
        cfg = McpServersConfig()
        assert cfg.servers == []


class TestQualifiedName:
    """工具名命名空间契约。"""

    def test_format(self) -> None:
        """qualified name 必须稳定为 mcp__server__tool。"""
        assert build_qualified_name("fs", "read_file") == "mcp__fs__read_file"


class TestToolDescriptor:
    """MCP 工具描述模型契约。"""

    def test_minimal_fields(self) -> None:
        """最小描述应带有安全默认值。"""
        desc = McpToolDescriptor(
            server_name="fs",
            tool_name="read_file",
            qualified_name="mcp__fs__read_file",
            description="读取文件",
        )
        assert desc.input_schema == {}
        assert desc.risk_level == "read_only"
