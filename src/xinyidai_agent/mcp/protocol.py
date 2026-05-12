"""MCP 协议层数据模型。

所有模型都使用 frozen=True、extra=forbid，作为 MCP 基础设施边界的稳定契约。
当前仅支持 stdio transport，Resources / Prompts 留给后续独立扩展。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from xinyidai_agent.protocol import RiskLevel


McpTransport = Literal["stdio"]


class McpServerConfig(BaseModel):
    """单个 MCP Server 的启动与治理配置。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    transport: McpTransport = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = "read_only"
    description: str = ""
    timeout_seconds: float = Field(default=30.0, gt=0)
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """限制 server 名称，避免 qualified_name 出现不可控字符。"""
        if not value or not value.replace("_", "").isalnum() or value.lower() != value:
            raise ValueError("MCP server name 仅允许小写字母、数字和下划线")
        return value


class McpServersConfig(BaseModel):
    """整体 MCP 配置文件根模型。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    servers: list[McpServerConfig] = Field(default_factory=list)


class McpToolDescriptor(BaseModel):
    """从 Server 拉取并规范化后的工具描述。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    server_name: str
    tool_name: str
    qualified_name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel = "read_only"


class McpToolCallResult(BaseModel):
    """MCP Server 工具调用结果的本地规范化形态。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_error: bool
    text_content: str = ""
    structured: dict[str, Any] = Field(default_factory=dict)
    raw_content: list[dict[str, Any]] = Field(default_factory=list)


def build_qualified_name(server_name: str, tool_name: str) -> str:
    """统一拼接 qualified_name：mcp__{server}__{tool}。"""
    return f"mcp__{server_name}__{tool_name}"
