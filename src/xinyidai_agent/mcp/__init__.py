"""MCP（Model Context Protocol）客户端基础设施。"""

from xinyidai_agent.mcp.config_loader import default_config_path, load_mcp_servers_config
from xinyidai_agent.mcp.manager import McpManager
from xinyidai_agent.mcp.protocol import (
    McpServerConfig,
    McpServersConfig,
    McpToolCallResult,
    McpToolDescriptor,
)

__all__ = [
    "McpManager",
    "McpServerConfig",
    "McpServersConfig",
    "McpToolCallResult",
    "McpToolDescriptor",
    "default_config_path",
    "load_mcp_servers_config",
]
