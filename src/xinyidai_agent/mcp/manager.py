"""MCP 多 Server 生命周期管理器。"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from xinyidai_agent.mcp.client import McpClient
from xinyidai_agent.mcp.protocol import McpServerConfig, McpToolCallResult, McpToolDescriptor
from xinyidai_agent.rag.async_runtime import AsyncRuntime


logger = logging.getLogger(__name__)


class McpManager:
    """统一管理多个 MCP Server 的同步外观。"""

    def __init__(self, configs: list[McpServerConfig], async_runtime: AsyncRuntime) -> None:
        """保存配置与异步运行时引用。"""
        self._configs = [config for config in configs if config.enabled]
        self._runtime = async_runtime
        self._clients: dict[str, McpClient] = {}
        self._descriptors: dict[str, McpToolDescriptor] = {}
        self._call_lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        """启动所有 enabled Server，并缓存可用工具描述。"""
        if self._started:
            return
        self._runtime.run_coroutine(self._start_async())
        self._started = True

    def shutdown(self) -> None:
        """关闭所有 MCP Client。"""
        if not self._started:
            return
        self._runtime.run_coroutine(self._shutdown_async())
        self._started = False

    def iter_descriptors(self) -> list[McpToolDescriptor]:
        """返回当前已注册工具描述的拷贝。"""
        return list(self._descriptors.values())

    def call_tool_sync(self, qualified_name: str, arguments: dict[str, Any]) -> McpToolCallResult:
        """同步调用 MCP 工具，并将执行提交到 AsyncRuntime。"""
        if not self._started:
            raise RuntimeError("McpManager 尚未 start()")
        descriptor = self._descriptors.get(qualified_name)
        if descriptor is None:
            raise KeyError(f"未注册的 MCP 工具：{qualified_name}")
        client = self._clients.get(descriptor.server_name)
        if client is None:
            raise KeyError(f"Server 未就绪：{descriptor.server_name}")
        with self._call_lock:
            # 同步等待时长以 client 配置的 timeout 为准，外加 1 秒缓冲覆盖队列调度耗时，
            # 避免后台 worker 抛 TimeoutError 之前外层就已超时返回。
            return self._runtime.run_coroutine(
                client.call_tool(descriptor.tool_name, arguments),
                timeout=client.timeout_seconds + 1.0,
            )

    async def _start_async(self) -> None:
        """并发拉起所有 Server，单个 Server 失败不影响其它 Server。"""
        await asyncio.gather(*(self._start_single(config) for config in self._configs))

    async def _start_single(self, config: McpServerConfig) -> None:
        """启动单个 Server 并缓存其工具描述符。"""
        client = McpClient(config)
        try:
            await client.connect()
            descriptors = await client.list_tools()
        except (OSError, RuntimeError, ValueError, TimeoutError, asyncio.TimeoutError) as exc:
            logger.warning("MCP Server %s 启动失败，跳过该 Server：%s", config.name, exc)
            await client.aclose()
            return
        self._clients[config.name] = client
        for descriptor in descriptors:
            self._descriptors[descriptor.qualified_name] = descriptor
        logger.info("MCP Server %s 已就绪，注册 %d 个工具", config.name, len(descriptors))

    async def _shutdown_async(self) -> None:
        """逐个关闭所有 Client。"""
        for name, client in list(self._clients.items()):
            try:
                await client.aclose()
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning("MCP Server %s 关闭异常：%s", name, exc)
        self._clients.clear()
        self._descriptors.clear()
