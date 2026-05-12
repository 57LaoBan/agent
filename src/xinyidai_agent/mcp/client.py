"""MCP stdio 客户端封装。"""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Literal

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from xinyidai_agent.mcp.protocol import (
    McpServerConfig,
    McpToolCallResult,
    McpToolDescriptor,
    build_qualified_name,
)


_RequestKind = Literal["list_tools", "call_tool", "close"]


@dataclass(frozen=True)
class _ClientRequest:
    """提交给单 server worker task 的串行请求。"""

    kind: _RequestKind
    future: asyncio.Future[Any]
    tool_name: str = ""
    arguments: dict[str, Any] | None = None


class McpClient:
    """单个 MCP Server 的 stdio 客户端。

    MCP SDK 的 stdio context 需要在同一个 asyncio task 内 enter/exit。
    因此本客户端为每个 server 启动一个常驻 worker task，所有 MCP 调用通过
    队列提交到该 task 内串行执行，避免跨 task 关闭导致子进程泄漏。
    """

    def __init__(self, config: McpServerConfig) -> None:
        """保存配置，延迟到 connect 时拉起子进程。"""
        self._config = config
        self._queue: asyncio.Queue[_ClientRequest] | None = None
        self._task: asyncio.Task[None] | None = None
        self._connected = False

    @property
    def server_name(self) -> str:
        """返回 Server 标识。"""
        return self._config.name

    @property
    def timeout_seconds(self) -> float:
        """暴露当前 Server 的调用超时时长，供上层调度同步等待。"""
        return self._config.timeout_seconds

    async def connect(self) -> None:
        """拉起子进程并完成 MCP initialize 握手。"""
        if self._connected:
            return
        if self._config.transport != "stdio":
            raise NotImplementedError(f"暂不支持的 transport: {self._config.transport}")
        if not self._config.command:
            raise ValueError(f"Server {self._config.name} 缺少 command 字段")

        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run_server(ready))
        await asyncio.wait_for(ready, timeout=self._config.timeout_seconds)
        self._connected = True

    async def list_tools(self) -> list[McpToolDescriptor]:
        """拉取工具清单，并按 allowed_tools 白名单过滤。"""
        response = await self._submit("list_tools")
        allowed = set(self._config.allowed_tools)
        descriptors: list[McpToolDescriptor] = []
        for tool in response.tools:
            if allowed and tool.name not in allowed:
                continue
            schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
            descriptors.append(
                McpToolDescriptor(
                    server_name=self._config.name,
                    tool_name=tool.name,
                    qualified_name=build_qualified_name(self._config.name, tool.name),
                    description=self._compose_description(tool.description or ""),
                    input_schema=dict(schema),
                    risk_level=self._config.risk_level,
                )
            )
        return descriptors

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> McpToolCallResult:
        """调用 MCP tools/call，并规范化返回结构。"""
        response = await self._submit("call_tool", tool_name=tool_name, arguments=arguments or {})
        text_parts: list[str] = []
        raw_content: list[dict[str, Any]] = []
        for item in response.content or []:
            payload = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            raw_content.append(payload)
            if payload.get("type") == "text":
                text_parts.append(str(payload.get("text") or ""))

        structured: dict[str, Any] = {}
        structured_payload = getattr(response, "structuredContent", None) or getattr(
            response,
            "structured_content",
            None,
        )
        if isinstance(structured_payload, dict):
            structured = dict(structured_payload)

        is_error = bool(getattr(response, "isError", False) or getattr(response, "is_error", False))
        return McpToolCallResult(
            is_error=is_error,
            text_content="\n".join(text_parts),
            structured=structured,
            raw_content=raw_content,
        )

    async def aclose(self) -> None:
        """优雅关闭 worker task、MCP session 与子进程。"""
        if not self._connected:
            return
        self._connected = False
        try:
            await self._submit("close")
        finally:
            task = self._task
            self._task = None
            self._queue = None
            if task is not None:
                await asyncio.wait_for(task, timeout=self._config.timeout_seconds)

    async def _run_server(self, ready: asyncio.Future[None]) -> None:
        """在单一 task 内持有 stdio context，并串行处理调用请求。"""
        command = sys.executable if self._config.command == "python" else self._config.command
        params = StdioServerParameters(
            command=command,
            args=list(self._config.args),
            env={**self._config.env} if self._config.env else None,
        )
        try:
            async with AsyncExitStack() as stack:
                read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                await asyncio.wait_for(session.initialize(), timeout=self._config.timeout_seconds)
                ready.set_result(None)
                await self._serve_requests(session)
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                raise

    async def _serve_requests(self, session: ClientSession) -> None:
        """消费请求队列，保证同一个 server 的调用按提交顺序执行。"""
        queue = self._require_queue()
        while True:
            request = await queue.get()
            if request.kind == "close":
                request.future.set_result(None)
                return
            try:
                if request.kind == "list_tools":
                    response = await session.list_tools()
                else:
                    response = await session.call_tool(request.tool_name, request.arguments or {})
                request.future.set_result(response)
            except BaseException as exc:
                request.future.set_exception(exc)

    async def _submit(
        self,
        kind: _RequestKind,
        *,
        tool_name: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """向 worker task 提交请求并等待结果。"""
        if not self._connected and kind != "close":
            raise RuntimeError(f"McpClient[{self._config.name}] 尚未 connect")
        queue = self._require_queue()
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        await queue.put(
            _ClientRequest(
                kind=kind,
                future=future,
                tool_name=tool_name,
                arguments=arguments,
            )
        )
        return await asyncio.wait_for(future, timeout=self._config.timeout_seconds)

    def _require_queue(self) -> asyncio.Queue[_ClientRequest]:
        """读取已初始化的请求队列。"""
        if self._queue is None:
            raise RuntimeError(f"McpClient[{self._config.name}] 请求队列尚未初始化")
        return self._queue

    def _compose_description(self, raw: str) -> str:
        """拼接 Server 描述与工具原始描述。"""
        prefix = self._config.description.strip()
        suffix = raw.strip()
        if prefix and suffix:
            return f"[{prefix}] {suffix}"
        return prefix or suffix or f"MCP 工具 {self._config.name}"
