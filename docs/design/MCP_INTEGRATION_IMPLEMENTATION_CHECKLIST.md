# MCP 集成实现清单（Codex 可执行）

> 配套文档：`docs/design/MCP_INTEGRATION_DESIGN.md`
> 适用对象：可独立读取代码的 Coding Agent（Codex / Claude Code 等）
> 编码语言：Python 3.11+，使用项目已有的 conda 环境 `xinyidai-chat-agent`
> **严格要求**：生产级代码、最终态、不允许 mock 占位、不允许 TODO 留白、不允许双轨兼容

---

## 0. 总则与约束

### 0.1 不可违反的规则

1. **禁止 mock**：所有新增组件必须给出最终实现。
2. **禁止 TODO**：清单要求完成的代码不允许留 `TODO/FIXME/pass`。
3. **fail-closed**：所有新增枚举默认值向"拒绝"倾斜；放行必须显式声明。
4. **不接征信后端**：MCP 任何示例、测试、配置均不允许引入征信后端 API。
5. **不破坏现有接口**：`AgentTool` Protocol、`ToolSpec`、`ToolResult`、`ToolCategory` 全部**只增不改**。
6. **中文注释**：类级、方法级、关键逻辑必须含中文注释。
7. **测试先行**：每个 Phase 的验收命令必须 100% 通过才能进入下一 Phase。
8. **不做治理**：本清单不涉及熔断 / 灰度 / 版本管理 / 自动重连。

### 0.2 项目环境

- 工作目录：`d:\companyCode\agent`
- Python 环境：conda env `xinyidai-chat-agent`（**禁止 base**）
- Node.js：需安装 `node >= 18` 与 `npx`（用于 Demo Server B）
- 测试命令：
  ```powershell
  conda activate xinyidai-chat-agent
  pytest test/ -x -v
  ```

### 0.3 命名规范

- MCP Client 主包：`src/xinyidai_agent/mcp/`
- 自实现 Server 子包：`xinyidai_mcp_servers/`（与 `src/` 平级，独立可发布）
- 配置文件：`config/mcp_servers.yaml`
- Demo 脚本：`scripts/demo_mcp_flow.py`
- 测试：`test/contract/test_mcp_*.py`、`test/smoke/test_mcp_*.py`、`test/e2e/test_mcp_*.py`
- qualified_name：`mcp__{server}__{tool}`（双下划线分隔）

### 0.4 依赖项变更

`pyproject.toml` 的 `dependencies` 追加：

```toml
"mcp>=1.2.0",
"openpyxl>=3.1.0",
"pyyaml>=6.0",
```

---

## 1. 改动总览（执行顺序 = 依赖顺序）

| Phase | 主题 | 新增 | 修改 |
|---|---|---|---|
| 0 | 依赖与目录基线 | `xinyidai_mcp_servers/__init__.py` | `pyproject.toml` |
| 1 | 协议层 | `src/xinyidai_agent/mcp/protocol.py` | - |
| 2 | BusinessStatus 扩展 | - | `src/xinyidai_agent/protocol.py` |
| 3 | 配置加载 | `src/xinyidai_agent/mcp/config_loader.py` `config/mcp_servers.yaml` | - |
| 4 | McpClient | `src/xinyidai_agent/mcp/client.py` | - |
| 5 | McpManager | `src/xinyidai_agent/mcp/manager.py` `src/xinyidai_agent/mcp/__init__.py` | - |
| 6 | McpToolAdapter | `src/xinyidai_agent/mcp/tool_adapter.py` | - |
| 7 | ToolRegistry 集成 | - | `src/xinyidai_agent/tools/registry.py` `src/xinyidai_agent/tools/__init__.py` |
| 8 | 自实现 template_server | `xinyidai_mcp_servers/template_server/server.py` `xinyidai_mcp_servers/template_server/__init__.py` `xinyidai_mcp_servers/template_server/README.md` | - |
| 9 | filesystem server 配置 | `workspace/sample.csv` `workspace/.gitkeep` | `config/mcp_servers.yaml` |
| 10 | API lifespan 集成 | - | `src/xinyidai_agent/api.py` `src/xinyidai_agent/config.py` |
| 11 | 测试矩阵 + Demo | `test/contract/test_mcp_protocol.py` `test/contract/test_mcp_tool_adapter.py` `test/smoke/test_mcp_connectivity.py` `test/e2e/test_mcp_e2e.py` `scripts/demo_mcp_flow.py` | - |

---

## 2. Phase 0：依赖与目录基线

### 2.1 目标

完成依赖安装与目录骨架；不动业务代码。

### 2.2 任务

#### 步骤 A：编辑 `pyproject.toml`

在 `dependencies` 数组末尾追加 3 行（保持字母序无强制）：

```toml
    "mcp>=1.2.0",
    "openpyxl>=3.1.0",
    "pyyaml>=6.0",
```

#### 步骤 B：安装依赖

```powershell
conda activate xinyidai-chat-agent
pip install -e .
```

#### 步骤 C：创建目录骨架（仅占位文件）

新建以下文件，内容如下：

`xinyidai_mcp_servers/__init__.py`：

```python
"""自实现的 MCP Server 集合。独立子包，可单独打包发布。"""
```

`xinyidai_mcp_servers/template_server/__init__.py`：

```python
"""通用模板生成 MCP Server。"""
```

`src/xinyidai_agent/mcp/__init__.py`：

```python
"""MCP（Model Context Protocol）客户端基础设施。

对外只暴露 McpManager 与 McpServerConfig，其余模块视为实现细节。
"""
```

`workspace/.gitkeep`：空文件。

#### 步骤 D：环境检查

```powershell
python -c "import mcp; import openpyxl; import yaml; print('ok')"
npx --version
node --version
```

### 2.3 验收

- 三个 `import` 全部成功输出 `ok`
- `npx --version` 返回非空
- 目录结构含：
  - `src/xinyidai_agent/mcp/__init__.py`
  - `xinyidai_mcp_servers/__init__.py`
  - `xinyidai_mcp_servers/template_server/__init__.py`
  - `workspace/.gitkeep`

---

## 3. Phase 1：协议层 `mcp/protocol.py`

### 3.1 目标

定义所有 MCP 相关的 Pydantic 数据模型，frozen + extra=forbid。

### 3.2 新建 `src/xinyidai_agent/mcp/protocol.py`

**完整文件内容**：

```python
"""MCP 协议层数据模型。

所有模型均 frozen=True、extra=forbid，作为模块边界的稳定契约。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from xinyidai_agent.protocol import RiskLevel

# 当前只支持 stdio；后续扩展位预留：Literal["stdio", "sse", "streamable_http"]
McpTransport = Literal["stdio"]


class McpServerConfig(BaseModel):
    """单个 MCP Server 配置。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    transport: McpTransport = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = "read_only"
    description: str = ""
    timeout_seconds: float = 30.0
    enabled: bool = True


class McpServersConfig(BaseModel):
    """整体配置文件根模型。"""

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
    """MCP Server 工具调用结果（规范化后）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_error: bool
    text_content: str = ""
    structured: dict[str, Any] = Field(default_factory=dict)
    raw_content: list[dict[str, Any]] = Field(default_factory=list)


def build_qualified_name(server_name: str, tool_name: str) -> str:
    """统一拼接 qualified_name：mcp__{server}__{tool}。

    server / tool 名内的下划线、连字符一律保留。
    """
    return f"mcp__{server_name}__{tool_name}"
```

### 3.3 验收

```powershell
python -c "from xinyidai_agent.mcp.protocol import McpServerConfig, McpServersConfig, McpToolDescriptor, McpToolCallResult, build_qualified_name; print(build_qualified_name('fs', 'read_file'))"
```

**期望输出**：`mcp__fs__read_file`

---

## 4. Phase 2：`BusinessStatus` 扩展

### 4.1 目标

在 `BusinessStatus` Literal 中追加 MCP 相关业务状态，**只增不删**。

### 4.2 修改 `src/xinyidai_agent/protocol.py`

定位 `BusinessStatus` Literal（搜索关键字 `BusinessStatus = Literal[`），在末尾追加 3 个值（保持已有顺序，添加到 `]` 之前）：

```python
    "TOOL_TIMEOUT",
    "TOOL_INPUT_INVALID",
    "TOOL_FAILED",
```

> 若以上常量已存在，跳过。**严禁删除任何已有常量**。

### 4.3 验收

```powershell
python -c "from xinyidai_agent.protocol import BusinessStatus; from typing import get_args; vals = get_args(BusinessStatus); assert 'TOOL_TIMEOUT' in vals and 'TOOL_INPUT_INVALID' in vals and 'TOOL_FAILED' in vals; print('ok')"
```

**期望输出**：`ok`

---

## 5. Phase 3：配置加载 `mcp/config_loader.py`

### 5.1 目标

从 YAML 加载 `McpServersConfig`；支持环境变量占位符替换 `${VAR_NAME}`。

### 5.2 新建 `src/xinyidai_agent/mcp/config_loader.py`

**完整文件内容**：

```python
"""MCP 配置文件加载器。

支持：
- YAML → McpServersConfig 校验
- 字符串字段内 ${ENV_VAR} 占位符替换（仅对 str / list[str] / dict[str, str] 生效）
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from xinyidai_agent.mcp.protocol import McpServersConfig


_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _substitute_env(value: Any) -> Any:
    """递归替换 str 内的 ${ENV_VAR}；未定义环境变量保留原字面量。"""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, list):
        return [_substitute_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _substitute_env(val) for key, val in value.items()}
    return value


def load_mcp_servers_config(path: Path | str) -> McpServersConfig:
    """读取并校验 MCP 配置文件；文件缺失返回空配置（启用零 Server 模式）。"""
    config_path = Path(path)
    if not config_path.exists():
        return McpServersConfig(servers=[])

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    substituted = _substitute_env(raw)
    return McpServersConfig.model_validate(substituted)


def default_config_path() -> Path:
    """默认配置路径：项目根 config/mcp_servers.yaml。"""
    return Path(__file__).resolve().parents[3] / "config" / "mcp_servers.yaml"
```

### 5.3 新建 `config/mcp_servers.yaml`

```yaml
# MCP Server 配置。Phase 8 / Phase 9 会补全 demo server。
# 字符串可使用 ${ENV_VAR} 占位符，未定义则保留字面量。
servers: []
```

### 5.4 验收

```powershell
python -c "from xinyidai_agent.mcp.config_loader import load_mcp_servers_config, default_config_path; cfg = load_mcp_servers_config(default_config_path()); print(len(cfg.servers))"
```

**期望输出**：`0`

---

## 6. Phase 4：McpClient（stdio）

### 6.1 目标

基于官方 `mcp` Python SDK 实现单 Server stdio 客户端，所有方法 async。

### 6.2 新建 `src/xinyidai_agent/mcp/client.py`

**完整文件内容**：

```python
"""MCP stdio 客户端封装。

设计要点：
- 持久 ClientSession：connect 之后保持长连接，避免每次调用重启子进程；
- 失败映射：超时 / 异常 / Server 返回 isError 都统一翻译为 McpToolCallResult；
- 关闭顺序：先关 session，再终止子进程，确保资源释放。
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from xinyidai_agent.mcp.protocol import (
    McpServerConfig,
    McpToolCallResult,
    McpToolDescriptor,
    build_qualified_name,
)


class McpClient:
    """单个 MCP Server 的 stdio 客户端。

    生命周期：connect() → list_tools() / call_tool() ×N → aclose()。
    非线程安全：必须在同一 event loop 内串行使用，依赖外层 McpManager 调度。
    """

    def __init__(self, config: McpServerConfig) -> None:
        """保存配置，不在构造期建立连接。"""
        self._config = config
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._connected = False

    @property
    def server_name(self) -> str:
        """返回 Server 标识，供日志与 trace 引用。"""
        return self._config.name

    async def connect(self) -> None:
        """拉起子进程并完成 MCP initialize 握手。"""
        if self._connected:
            return
        if self._config.transport != "stdio":
            raise NotImplementedError(f"暂不支持的 transport: {self._config.transport}")
        if not self._config.command:
            raise ValueError(f"Server {self._config.name} 缺少 command 字段")

        # AsyncExitStack 统一管理 stdio_client / ClientSession 两层上下文
        stack = AsyncExitStack()
        try:
            params = StdioServerParameters(
                command=self._config.command,
                args=list(self._config.args),
                env={**self._config.env} if self._config.env else None,
            )
            read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await asyncio.wait_for(session.initialize(), timeout=self._config.timeout_seconds)
        except BaseException:
            # 任何失败都要回滚已进入的上下文，避免子进程泄漏
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session
        self._connected = True

    async def list_tools(self) -> list[McpToolDescriptor]:
        """拉取工具清单，并按 allowed_tools 白名单过滤。"""
        session = self._require_session()
        response = await asyncio.wait_for(session.list_tools(), timeout=self._config.timeout_seconds)
        allowed = set(self._config.allowed_tools)
        descriptors: list[McpToolDescriptor] = []
        for tool in response.tools:
            if allowed and tool.name not in allowed:
                continue
            descriptors.append(
                McpToolDescriptor(
                    server_name=self._config.name,
                    tool_name=tool.name,
                    qualified_name=build_qualified_name(self._config.name, tool.name),
                    description=self._compose_description(tool.description or ""),
                    input_schema=dict(tool.inputSchema or {}),
                    risk_level=self._config.risk_level,
                )
            )
        return descriptors

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> McpToolCallResult:
        """同步等待 MCP tools/call 返回，并规范化结果。"""
        session = self._require_session()
        response = await asyncio.wait_for(
            session.call_tool(tool_name, arguments or {}),
            timeout=self._config.timeout_seconds,
        )
        text_parts: list[str] = []
        raw_content: list[dict[str, Any]] = []
        for item in response.content or []:
            payload = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            raw_content.append(payload)
            if payload.get("type") == "text":
                text_parts.append(str(payload.get("text") or ""))
        structured: dict[str, Any] = {}
        # 新版 SDK 可能携带 structuredContent，向下兼容空字段
        structured_payload = getattr(response, "structuredContent", None)
        if isinstance(structured_payload, dict):
            structured = dict(structured_payload)
        return McpToolCallResult(
            is_error=bool(response.isError),
            text_content="\n".join(text_parts),
            structured=structured,
            raw_content=raw_content,
        )

    async def aclose(self) -> None:
        """优雅关闭 session 与子进程。"""
        if not self._connected:
            return
        self._connected = False
        self._session = None
        if self._stack is not None:
            try:
                await self._stack.aclose()
            finally:
                self._stack = None

    def _require_session(self) -> ClientSession:
        """读取已建立的 session，未连接时抛出明确错误。"""
        if not self._connected or self._session is None:
            raise RuntimeError(f"McpClient[{self._config.name}] 尚未 connect")
        return self._session

    def _compose_description(self, raw: str) -> str:
        """把 Server 描述与工具描述拼接，便于模型识别工具来源。"""
        prefix = self._config.description.strip()
        suffix = raw.strip()
        if prefix and suffix:
            return f"[{prefix}] {suffix}"
        return prefix or suffix or f"MCP 工具 {self._config.name}"
```

### 6.3 验收

```powershell
python -c "from xinyidai_agent.mcp.client import McpClient; print(McpClient.__doc__.splitlines()[0])"
```

**期望输出**：`单个 MCP Server 的 stdio 客户端。`

> 真正的连接测试在 Phase 11 的 smoke 测试中验证（需要 Server 子进程）。

---

## 7. Phase 5：McpManager

### 7.1 目标

管理多个 `McpClient`，提供同步 `call_tool_sync()` 入口，桥接到现有 `AsyncRuntime`。

### 7.2 新建 `src/xinyidai_agent/mcp/manager.py`

**完整文件内容**：

```python
"""MCP 多 Server 生命周期管理器。

承担三件事：
1. 启动期：按配置并发拉起 Client，并 list_tools 缓存；
2. 运行期：把同步调用转译到 AsyncRuntime 上的协程；
3. 关闭期：按相反顺序 aclose 所有 Client。

非线程安全的内部状态全部位于后台 loop 内访问；对外暴露的同步方法做了
线程间序列化（_call_lock）。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from xinyidai_agent.mcp.client import McpClient
from xinyidai_agent.mcp.protocol import (
    McpServerConfig,
    McpToolCallResult,
    McpToolDescriptor,
)
from xinyidai_agent.rag.async_runtime import AsyncRuntime


logger = logging.getLogger(__name__)


class McpManager:
    """统一管理多 MCP Server 的同步外观（facade）。"""

    def __init__(
        self,
        configs: list[McpServerConfig],
        async_runtime: AsyncRuntime,
    ) -> None:
        """保存配置与运行时引用；不会立即拉起子进程。"""
        self._configs = [cfg for cfg in configs if cfg.enabled]
        self._runtime = async_runtime
        self._clients: dict[str, McpClient] = {}
        self._descriptors: dict[str, McpToolDescriptor] = {}
        self._call_lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        """阻塞拉起所有 enabled Server，list_tools 缓存到本地。

        单个 Server 启动失败仅记录日志，不影响其它 Server；该 Server
        的工具不会出现在 iter_descriptors 中。
        """
        if self._started:
            return
        self._runtime.run_coroutine(self._start_async())
        self._started = True

    def shutdown(self) -> None:
        """按启动相反顺序关闭所有 Server。"""
        if not self._started:
            return
        self._runtime.run_coroutine(self._shutdown_async())
        self._started = False

    def iter_descriptors(self) -> list[McpToolDescriptor]:
        """返回当前可用的工具描述符列表（拷贝，避免外部修改）。"""
        return list(self._descriptors.values())

    def call_tool_sync(
        self,
        qualified_name: str,
        arguments: dict[str, Any],
    ) -> McpToolCallResult:
        """同步调用 MCP 工具，结果完全规范化。

        线程安全：通过 _call_lock 串行化对同一后台 loop 的提交。
        """
        if not self._started:
            raise RuntimeError("McpManager 尚未 start()")
        descriptor = self._descriptors.get(qualified_name)
        if descriptor is None:
            raise KeyError(f"未注册的 MCP 工具：{qualified_name}")
        client = self._clients.get(descriptor.server_name)
        if client is None:
            raise KeyError(f"Server 未就绪：{descriptor.server_name}")
        with self._call_lock:
            return self._runtime.run_coroutine(
                client.call_tool(descriptor.tool_name, arguments)
            )

    async def _start_async(self) -> None:
        """并发拉起所有 Server，捕获单点失败。"""
        tasks = [self._start_single(config) for config in self._configs]
        await asyncio.gather(*tasks, return_exceptions=False)

    async def _start_single(self, config: McpServerConfig) -> None:
        """启动单个 Server 并缓存其工具描述符。"""
        client = McpClient(config)
        try:
            await client.connect()
            descriptors = await client.list_tools()
        except Exception as exc:
            logger.warning(
                "MCP Server %s 启动失败，跳过该 Server：%s",
                config.name,
                exc,
            )
            await client.aclose()
            return
        self._clients[config.name] = client
        for descriptor in descriptors:
            self._descriptors[descriptor.qualified_name] = descriptor
        logger.info(
            "MCP Server %s 已就绪，注册 %d 个工具",
            config.name,
            len(descriptors),
        )

    async def _shutdown_async(self) -> None:
        """逐个关闭所有 Client，错误吞掉但写日志。"""
        for name, client in list(self._clients.items()):
            try:
                await client.aclose()
            except Exception as exc:
                logger.warning("MCP Server %s 关闭异常：%s", name, exc)
        self._clients.clear()
        self._descriptors.clear()
```

### 7.3 编辑 `src/xinyidai_agent/mcp/__init__.py`

```python
"""MCP（Model Context Protocol）客户端基础设施。

对外只暴露 McpManager 与配置加载入口；其余视为实现细节。
"""

from xinyidai_agent.mcp.config_loader import (
    default_config_path,
    load_mcp_servers_config,
)
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
```

### 7.4 验收

```powershell
python -c "from xinyidai_agent.mcp import McpManager, load_mcp_servers_config, default_config_path; print('ok')"
```

**期望输出**：`ok`

---

## 8. Phase 6：McpToolAdapter

### 8.1 目标

把 `McpToolDescriptor` 包装成符合 `AgentTool` Protocol 的对象，主循环零感知。

### 8.2 新建 `src/xinyidai_agent/mcp/tool_adapter.py`

**完整文件内容**：

```python
"""MCP → AgentTool 适配层。

把 McpToolDescriptor 适配成 ToolRegistry 可注册的 AgentTool 实现，并
负责把 McpToolCallResult 转译为标准 ToolResult / ToolExecution。
"""

from __future__ import annotations

import asyncio
from typing import Any

from xinyidai_agent.mcp.manager import McpManager
from xinyidai_agent.mcp.protocol import McpToolCallResult, McpToolDescriptor
from xinyidai_agent.protocol import (
    ChatRequest,
    RouteDecision,
    ToolCall,
    ToolCategory,
    ToolResult,
    ToolResultEnvelope,
)
from xinyidai_agent.tools.base import SlotSpec, ToolExecution, ToolSpec


_MCP_CATEGORY: ToolCategory = "utility"


class McpToolAdapter:
    """单个 MCP 工具的本地适配器。一对一对应一个 McpToolDescriptor。"""

    def __init__(self, descriptor: McpToolDescriptor, manager: McpManager) -> None:
        """绑定描述符与 manager，运行期通过 manager 走 sync 桥接。"""
        self._descriptor = descriptor
        self._manager = manager
        self.name: str = descriptor.qualified_name
        self.category: ToolCategory = _MCP_CATEGORY
        self.risk_level = descriptor.risk_level
        self.description = descriptor.description
        self.requires_confirmation = descriptor.risk_level != "read_only"

    def spec(self) -> ToolSpec:
        """根据 input_schema 派生 SlotSpec，构造完整 ToolSpec。"""
        input_slots = _derive_slot_specs(self._descriptor.input_schema)
        is_read_only = self.risk_level == "read_only"
        return ToolSpec(
            name=self.name,
            category=self.category,
            risk_level=self.risk_level,
            description=self.description,
            requires_confirmation=self.requires_confirmation,
            input_slots=input_slots,
            output_slots=[
                SlotSpec(
                    "text_content",
                    "string",
                    required=False,
                    description="MCP Server 返回的文本内容",
                    allow_empty=True,
                ),
                SlotSpec(
                    "structured",
                    "object",
                    required=False,
                    description="MCP Server 返回的结构化结果",
                    allow_empty=True,
                ),
            ],
            input_schema=dict(self._descriptor.input_schema),
            is_read_only=is_read_only,
            is_idempotent=is_read_only,
            is_concurrency_safe=is_read_only,
            cost_class="medium",
            max_duration_ms=int(30 * 1000),
        )

    def execute(
        self,
        request: ChatRequest,
        route: RouteDecision,
        tool_call: ToolCall,
    ) -> ToolExecution:
        """同步执行 MCP 工具并产出标准 ToolResult。"""
        arguments = dict(tool_call.arguments or {})
        try:
            call_result = self._manager.call_tool_sync(self.name, arguments)
        except asyncio.TimeoutError:
            return self._build_failed_execution(
                tool_call,
                business_status="TOOL_TIMEOUT",
                message=f"MCP 工具 {self.name} 调用超时。",
                model_hint="工具调用超时，请改用其它路径或要求用户提供更精简的输入。",
            )
        except KeyError as exc:
            return self._build_failed_execution(
                tool_call,
                business_status="TOOL_UNAVAILABLE",
                message=f"MCP 工具不可用：{exc}",
                model_hint="该工具暂不可用，请考虑改走其它工具或转人工。",
            )
        except Exception as exc:  # noqa: BLE001 — 边界处统一兜底，写明业务码
            return self._build_failed_execution(
                tool_call,
                business_status="TOOL_FAILED",
                message=f"MCP 工具异常：{exc}",
                model_hint="工具内部错误，不要重复同样的调用，换工具或解释失败。",
            )
        return self._build_success_execution(tool_call, call_result)

    def _build_success_execution(
        self,
        tool_call: ToolCall,
        call_result: McpToolCallResult,
    ) -> ToolExecution:
        """根据 MCP 返回构造成功 / 空结果分支。"""
        if call_result.is_error:
            return self._build_failed_execution(
                tool_call,
                business_status="TOOL_FAILED",
                message=call_result.text_content or "MCP 工具返回 isError=true。",
                model_hint="工具明确报错，请勿继续调用相同参数。",
            )
        has_payload = bool(call_result.text_content) or bool(call_result.structured)
        business_status = "OK" if has_payload else "PARTIAL_DATA"
        message = (
            f"MCP 工具 {self.name} 执行完成。"
            if has_payload
            else f"MCP 工具 {self.name} 未返回有效内容。"
        )
        output: dict[str, Any] = {
            "text_content": call_result.text_content,
            "structured": call_result.structured,
        }
        envelope = ToolResultEnvelope(
            success=True,
            status=business_status,
            code=0,
            message=message,
            data=output,
        )
        result = ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            tool_category=self.category,
            status="success",
            output=output,
            envelope=envelope,
            business_status=business_status,
            code=0,
            message=message,
            terminal=True,
            model_observation=(
                "MCP 工具执行完成，可基于 text_content / structured 组织答复。"
                if has_payload
                else "MCP 工具未返回内容，请考虑换工具或转人工。"
            ),
            user_visible_message=None,
        )
        return ToolExecution(result=result)

    def _build_failed_execution(
        self,
        tool_call: ToolCall,
        *,
        business_status: str,
        message: str,
        model_hint: str,
    ) -> ToolExecution:
        """统一构造失败分支，business_status 必须在 BusinessStatus Literal 中。"""
        envelope = ToolResultEnvelope(
            success=False,
            status=business_status,  # type: ignore[arg-type]
            code=-1,
            message=message,
            data={},
        )
        result = ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            tool_category=self.category,
            status="failed",
            output={},
            envelope=envelope,
            business_status=business_status,  # type: ignore[arg-type]
            code=-1,
            message=message,
            terminal=True,
            model_observation=model_hint,
            user_visible_message=None,
        )
        return ToolExecution(result=result)


def _derive_slot_specs(schema: dict[str, Any]) -> list[SlotSpec]:
    """从 JSON Schema 顶层 properties 派生 SlotSpec 列表。"""
    if not schema:
        return []
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    slots: list[SlotSpec] = []
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            continue
        value_type = _map_json_type(prop_schema.get("type"))
        description = str(prop_schema.get("description") or "")
        slots.append(
            SlotSpec(
                name=str(prop_name),
                value_type=value_type,
                required=prop_name in required,
                description=description,
                allow_empty=False,
            )
        )
    return slots


def _map_json_type(json_type: Any) -> str:
    """把 JSON Schema 的 type 映射为现有 SlotSpec.value_type 词表。

    现有 SlotSpec.value_type 是开放字符串，无严格 Literal 约束，
    这里采用业界通用约定（string / integer / number / boolean / array / object）。
    """
    if isinstance(json_type, list):
        # 多类型时取第一个非 null
        for candidate in json_type:
            if candidate and candidate != "null":
                return str(candidate)
        return "string"
    if isinstance(json_type, str):
        return json_type
    return "string"
```

### 8.3 验收

```powershell
python -c "from xinyidai_agent.mcp.tool_adapter import McpToolAdapter; import inspect; print('execute' in dict(inspect.getmembers(McpToolAdapter)))"
```

**期望输出**：`True`

---

## 9. Phase 7：ToolRegistry 集成

### 9.1 目标

让 `ToolRegistry` 接受运行期注入的 `McpToolAdapter`，并提供组合工厂。

### 9.2 修改 `src/xinyidai_agent/tools/registry.py`

#### 步骤 A：定位 `ToolRegistry.__init__`（约第 28 行起）

在 `__init__` 结尾追加一个公开方法 `register_external`，用于运行期注入 MCP 工具：

```python
    def register_external(self, tool: AgentTool) -> None:
        """运行期把外部工具（如 McpToolAdapter）注入注册表。

        重名直接抛错，避免无声覆盖；调用方应保证 name 唯一。
        """
        if tool.name in self._tools:
            raise ValueError(f"工具名重复注册：{tool.name}")
        self._tools[tool.name] = tool
        self._tools_by_category.setdefault(tool.category, []).append(tool)
```

> **修改要求**：
> - 在类的现有方法之间插入；保持缩进；
> - **不要修改 `_tools` / `_tools_by_category` 的初始化顺序**。

#### 步骤 B：在文件末尾 `default_tool_registry` 后追加 `build_tool_registry`

```python
def build_tool_registry(
    retriever: Retriever | None = None,
    mcp_adapters: Iterable[AgentTool] | None = None,
) -> ToolRegistry:
    """组合工厂：默认工具 + 可选 MCP 适配器一并注册。"""
    registry = default_tool_registry(retriever)
    if mcp_adapters:
        for adapter in mcp_adapters:
            registry.register_external(adapter)
    return registry
```

### 9.3 修改 `src/xinyidai_agent/tools/__init__.py`

在 `__all__` 中追加 `"build_tool_registry"`，并补充对应 import：

```python
from xinyidai_agent.tools.registry import (
    ToolRegistry,
    build_tool_registry,
    default_tool_registry,
)
```

### 9.4 验收

```powershell
python -c "from xinyidai_agent.tools import build_tool_registry, default_tool_registry; r = build_tool_registry(); print(len(r.names_for_categories(['knowledge', 'data_query', 'application', 'authorization', 'status', 'utility'])))"
```

**期望输出**：等于现有默认工具数量（非零整数）。

---

## 10. Phase 8：自实现 template MCP Server

### 10.1 目标

用 Python `mcp` SDK 实现一个最小可用的 MCP Server，对外暴露 3 个通用模板生成工具。证明项目对接的是协议本身，而非特定 Server。

### 10.2 新建 `xinyidai_mcp_servers/template_server/server.py`

**完整文件内容**：

```python
"""通用模板生成 MCP Server。

对外工具：
- generate_markdown_template：按场景生成 Markdown 模板文件；
- generate_excel_template：按列定义生成 Excel 模板文件；
- list_generated_templates：列出已生成的模板文件。

所有产物落在 .data/mcp_templates/，幂等可重入。
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from openpyxl import Workbook


_OUTPUT_DIR = Path(os.environ.get("TEMPLATE_OUTPUT_DIR", ".data/mcp_templates")).resolve()


_MARKDOWN_TEMPLATES: dict[str, str] = {
    "loan_apply": (
        "# 贷款申请采集模板\n\n"
        "## 基本信息\n"
        "- 客户名称：\n"
        "- 统一社会信用代码：\n"
        "- 联系电话：\n\n"
        "## 申请要素\n"
        "- 申请额度（万元）：\n"
        "- 期限（月）：\n"
        "- 用途：\n"
    ),
    "customer_intake": (
        "# 客户准入信息采集模板\n\n"
        "## 主体信息\n"
        "- 主体名称：\n"
        "- 注册地址：\n"
        "- 法人代表：\n\n"
        "## 经营信息\n"
        "- 主营业务：\n"
        "- 上一年度营收（万元）：\n"
    ),
}


server = Server("xinyidai-template-server")


@server.list_tools()
async def _list_tools() -> list[Tool]:
    """注册工具清单，供 MCP Client 在 initialize 后调用。"""
    return [
        Tool(
            name="generate_markdown_template",
            description="按场景生成 Markdown 业务模板文件，落地到工作区。",
            inputSchema={
                "type": "object",
                "properties": {
                    "scene": {
                        "type": "string",
                        "description": "模板场景，如 loan_apply、customer_intake",
                    },
                    "title": {
                        "type": "string",
                        "description": "模板标题，写入文件首行 H1",
                    },
                },
                "required": ["scene"],
            },
        ),
        Tool(
            name="generate_excel_template",
            description="按列定义生成 Excel 模板文件（首行为列标题）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "列标题数组",
                    },
                    "filename": {
                        "type": "string",
                        "description": "目标文件名（不带扩展名）",
                    },
                },
                "required": ["columns", "filename"],
            },
        ),
        Tool(
            name="list_generated_templates",
            description="列出工作区已生成的模板文件清单。",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """统一分发入口；所有失败都抛异常，由 MCP SDK 包装 isError=true。"""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if name == "generate_markdown_template":
        return [TextContent(type="text", text=_generate_markdown(arguments))]
    if name == "generate_excel_template":
        return [TextContent(type="text", text=_generate_excel(arguments))]
    if name == "list_generated_templates":
        return [TextContent(type="text", text=_list_templates())]
    raise ValueError(f"未知工具：{name}")


def _generate_markdown(arguments: dict[str, Any]) -> str:
    """根据 scene 选模板，写入文件并返回相对路径。"""
    scene = str(arguments.get("scene") or "").strip()
    title = str(arguments.get("title") or "").strip()
    if not scene:
        raise ValueError("缺少必填字段：scene")
    body = _MARKDOWN_TEMPLATES.get(scene)
    if body is None:
        raise ValueError(f"未知场景：{scene}")
    if title:
        body = body.replace(body.splitlines()[0], f"# {title}", 1)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    target = _OUTPUT_DIR / f"{scene}_{ts}.md"
    target.write_text(body, encoding="utf-8")
    return f"已生成 Markdown 模板：{target.as_posix()}"


def _generate_excel(arguments: dict[str, Any]) -> str:
    """生成首行带标题的 Excel 模板文件。"""
    columns = arguments.get("columns") or []
    filename = str(arguments.get("filename") or "").strip()
    if not columns:
        raise ValueError("缺少必填字段：columns")
    if not filename:
        raise ValueError("缺少必填字段：filename")
    if not isinstance(columns, list):
        raise TypeError("columns 必须是数组")
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "template"
    sheet.append([str(col) for col in columns])
    target = _OUTPUT_DIR / f"{filename}.xlsx"
    workbook.save(target)
    return f"已生成 Excel 模板：{target.as_posix()}"


def _list_templates() -> str:
    """返回已生成的模板文件相对路径列表。"""
    if not _OUTPUT_DIR.exists():
        return "尚未生成任何模板。"
    files = sorted(p for p in _OUTPUT_DIR.iterdir() if p.is_file())
    if not files:
        return "尚未生成任何模板。"
    return "\n".join(file.as_posix() for file in files)


async def _main() -> None:
    """stdio 入口；交给 MCP SDK 接管 stdin/stdout。"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(_main())
```

### 10.3 新建 `xinyidai_mcp_servers/template_server/README.md`

```markdown
# Template MCP Server

通用业务模板生成 MCP Server，对外暴露三个工具：

- `generate_markdown_template(scene, title?)`
- `generate_excel_template(columns, filename)`
- `list_generated_templates()`

## 运行

```bash
python -m xinyidai_mcp_servers.template_server.server
```

## 产物目录

默认 `./.data/mcp_templates/`；通过环境变量 `TEMPLATE_OUTPUT_DIR` 覆盖。
```

### 10.4 验收

```powershell
python -c "from xinyidai_mcp_servers.template_server import server as srv_module; print(getattr(srv_module, 'server').name)"
```

**期望输出**：`xinyidai-template-server`

---

## 11. Phase 9：filesystem server 配置 + 演示工作区

### 11.1 目标

接入官方 `@modelcontextprotocol/server-filesystem`，演示文件读取能力；并补全 `config/mcp_servers.yaml`。

### 11.2 修改 `config/mcp_servers.yaml`

**覆盖**为以下内容：

```yaml
# MCP Server 配置。
# 字符串可使用 ${ENV_VAR} 占位符，未定义则保留字面量。

servers:
  # Demo Server A：自实现的 Python 模板生成器
  - name: template
    transport: stdio
    command: python
    args:
      - "-m"
      - "xinyidai_mcp_servers.template_server.server"
    description: "通用业务模板生成器"
    allowed_tools:
      - generate_markdown_template
      - generate_excel_template
      - list_generated_templates
    risk_level: read_only
    timeout_seconds: 30
    enabled: true

  # Demo Server B：官方 filesystem，限定工作区只读
  - name: filesystem
    transport: stdio
    command: npx
    args:
      - "-y"
      - "@modelcontextprotocol/server-filesystem"
      - "./workspace"
    description: "工作区文件读取（只读）"
    allowed_tools:
      - read_file
      - list_directory
    risk_level: read_only
    timeout_seconds: 30
    enabled: true
```

### 11.3 新建 `workspace/sample.csv`

**内容**：

```csv
客户名称,统一社会信用代码,联系电话,申请额度
重庆好客来商贸有限公司,91500000000000000A,13800000001,50
重庆鑫达制造有限公司,91500000000000000B,13800000002,200
重庆三和建筑工程有限公司,91500000000000000C,13800000003,800
```

### 11.4 验收

```powershell
python -c "from xinyidai_agent.mcp import load_mcp_servers_config, default_config_path; cfg = load_mcp_servers_config(default_config_path()); print([(s.name, s.transport, len(s.allowed_tools)) for s in cfg.servers])"
```

**期望输出**：`[('template', 'stdio', 3), ('filesystem', 'stdio', 2)]`

---

## 12. Phase 10：API lifespan 集成

### 12.1 目标

在 FastAPI lifespan 内启停 `McpManager`，把 MCP 工具注入 `ToolRegistry`。

### 12.2 修改 `src/xinyidai_agent/config.py`

在文件末尾追加：

```python
@dataclass(frozen=True)
class McpConfig:
    """MCP 客户端启停配置。"""

    enabled: bool = True
    config_path: str = ""
    """空字符串表示使用 default_config_path()。"""

    @classmethod
    def from_env(cls) -> "McpConfig":
        """从环境变量读取；MCP_ENABLED=0 时整体停用。"""
        enabled_raw = os.getenv("MCP_ENABLED", "1").strip().lower()
        return cls(
            enabled=enabled_raw not in {"0", "false", "no"},
            config_path=os.getenv("MCP_CONFIG_PATH", "").strip(),
        )
```

### 12.3 修改 `src/xinyidai_agent/api.py`

#### 步骤 A：补充 imports（顶部）

```python
from pathlib import Path

from xinyidai_agent.config import McpConfig
from xinyidai_agent.mcp import (
    McpManager,
    default_config_path,
    load_mcp_servers_config,
)
from xinyidai_agent.mcp.tool_adapter import McpToolAdapter
from xinyidai_agent.rag.async_runtime import AsyncRuntime
from xinyidai_agent.tools import build_tool_registry
```

> 若以上某些 import 已存在，跳过。新增 import 必须放在文件顶部 imports 块。

#### 步骤 B：在现有 FastAPI `lifespan` 函数内拼装 MCP

定位 `lifespan` 异步上下文管理器（搜索关键字 `@asynccontextmanager` 或 `lifespan`）。

- **启动阶段**（yield 之前）追加：

  ```python
      mcp_cfg = McpConfig.from_env()
      async_runtime: AsyncRuntime = app.state.async_runtime
      mcp_manager: McpManager | None = None
      mcp_adapters: list[McpToolAdapter] = []
      if mcp_cfg.enabled:
          path = Path(mcp_cfg.config_path) if mcp_cfg.config_path else default_config_path()
          servers_cfg = load_mcp_servers_config(path)
          mcp_manager = McpManager(servers_cfg.servers, async_runtime)
          mcp_manager.start()
          mcp_adapters = [
              McpToolAdapter(descriptor, mcp_manager)
              for descriptor in mcp_manager.iter_descriptors()
          ]
      app.state.mcp_manager = mcp_manager
      app.state.tool_registry = build_tool_registry(
          retriever=app.state.retriever,
          mcp_adapters=mcp_adapters,
      )
  ```

- **关闭阶段**（yield 之后）追加：

  ```python
      if mcp_manager is not None:
          mcp_manager.shutdown()
  ```

> **修改要求**：
> - 必须使用现有 `app.state.async_runtime` / `app.state.retriever`；若变量名不同，按文件内实际命名替换；
> - 不允许新建 `AsyncRuntime` 实例（必须复用）；
> - 不允许吞掉 `mcp_manager.start()` 的异常（fail-closed 已经在 manager 内做了单 Server 兜底，外层异常视为基础设施错误）。

### 12.4 验收

```powershell
conda activate xinyidai-chat-agent
$env:MCP_ENABLED="1"
uvicorn xinyidai_agent.api:app --port 18000 --log-level warning
```

启动后通过另一个终端：

```powershell
curl http://127.0.0.1:18000/health
```

**期望**：启动日志出现 `MCP Server template 已就绪，注册 3 个工具` 与 `MCP Server filesystem 已就绪，注册 2 个工具`；`/health` 返回 200。

---

## 13. Phase 11：测试矩阵 + Demo 脚本

### 13.1 目标

完成 contract / smoke / e2e 三层测试，并交付 Demo 脚本。

### 13.2 新建 `test/contract/test_mcp_protocol.py`

```python
"""协议层契约测试：模型 frozen / extra=forbid / 默认值。"""

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
    def test_default_risk_level_is_read_only(self) -> None:
        cfg = McpServerConfig(name="x", command="python")
        assert cfg.risk_level == "read_only"
        assert cfg.transport == "stdio"
        assert cfg.enabled is True

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            McpServerConfig(name="x", command="python", unknown="value")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        cfg = McpServerConfig(name="x", command="python")
        with pytest.raises(ValidationError):
            cfg.name = "y"  # type: ignore[misc]


class TestServersConfig:
    def test_empty_servers_default(self) -> None:
        cfg = McpServersConfig()
        assert cfg.servers == []


class TestQualifiedName:
    def test_format(self) -> None:
        assert build_qualified_name("fs", "read_file") == "mcp__fs__read_file"


class TestToolDescriptor:
    def test_minimal_fields(self) -> None:
        desc = McpToolDescriptor(
            server_name="fs",
            tool_name="read_file",
            qualified_name="mcp__fs__read_file",
            description="读取文件",
        )
        assert desc.input_schema == {}
        assert desc.risk_level == "read_only"
```

### 13.3 新建 `test/contract/test_mcp_tool_adapter.py`

```python
"""适配层契约测试：JSON Schema → SlotSpec / 失败分支映射。"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from xinyidai_agent.mcp.protocol import McpToolCallResult, McpToolDescriptor
from xinyidai_agent.mcp.tool_adapter import McpToolAdapter
from xinyidai_agent.protocol import ChatRequest, RouteDecision, ToolCall


def _descriptor(input_schema: dict[str, Any] | None = None) -> McpToolDescriptor:
    return McpToolDescriptor(
        server_name="template",
        tool_name="generate_markdown_template",
        qualified_name="mcp__template__generate_markdown_template",
        description="测试",
        input_schema=input_schema or {},
        risk_level="read_only",
    )


def _tool_call() -> ToolCall:
    return ToolCall(
        tool_call_id="t1",
        tool_name="mcp__template__generate_markdown_template",
        arguments={"scene": "loan_apply"},
        risk_level="read_only",
    )


def _route() -> RouteDecision:
    return RouteDecision(
        scene="UTILITY",
        allowed_tools=["mcp__template__generate_markdown_template"],
        allowed_tool_categories=["utility"],
        risk_level="read_only",
        route_reason="test",
    )


def _request() -> ChatRequest:
    return ChatRequest(user_message="x", top_k=3)


class TestSpecDerivation:
    def test_required_slot_from_schema(self) -> None:
        manager = MagicMock()
        adapter = McpToolAdapter(
            _descriptor(
                {
                    "type": "object",
                    "properties": {
                        "scene": {"type": "string", "description": "场景"},
                        "title": {"type": "string"},
                    },
                    "required": ["scene"],
                }
            ),
            manager,
        )
        spec = adapter.spec()
        slot_names = {slot.name: slot.required for slot in spec.input_slots}
        assert slot_names == {"scene": True, "title": False}
        assert spec.is_read_only is True
        assert spec.is_idempotent is True


class TestExecution:
    def test_success_branch(self) -> None:
        manager = MagicMock()
        manager.call_tool_sync.return_value = McpToolCallResult(
            is_error=False,
            text_content="已生成模板：x.md",
        )
        adapter = McpToolAdapter(_descriptor(), manager)
        execution = adapter.execute(_request(), _route(), _tool_call())
        assert execution.result.status == "success"
        assert execution.result.business_status == "OK"
        assert execution.result.output["text_content"] == "已生成模板：x.md"

    def test_empty_branch(self) -> None:
        manager = MagicMock()
        manager.call_tool_sync.return_value = McpToolCallResult(
            is_error=False, text_content=""
        )
        adapter = McpToolAdapter(_descriptor(), manager)
        execution = adapter.execute(_request(), _route(), _tool_call())
        assert execution.result.business_status == "PARTIAL_DATA"

    def test_is_error_branch(self) -> None:
        manager = MagicMock()
        manager.call_tool_sync.return_value = McpToolCallResult(
            is_error=True, text_content="缺少必填字段：scene"
        )
        adapter = McpToolAdapter(_descriptor(), manager)
        execution = adapter.execute(_request(), _route(), _tool_call())
        assert execution.result.status == "failed"
        assert execution.result.business_status == "TOOL_FAILED"

    def test_unavailable_branch(self) -> None:
        manager = MagicMock()
        manager.call_tool_sync.side_effect = KeyError("missing")
        adapter = McpToolAdapter(_descriptor(), manager)
        execution = adapter.execute(_request(), _route(), _tool_call())
        assert execution.result.business_status == "TOOL_UNAVAILABLE"
```

### 13.4 新建 `test/smoke/test_mcp_connectivity.py`

```python
"""冒烟测试：拉起自实现 template_server，验证 list_tools / call_tool 全链路。

依赖：mcp SDK、openpyxl、Python 3.11+。不依赖 Node.js。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xinyidai_agent.mcp import McpManager, McpServerConfig
from xinyidai_agent.rag.async_runtime import AsyncRuntime


@pytest.fixture(scope="module")
def async_runtime() -> AsyncRuntime:
    runtime = AsyncRuntime()
    try:
        yield runtime
    finally:
        runtime.close()


@pytest.fixture(scope="module")
def template_manager(async_runtime: AsyncRuntime, tmp_path_factory: pytest.TempPathFactory) -> McpManager:
    output_dir = tmp_path_factory.mktemp("mcp_templates")
    config = McpServerConfig(
        name="template",
        command="python",
        args=["-m", "xinyidai_mcp_servers.template_server.server"],
        env={"TEMPLATE_OUTPUT_DIR": str(output_dir)},
        allowed_tools=[
            "generate_markdown_template",
            "generate_excel_template",
            "list_generated_templates",
        ],
    )
    manager = McpManager([config], async_runtime)
    manager.start()
    yield manager
    manager.shutdown()


def test_list_descriptors(template_manager: McpManager) -> None:
    names = sorted(d.qualified_name for d in template_manager.iter_descriptors())
    assert names == sorted(
        [
            "mcp__template__generate_markdown_template",
            "mcp__template__generate_excel_template",
            "mcp__template__list_generated_templates",
        ]
    )


def test_call_markdown_template(template_manager: McpManager) -> None:
    result = template_manager.call_tool_sync(
        "mcp__template__generate_markdown_template",
        {"scene": "loan_apply", "title": "贷款采集（测试）"},
    )
    assert result.is_error is False
    assert "贷款采集" in result.text_content or "已生成" in result.text_content


def test_call_excel_template(template_manager: McpManager, tmp_path: Path) -> None:
    result = template_manager.call_tool_sync(
        "mcp__template__generate_excel_template",
        {"columns": ["客户名称", "额度"], "filename": "test_excel"},
    )
    assert result.is_error is False
    assert "已生成" in result.text_content
```

### 13.5 新建 `test/e2e/test_mcp_e2e.py`

```python
"""E2E：McpToolAdapter 注入 ToolRegistry，并通过 execute_observed 走完整链路。

不启动 FastAPI，直接走运行时组件以缩短反馈周期。
"""

from __future__ import annotations

import pytest

from xinyidai_agent.mcp import McpManager, McpServerConfig
from xinyidai_agent.mcp.tool_adapter import McpToolAdapter
from xinyidai_agent.protocol import ChatRequest, RouteDecision, ToolCall
from xinyidai_agent.rag.async_runtime import AsyncRuntime
from xinyidai_agent.tools import build_tool_registry


@pytest.fixture(scope="module")
def registry_with_mcp(tmp_path_factory: pytest.TempPathFactory):
    output_dir = tmp_path_factory.mktemp("mcp_templates_e2e")
    runtime = AsyncRuntime()
    config = McpServerConfig(
        name="template",
        command="python",
        args=["-m", "xinyidai_mcp_servers.template_server.server"],
        env={"TEMPLATE_OUTPUT_DIR": str(output_dir)},
        allowed_tools=[
            "generate_markdown_template",
            "generate_excel_template",
            "list_generated_templates",
        ],
    )
    manager = McpManager([config], runtime)
    manager.start()
    adapters = [
        McpToolAdapter(descriptor, manager) for descriptor in manager.iter_descriptors()
    ]
    registry = build_tool_registry(mcp_adapters=adapters)
    try:
        yield registry
    finally:
        manager.shutdown()
        runtime.close()


def test_mcp_tools_registered(registry_with_mcp) -> None:
    utility_names = registry_with_mcp.names_for_categories(["utility"])
    assert "mcp__template__generate_markdown_template" in utility_names


def test_execute_generates_template(registry_with_mcp) -> None:
    tool = registry_with_mcp._tools["mcp__template__generate_markdown_template"]
    request = ChatRequest(user_message="生成贷款模板", top_k=3)
    route = RouteDecision(
        scene="UTILITY",
        allowed_tools=["mcp__template__generate_markdown_template"],
        allowed_tool_categories=["utility"],
        risk_level="read_only",
        route_reason="test",
    )
    tool_call = ToolCall(
        tool_call_id="t1",
        tool_name="mcp__template__generate_markdown_template",
        arguments={"scene": "loan_apply"},
        risk_level="read_only",
    )
    execution = tool.execute(request, route, tool_call)
    assert execution.result.status == "success"
    assert execution.result.business_status == "OK"
    assert execution.result.output["text_content"]
```

### 13.6 新建 `scripts/demo_mcp_flow.py`

```python
"""MCP 全链路 Demo。

运行：
    conda activate xinyidai-chat-agent
    python scripts/demo_mcp_flow.py

步骤：
    1) 加载 config/mcp_servers.yaml
    2) 拉起 McpManager（template + filesystem）
    3) 列出已注册 MCP 工具
    4) 调用 template.generate_markdown_template
    5) 调用 template.generate_excel_template
    6) 调用 filesystem.list_directory（./workspace）
    7) 优雅 shutdown
"""

from __future__ import annotations

import sys
from pathlib import Path

from xinyidai_agent.mcp import (
    McpManager,
    default_config_path,
    load_mcp_servers_config,
)
from xinyidai_agent.mcp.tool_adapter import McpToolAdapter
from xinyidai_agent.rag.async_runtime import AsyncRuntime


def _print_step(idx: int, title: str) -> None:
    print(f"\n[{idx}] {title}")


def main() -> int:
    _print_step(1, "加载配置")
    cfg = load_mcp_servers_config(default_config_path())
    print(f"  servers={len(cfg.servers)}: {[s.name for s in cfg.servers]}")

    _print_step(2, "拉起 McpManager")
    runtime = AsyncRuntime()
    manager = McpManager(cfg.servers, runtime)
    manager.start()

    try:
        _print_step(3, "已注册 MCP 工具")
        descriptors = manager.iter_descriptors()
        for desc in descriptors:
            print(f"  - {desc.qualified_name}  ({desc.risk_level})")

        _print_step(4, "生成 Markdown 模板")
        result = manager.call_tool_sync(
            "mcp__template__generate_markdown_template",
            {"scene": "loan_apply", "title": "贷款采集（Demo）"},
        )
        print(f"  is_error={result.is_error}  text={result.text_content}")

        _print_step(5, "生成 Excel 模板")
        result = manager.call_tool_sync(
            "mcp__template__generate_excel_template",
            {
                "columns": ["客户名称", "统一社会信用代码", "联系电话", "申请额度"],
                "filename": "customer_intake_demo",
            },
        )
        print(f"  is_error={result.is_error}  text={result.text_content}")

        _print_step(6, "列出工作区文件")
        workspace_abs = Path("workspace").resolve()
        result = manager.call_tool_sync(
            "mcp__filesystem__list_directory",
            {"path": str(workspace_abs)},
        )
        print(f"  is_error={result.is_error}")
        print(f"  text=\n{result.text_content}")

        _print_step(7, "演示完成，开始 shutdown")
    finally:
        manager.shutdown()
        runtime.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### 13.7 验收

#### 单元 / 契约

```powershell
pytest test/contract/test_mcp_protocol.py test/contract/test_mcp_tool_adapter.py -v
```

**期望**：全部通过。

#### 冒烟

```powershell
pytest test/smoke/test_mcp_connectivity.py -v
```

**期望**：全部通过；测试期间不依赖 Node.js。

#### E2E

```powershell
pytest test/e2e/test_mcp_e2e.py -v
```

**期望**：全部通过。

#### Demo 脚本

```powershell
python scripts/demo_mcp_flow.py
```

**期望输出**（关键行）：

```
[1] 加载配置
  servers=2: ['template', 'filesystem']
[2] 拉起 McpManager
[3] 已注册 MCP 工具
  - mcp__template__generate_markdown_template  (read_only)
  - mcp__template__generate_excel_template     (read_only)
  - mcp__template__list_generated_templates    (read_only)
  - mcp__filesystem__read_file                 (read_only)
  - mcp__filesystem__list_directory            (read_only)
[4] 生成 Markdown 模板
  is_error=False  text=已生成 Markdown 模板：...
[5] 生成 Excel 模板
  is_error=False  text=已生成 Excel 模板：...
[6] 列出工作区文件
  is_error=False
  text=
  ...sample.csv...
[7] 演示完成，开始 shutdown
```

---

## 14. 完成判定

全部 Phase 完成后必须满足：

1. `pytest test/ -x` 在引入 MCP 后**通过数不下降**（新增测试 100% 通过；原有测试 0 失败）。
2. `ruff check src/xinyidai_agent/mcp xinyidai_mcp_servers test/contract/test_mcp_*.py test/smoke/test_mcp_*.py test/e2e/test_mcp_*.py scripts/demo_mcp_flow.py` 无 warning。
3. `python scripts/demo_mcp_flow.py` 输出完整 7 步、无异常退出。
4. FastAPI 启动日志出现两条 `MCP Server ... 已就绪` 行。
5. 任意一处征信后端 API 调用、Mock 工具改名为非 Mock、TODO 残留 → **不通过**。

---

## 15. 提交规范

每个 Phase 完成后提交一次，commit message 中文：

```
feat(mcp): Phase {N} - {主题}

新增：
- ...
修改：
- ...
验收：
- ...
```

`progress/` 目录建议存档每个 Phase 的 pytest 输出（可选，不强制）。

