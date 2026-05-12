# MCP 集成设计（综述 + 架构）

> 配套文档：`docs/design/MCP_INTEGRATION_IMPLEMENTATION_CHECKLIST.md`
> 适用对象：研发与评审；落地执行另见实施清单
> 改造范围：在 `xinyidai_agent` 内**自实现 MCP 客户端基础设施**，并提供 2 个通用 demo（自实现 Python `template_server` + 引入官方 `filesystem` MCP Server）
> **严格约束**：不接入征信后端任何接口；只演示通用工具能力（模板生成、文件导入导出）

---

## 0. 文档定位

本文档解答四个问题：

1. **MCP 是什么 / 为什么要用**（综述）
2. **本项目集成的边界与选型**（决策依据）
3. **目标架构与关键数据契约**（接口）
4. **演示用例与验收口径**（可见交付）

落地步骤、文件级清单、验收命令一律在 `MCP_INTEGRATION_IMPLEMENTATION_CHECKLIST.md`，本文不重复。

---

## 1. MCP 协议综述

### 1.1 是什么

**Model Context Protocol（MCP）** 是 Anthropic 在 2024 年 11 月开源、并由社区维护的协议，标准化 LLM 应用与外部能力（工具/数据/Prompt 模板）的对接。把 LLM 应用 × 工具 的 `M×N` 集成复杂度降到 `M+N`：能力提供方实现一次 MCP Server，应用方实现一次 MCP Client，即可互联。

### 1.2 三原语

| 原语 | 用途 | 本项目消费方式 |
|---|---|---|
| **Tools** | 可被模型调用的函数（有副作用） | 适配为 `AgentTool`，注入 `ToolRegistry` |
| **Resources** | 只读上下文资源（文件/文档/记录） | 暂不消费（后续可接入 evidence pool） |
| **Prompts** | 参数化 prompt 模板 | 暂不消费 |

> **本期范围**：只对接 **Tools**，Resources / Prompts 留作后续扩展。

### 1.3 传输层

- **stdio**：Server 作为本地子进程，通过 stdin/stdout 通信。最主流，本项目首选。
- **SSE**：远程 Server，单向流。生态正在淘汰。
- **Streamable HTTP**：2025 新规范，双向流式 HTTP。**留扩展位**，本期不实现。

### 1.4 消息格式

基于 **JSON-RPC 2.0**：

```jsonc
// 调用工具
{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
 "params": {"name": "read_file", "arguments": {"path": "/tmp/a.txt"}}}

// 响应
{"jsonrpc": "2.0", "id": 1,
 "result": {"content": [{"type": "text", "text": "..."}], "isError": false}}
```

### 1.5 生态现状（2026 初）

- **官方 SDK**：Python / TypeScript / Java / Kotlin / C# / Swift
- **官方 Server**（Node.js）：filesystem / git / github / slack / postgres / sentry / puppeteer 等约 30 个
- **采用方**：Claude Desktop / Cursor / Windsurf / Cline / Continue / Zed / Replit

---

## 2. 选型与边界

### 2.1 范围内（本期交付）

| 项 | 决策 |
|---|---|
| MCP 角色 | **Client 优先**：项目作为 MCP 消费方 |
| 传输 | **仅 stdio**（覆盖 ≥90% 场景，远程 server 留后续） |
| 三原语 | **仅 Tools** |
| Demo Server A | **自实现** `xinyidai_mcp_servers/template_server`（Python），演示能力："生成 Excel/Markdown 模板文件" |
| Demo Server B | **引入官方** `@modelcontextprotocol/server-filesystem`（Node.js），演示能力："读写工作目录文件，模拟文件导入导出" |
| 治理（熔断/灰度/版本灰度） | **不做** |
| 写工具高风险闭环 | **保留**：MCP 工具的 `risk_level` 与现有 `requires_confirmation` 联动，复用 `PendingActionValidator` |

### 2.2 范围外（明确不做）

- ❌ 不接入征信后端任何 API
- ❌ 不实现 MCP Resources / Prompts 消费
- ❌ 不实现 SSE / Streamable HTTP transport
- ❌ 不做熔断、灰度、版本管理、A/B
- ❌ 不实现 Skill bundle 升级（与 MCP 解耦，单独立项）

### 2.3 为什么自实现一个 MCP Server？

**两个目的**：

1. **基础设施验证**：证明项目对接的不只是某个特定 Server，而是任意符合 MCP 协议的实现
2. **演示自闭环能力**：以"生成业务模板"为典型场景（金融业务里非常常见的需求：批量数据导入模板、客户资料采集模板），展示 Agent + MCP 的最小价值闭环

---

## 3. 目标架构

### 3.1 模块视图

```
┌─────────────────────────────────────────────────────────────────┐
│                      ReactRuntime 主循环                         │
│                                                                  │
│   ┌──────────────┐         ┌───────────────────────────────┐    │
│   │ CapabilityCtx│────────→│        ToolRegistry           │    │
│   └──────────────┘         │                               │    │
│                            │  ┌──────────────────────────┐ │    │
│                            │  │  本地 AgentTool          │ │    │
│                            │  │  - RagSearchTool         │ │    │
│                            │  │  - MockCreditAmountTool  │ │    │
│                            │  └──────────────────────────┘ │    │
│                            │  ┌──────────────────────────┐ │    │
│                            │  │  McpToolAdapter（适配层）│ │    │
│                            │  │  - mcp__fs__read_file    │ │    │
│                            │  │  - mcp__tpl__gen_excel   │ │    │
│                            │  └──────────────────────────┘ │    │
│                            └────────────────┬──────────────┘    │
└─────────────────────────────────────────────┼───────────────────┘
                                              │ execute_observed
                            ┌─────────────────┴─────────────────┐
                            │           McpManager              │
                            │  - configs: list[ServerConfig]    │
                            │  - clients: dict[name, McpClient] │
                            │  - 复用 AsyncRuntime 后台 loop     │
                            └─────────┬───────────────────┬─────┘
                                      │                   │
                          ┌───────────▼────────┐ ┌────────▼────────┐
                          │  McpClient(stdio)  │ │ McpClient(stdio)│
                          └───────────┬────────┘ └────────┬────────┘
                                      │ subprocess         │ subprocess
                          ┌───────────▼────────┐ ┌────────▼─────────┐
                          │ template_server.py │ │ npx @modelcontext│
                          │ （Python 自实现）  │ │ protocol/server- │
                          │                    │ │ filesystem       │
                          └────────────────────┘ └──────────────────┘
```

### 3.2 数据流（一次工具调用）

```
1. ReactStepEngine 决定 call_tool("mcp__tpl__gen_excel", {...})
2. ToolRegistry.execute_observed() 查表 → McpToolAdapter
3. McpToolAdapter.execute() 调用 McpManager.call_tool_sync()
4. McpManager 通过 AsyncRuntime.run_coroutine 提交到后台 loop
5. McpClient 在后台 loop 内通过 stdio 把 JSON-RPC 发给子进程 Server
6. Server 执行 → 返回 content (text/blob)
7. McpClient 解析 → McpToolCallResult
8. McpToolAdapter 包装为 ToolResult/ToolExecution（含 envelope/business_status）
9. ToolRegistry 包成 ToolExecutionObservation 返回 ReactStepEngine
10. ReactStepEngine 沿用现有反幻觉护栏（schema/证据/重复阻断）
```

### 3.3 进程模型

- **主进程**：FastAPI + ReactRuntime
- **子进程 ×N**：每个 MCP Server 一个，由 `McpManager.start()` 拉起，`shutdown()` 优雅退出
- **后台线程 ×1**：`AsyncRuntime`（已存在，整个项目共用）

---

## 4. 关键设计决策

### 4.1 Decision-1：复用 `ToolCategory = "utility"`，不扩枚举

**问题**：MCP 工具是否需要新增 `ToolCategory = "mcp"` 枚举？

**决策**：**不新增**，复用 `utility`。

**理由**：
- 现有 `ToolCategory` 已有 `utility`，语义匹配
- 新增枚举会触动 `protocol.py` / capability 白名单 / 多处 Literal 校验，违背"最小改动"
- MCP 来源信息保留在 `qualified_name` 前缀（`mcp__{server}__{tool}`）与 `McpToolAdapter._descriptor.server_name` 字段，可观测性不丢

### 4.2 Decision-2：qualified_name 命名规则

格式：`mcp__{server_name}__{original_tool_name}`

- 双下划线分隔避免与单下划线工具名冲突
- 前缀 `mcp__` 明示来源，便于运行时日志、trace、白名单匹配
- 单工具名内 `_` 和 `-` 一律保留原样

**示例**：
- `mcp__filesystem__read_file`
- `mcp__template__generate_excel_template`

### 4.3 Decision-3：RiskLevel 映射

MCP 协议本身不带 `risk_level`，由我方在 `McpServerConfig.risk_level` 显式声明，**作用于该 Server 下所有工具**。映射到 `ToolSpec.risk_level`：

| `McpServerConfig.risk_level` | `ToolSpec.risk_level` | `requires_confirmation` |
|---|---|---|
| `read_only` | `read_only` | `False` |
| `state_create` | `state_create` | `True` |
| `state_update` | `state_update` | `True` |
| `final_submit` | `final_submit` | `True` |

- **Demo `filesystem` server**：限制为 `read_only`（白名单只允许 `read_file` / `list_directory`，不开放 `write_file`，避免不可逆动作）
- **Demo `template` server**：限制为 `read_only`（生成模板到工作区，不修改外部状态；幂等可重入）

### 4.4 Decision-4：fail-closed 默认

- `McpManager.start()` 任一 Server 启动失败 → 该 Server 工具**不注册**进 Registry，但**不阻塞** Agent 启动；其它 Server 正常工作
- 运行期 `McpClient.call_tool()` 抛异常 → 由 `McpToolAdapter` 转换为 `ToolResult(status="failed", business_status="TOOL_UNAVAILABLE")`，模型看到的是标准 `ToolExecutionObservation(kind="failed")`，进入现有兜底路径
- 模型输入参数若不符 `inputSchema` → `business_status="TOOL_INPUT_INVALID"`，`kind="schema_error"`

### 4.5 Decision-5：input_schema 透传

MCP `inputSchema` 是 JSON Schema 格式，直接塞进 `ToolSpec.input_schema` 字段（已有，类型 `dict[str, object]`）。`input_slots` 由 JSON Schema **顶层 properties** 自动派生为 `SlotSpec` 列表（required 字段从 `required` 数组判定）。

### 4.6 Decision-6：跨线程异步执行

`AsyncRuntime` 已成熟，**复用即可**：

- `McpClient` 内部全 async（用官方 `mcp` SDK）
- `McpManager` 持有 `AsyncRuntime` 引用
- 同步调用入口：`McpManager.call_tool_sync(qualified_name, arguments)` 内部走 `runtime.run_coroutine(self._call_async(...))`
- 主循环 / FastAPI handler 调 sync 接口，无感知 async

### 4.7 Decision-7：生命周期

- **启动期**（`api.py` lifespan）：
  1. 加载 `config/mcp_servers.yaml`
  2. `McpManager(configs, async_runtime).start()` → 并发拉起所有 Server，并 `list_tools` 拉到本地缓存
  3. 把 `McpManager.iter_adapters()` 的产物 `register` 进 `ToolRegistry`
- **运行期**：工具调用直接走 Registry，无感知 MCP
- **关闭期**：`McpManager.shutdown()` → 关闭所有 stdio session → 子进程优雅退出

---

## 5. 与现有系统融合

### 5.1 与 `AgentTool` 协议

`McpToolAdapter` 实现 `AgentTool` Protocol 的所有成员（`name / category / risk_level / description / requires_confirmation / execute() / spec()`），主循环零感知。

### 5.2 与反幻觉护栏

| 护栏 | MCP 工具是否复用 | 说明 |
|---|---|---|
| Pydantic schema 锁死 | ✅ | adapter 产出标准 `ToolResult` |
| 证据强制 | ⚠️ 不强制 | 通用工具非 RAG 类，不进 evidence pool |
| 重复阻断 | ✅ | `qualified_name + arguments_hash` 进入现有签名表 |
| 步数兜底 | ✅ | 主循环统一 |
| pending_action 两阶段确认 | ✅ | 高 `risk_level` 自动 `requires_confirmation=True` |

### 5.3 与 `RouteDecision.allowed_tool_categories`

MCP 工具归入 `utility` 类别。Capability 想暴露 MCP 工具时，`allowed_tool_categories` 须包含 `utility`。

### 5.4 与配置

新增 `config/mcp_servers.yaml`，由 `McpServersConfig.from_yaml(path)` 加载。环境变量 `MCP_CONFIG_PATH` 覆盖默认路径；`MCP_ENABLED=0` 关闭全部 MCP（应急开关）。

---

## 6. 数据模型契约

### 6.1 `xinyidai_agent.mcp.protocol`

```python
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

from xinyidai_agent.protocol import RiskLevel


McpTransport = Literal["stdio"]
# 后续扩展位：Literal["stdio", "sse", "streamable_http"]


class McpServerConfig(BaseModel):
    """单个 MCP Server 的配置，frozen 保证不可变。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    """Server 标识，用于命名 qualified_name 与日志检索。允许字符：a-z 0-9 _。"""

    transport: McpTransport = "stdio"

    command: str | None = None
    """stdio transport 必填：拉起子进程的可执行文件（如 npx、python）。"""

    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    allowed_tools: list[str] = Field(default_factory=list)
    """白名单，空数组表示该 Server 暴露的所有工具都不允许；必须显式声明。"""

    risk_level: RiskLevel = "read_only"
    """该 Server 下所有工具继承的风险等级。"""

    description: str = ""
    """运维语义说明，将注入工具 description 前缀。"""

    timeout_seconds: float = 30.0
    enabled: bool = True


class McpServersConfig(BaseModel):
    """整体配置文件，扁平管理。"""

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
    """MCP Server 工具调用结果（已规范化）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_error: bool
    text_content: str = ""
    """所有 type=text 的内容拼接（按顺序保留，分行加 \\n）。"""

    structured: dict[str, Any] = Field(default_factory=dict)
    """部分 Server 会返回 structuredContent；无则空。"""

    raw_content: list[dict[str, Any]] = Field(default_factory=list)
    """原始 content 数组，便于 trace 与排障。"""
```

### 6.2 `xinyidai_agent.mcp.client`

```python
class McpClient:
    """单 Server stdio 客户端。所有方法 async。"""

    def __init__(self, config: McpServerConfig) -> None: ...

    async def connect(self) -> None:
        """建立 stdio 子进程 + ClientSession.initialize()。"""

    async def list_tools(self) -> list[McpToolDescriptor]:
        """调用 session.list_tools()，过滤 allowed_tools 白名单。"""

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> McpToolCallResult:
        """调 session.call_tool()，将结果规范化。"""

    async def aclose(self) -> None:
        """优雅关闭 session 与子进程。"""
```

### 6.3 `xinyidai_agent.mcp.manager`

```python
class McpManager:
    """多 MCP Server 生命周期管理 + 同步接口适配。"""

    def __init__(
        self,
        configs: list[McpServerConfig],
        async_runtime: AsyncRuntime,
    ) -> None: ...

    def start(self) -> None:
        """同步阻塞：拉起所有 enabled 的 Server，list_tools 缓存。
        单个 Server 失败不影响其它 Server。失败信息通过日志暴露。
        """

    def shutdown(self) -> None: ...

    def iter_descriptors(self) -> list[McpToolDescriptor]: ...

    def call_tool_sync(
        self, qualified_name: str, arguments: dict[str, Any]
    ) -> McpToolCallResult: ...
```

### 6.4 `xinyidai_agent.mcp.tool_adapter`

```python
class McpToolAdapter:
    """实现 AgentTool Protocol，将 MCP 工具适配到现有工具系统。"""

    def __init__(self, descriptor: McpToolDescriptor, manager: McpManager) -> None:
        self._descriptor = descriptor
        self._manager = manager
        self.name = descriptor.qualified_name
        self.category: ToolCategory = "utility"
        self.risk_level = descriptor.risk_level
        self.description = descriptor.description
        self.requires_confirmation = descriptor.risk_level != "read_only"

    def spec(self) -> ToolSpec:
        """根据 input_schema 派生 SlotSpec 列表。"""

    def execute(
        self,
        request: ChatRequest,
        route: RouteDecision,
        tool_call: ToolCall,
    ) -> ToolExecution:
        """同步调用 manager.call_tool_sync()，包装为 ToolResult。"""
```

---

## 7. 安全与失败处理

### 7.1 白名单三层

1. **Config 层**：`McpServerConfig.allowed_tools` 启动期过滤
2. **Adapter 层**：单个 adapter 一对一映射，不存在的 tool_name 根本无 adapter
3. **Capability 层**：现有 `RouteDecision.allowed_tool_categories=["utility"]` 才暴露

### 7.2 失败映射表

| 场景 | `ToolResult.status` | `business_status` | `ToolExecutionObservation.kind` |
|---|---|---|---|
| Server 启动失败 | （未注册） | — | 模型看不到该工具 |
| 调用超时 | `failed` | `TOOL_TIMEOUT` | `failed` |
| Server 返回 isError=true | `failed` | `TOOL_FAILED` | `failed` |
| inputSchema 校验失败 | `failed` | `TOOL_INPUT_INVALID` | `schema_error` |
| 工具名未注册 | `failed` | `TOOL_UNAVAILABLE` | `unavailable` |
| 成功且有 text_content | `success` | `OK` | `success` |
| 成功但内容为空 | `success` | `PARTIAL_DATA` | `empty` |

> 注：若 `BusinessStatus` Literal 暂未含 `TOOL_TIMEOUT` / `TOOL_INPUT_INVALID` / `TOOL_FAILED`，由实施清单 Phase 1 统一追加。

### 7.3 子进程稳定性

- 启动期 `connect()` 单 Server 超时 10s
- 子进程异常退出（SIGCHLD）→ `McpManager` 标记该 Server `unavailable`，相关 adapter 调用直接返回 `TOOL_UNAVAILABLE`
- 不做自动重连（本期范围外）

---

## 8. 演示用例（验收口径）

### 8.1 Demo Server A：`template_server`（自实现 Python）

**位置**：`xinyidai_mcp_servers/template_server/server.py`（独立子包，不污染 `xinyidai_agent`）

**对外工具**：

1. `generate_markdown_template`
   - 入参：`scene: str`（如 `loan_apply` / `customer_intake`）、`title: str`
   - 输出：写入 `.data/mcp_templates/{scene}_{ts}.md`，content 为 Markdown 模板文本
2. `generate_excel_template`
   - 入参：`columns: list[str]`、`filename: str`
   - 输出：写入 `.data/mcp_templates/{filename}.xlsx`，使用 `openpyxl` 生成首行标题模板
3. `list_generated_templates`
   - 入参：无
   - 输出：返回 `.data/mcp_templates/` 下的文件清单

### 8.2 Demo Server B：`filesystem`（官方）

**启动方式**：`npx -y @modelcontextprotocol/server-filesystem ./workspace`

**白名单工具**：`read_file`、`list_directory`

**演示能力**：模型读取用户提供的 CSV / Markdown，作为"文件导入"的最小闭环。

### 8.3 演示脚本

`scripts/demo_mcp_flow.py`：

```text
1) 启动 AsyncRuntime + McpManager（载入两份 server 配置）
2) 列出 ToolRegistry 中以 mcp__ 开头的工具，打印数量与签名
3) 直接调用 mcp__template__generate_markdown_template
   输入 scene=loan_apply, title="贷款申请采集模板"
   断言文件生成且 size > 0
4) 启动 ReactRuntime，模拟用户输入：
   "请为客户资料采集生成一份 Excel 模板，列包含：客户名称、统一社会信用代码、联系电话、申请额度"
   断言 Agent 走 ReAct 一步选中 mcp__template__generate_excel_template 并返回路径
5) 模拟用户输入：
   "请把工作区 sample.csv 的前 5 行读出来"
   断言 Agent 走 mcp__filesystem__read_file 并返回内容
6) 优雅 shutdown
```

### 8.4 验收口径

| 项 | 标准 |
|---|---|
| 单元测试 | `pytest test/contract/test_mcp_*.py -v` 全部通过 |
| 冒烟测试 | `pytest test/smoke/test_mcp_connectivity.py -v` 通过 |
| E2E 测试 | `pytest test/e2e/test_mcp_e2e.py -v` 通过 |
| Demo 脚本 | `python scripts/demo_mcp_flow.py` 输出 6 步全部 ✓ |
| Lint | `ruff check src/xinyidai_agent/mcp xinyidai_mcp_servers test/contract/test_mcp_*.py` 0 warning |

---

## 9. 不做的事（明确边界）

| 不做的事 | 替代或说明 |
|---|---|
| 接入征信后端 API | 明确禁止，演示用通用工具 |
| MCP Resources / Prompts | 后续单独立项 |
| SSE / Streamable HTTP transport | 协议层留扩展位，本期只实现 stdio |
| 熔断 / 灰度 / 版本管理 | 本期范围外 |
| MCP Server 自动重启 | 本期范围外 |
| Skills v2 bundle 升级 | 与 MCP 解耦，单独立项 |
| 沙箱执行 | 不在本期；Demo Server 仅做幂等写入到固定目录 |

---

## 10. 风险与提醒

- **依赖 Node.js**：官方 `@modelcontextprotocol/server-filesystem` 需 `node`/`npx`。环境检查脚本会先校验。
- **Windows stdio 兼容**：Python `mcp` SDK 已封装；`AsyncRuntime` 已对 Windows ProactorEventLoop 做了规避（切换 SelectorEventLoop）。
- **首次启动慢**：`npx -y` 可能联网下载 server，建议在本地 prebuilt 一次。

---

## 11. 后续扩展点（不阻塞本期）

1. MCP Resources → 接入 evidence pool（与 RAG 同权重）
2. SSE / Streamable HTTP transport（接入远端 Server）
3. MCP Server 健康检查 + 自动重连 + 熔断
4. Skill Bundle 升级（让 Skill 声明 `depends_on_mcp` 实现能力组合）
5. 同一 MCP 工具的灰度（A/B 选择不同 Server 实现）
