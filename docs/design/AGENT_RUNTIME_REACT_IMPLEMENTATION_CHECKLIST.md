# Agent Runtime ReAct 实现清单（Codex 可执行）

> 配套文档：`docs/design/AGENT_RUNTIME_REACT_DESIGN.md`  
> 适用对象：可独立读取代码的 Coding Agent（Codex / Claude Code 等）  
> 编码语言：Python 3.11+，使用项目已有的 conda 环境  
> 严格要求：**生产级代码、最终态、不允许 mock 占位、不允许 TODO 留白、不允许双轨兼容**

---

## 0. 总则与约束

### 0.1 不可违反的规则

1. **禁止 mock**：所有新工具、新组件必须给出最终实现。`MockCreditAmountTool` 是已存在的占位，本次不动；新增工具不允许命名以 `Mock` 开头。
2. **禁止 TODO**：清单要求完成的代码不允许留 `TODO/FIXME/pass/...`。
3. **禁止旧路径**：本次改造为**单轨切换**——`RuntimeStepPolicy` / `RuntimeStepDecision` 在 Phase 8 完成后**整体删除**，不保留 deprecated 双轨。
4. **禁止幻觉路径**：`runtime._answer_without_tool`、`_plan_tool`、`_parse_tool_plan`、`_build_deterministic_tool_call` 在 Phase 8 完成后整体删除。回答路径只有：`fast_path` + `react_loop`，无第三条。
5. **fail-closed**：所有新增字段、新增枚举的默认值都向"拒绝/不允许"倾斜；放行必须显式声明。
6. **测试先行**：每个 Phase 的验收命令必须 100% 通过才能进入下一 Phase。
7. **中文注释**：所有新增代码必须有类级、方法级、关键逻辑的中文注释。
8. **并发安全**：所有共享状态用 `frozen=True` 的 pydantic / dataclass；运行时状态通过显式参数传递，不放模块级全局。

### 0.2 项目环境

- 工作目录：`d:\companyCode\agent`
- Python 环境：conda env `xinyidai-agent`（**禁止 base 环境**）
- 测试命令：
  ```powershell
  conda activate xinyidai-agent
  pytest test/ -x -v
  ```

### 0.3 命名规范

- 新增主循环目录：`src/xinyidai_agent/runtime/`（包）
- 文件名：`react_engine.py` / `react_loop.py` / `react_prompt.py` / `react_observation_builder.py` / `fast_path.py`
- L0/L1：`router/rules_guard.py` / `router/scene_direct.py`
- 类名：`ReactStepEngine` / `ReactRuntime` / `SceneDirectDispatcher` / `RulesGuard` / `FastPathRunner`

---

## 1. 改动总览（执行顺序 = 依赖顺序）

| Phase | 主题 | 新增 | 修改 | 删除 |
|---|---|---|---|---|
| 0 | 基线快照 | - | - | - |
| 1 | 协议层 | `react_observation.py` | `protocol.py` | - |
| 2 | Capability 扩展 | - | `capabilities/base.py` `capabilities/catalog.py` | - |
| 3 | ToolSpec fail-closed | - | `tools/base.py` 全部已注册工具 | - |
| 4 | Registry → Observation | - | `tools/registry.py` | - |
| 5 | L0 + L1 短路 | `router/rules_guard.py` `router/scene_direct.py` | `router/service.py` `router/__init__.py` | - |
| 6 | L3-fast 快速通道 | `runtime/fast_path.py` | - | - |
| 7 | ReactStepEngine | `runtime/react_engine.py` `runtime/react_prompt.py` `runtime/react_observation_builder.py` | - | - |
| 8 | 主循环切换 | `runtime/react_loop.py` | `runtime.py` `api.py` | `runtime_policy.py` |
| 9 | 全量测试矩阵 | `test/contract/test_react_*.py` `test/e2e/test_react_loop_matrix.py` | - | 旧 step_policy 测试 |

---

## 2. Phase 0：基线快照

**目标**：进入改造前冻结测试基线。

### 任务

1. `progress.txt` 追加：`[REACT_REFAC_START] <UTC ISO 时间>` + 当前 pytest 通过数 + smoke 路由准确数。
2. 跑完整测试并保存结果到 `progress/baseline_pytest.txt`：
   ```powershell
   pytest test/ --tb=short -q | Tee-Object -FilePath ".\progress\baseline_pytest.txt"
   ```
3. 跑诊断脚本并保存到 `progress/baseline_router.txt`：
   ```powershell
   python scripts/diag_intent_router.py | Tee-Object -FilePath ".\progress\baseline_router.txt"
   ```

### 验收

`baseline_pytest.txt` 末行包含 `passed`，`failed=0`。

---

## 3. Phase 1：协议层

**目标**：补齐 ReAct 主循环必需的协议类型，`protocol.py` 只增不删。

### 3.1 修改 `src/xinyidai_agent/protocol.py`

**步骤 A**：在 `RiskLevel` 之后（约第 70 行）追加 3 个枚举：

```python
ReactActionType = Literal["call_tool", "answer", "ask_user", "handoff"]
ReactObservationKind = Literal[
    "success",       # 工具成功且业务结果可用
    "empty",         # 工具成功但业务结果为空（NOT_FOUND / PARTIAL_DATA）
    "failed",        # 工具执行失败（5xx / timeout / 抛异常）
    "unavailable",   # 工具未在 ToolRegistry 注册
    "blocked",       # 工具被 capability 白名单或风险策略阻断
    "schema_error",  # 输入或输出 schema 校验失败
]
ReactTerminalReason = Literal[
    "completed",
    "answered_no_tool",
    "ask_user",
    "wait_confirmation",
    "handoff",
    "tool_blocked",
    "max_steps",
    "runtime_error",
]
```

**步骤 B**：在 `BusinessStatus` Literal 中追加 `"TOOL_UNAVAILABLE"`。

**步骤 C**：在文件末尾追加 4 个模型（必须使用 `extra="forbid"`）：

```python
class ReactAction(BaseModel):
    """模型每轮 reasoning 的输出契约。

    四种 action 互斥：
    - call_tool: 必须给出 tool_name + arguments
    - answer:    必须给出 final_answer；KNOWLEDGE_QA 场景必须填 evidence_used
    - ask_user / handoff: 必须给出 message
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: ReactActionType
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    final_answer: str | None = None
    evidence_used: list[str] = Field(default_factory=list)
    message: str | None = None


class ReactStepDecision(BaseModel):
    """模型每轮的完整 reasoning 输出。"""
    model_config = ConfigDict(frozen=True, extra="forbid")
    thought: str = Field(default="", max_length=500)
    action: ReactAction
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ToolExecutionObservation(BaseModel):
    """单次工具执行结果的结构化封装，供模型下一轮推理使用。

    任何工具调用尝试（成功 / 失败 / 阻断 / 未注册）都必须产出一个观察，
    不允许把工具异常或 None 暴露给上层循环。
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: ReactObservationKind
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    business_status: str | None = None
    sources_count: int = 0
    reason: str | None = None
    hint: str | None = None
    alternative_tools: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0


class ReactTerminal(BaseModel):
    """ReAct 循环的终止决策。"""
    model_config = ConfigDict(frozen=True, extra="forbid")
    reason: ReactTerminalReason
    final_answer: str
    sources: list[SourceDocument] = Field(default_factory=list)
    pending_action: PendingAction | None = None
    handoff_reason: str | None = None
    missing_slots: list[str] = Field(default_factory=list)
    error_class: str | None = None
    error_message: str | None = None
```

### 3.2 新增 `src/xinyidai_agent/react_observation.py`

```python
"""ReAct 主循环喂给模型的 observation 容器。

ToolExecutionObservation 描述单次工具执行；
ReactObservation 是模型每轮收到的完整上下文。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.protocol import (
    ChatRequest,
    RouteDecision,
    SessionStateSnapshot,
    ToolExecutionObservation,
)


class ToolSpecBrief(BaseModel):
    """喂给模型的工具说明（裁剪版 ToolSpec）。"""
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    description: str
    risk_level: str
    is_read_only: bool
    input_schema: dict[str, Any] = Field(default_factory=dict)
    required_arguments: list[str] = Field(default_factory=list)
    already_executed_count: int = 0


class ReactObservation(BaseModel):
    """每轮喂给模型的完整观察。"""
    model_config = ConfigDict(frozen=True, extra="forbid")
    step: int
    max_steps: int
    request: ChatRequest
    route: RouteDecision
    capability: CapabilityPolicy
    session_state: SessionStateSnapshot
    available_tools: list[ToolSpecBrief]
    executed_tools: list[ToolExecutionObservation] = Field(default_factory=list)
    last_observation: ToolExecutionObservation | None = None
    invalid_action_hint: str | None = None
```

### 3.3 验收

新增 `test/contract/test_react_protocol.py`：

```python
def test_react_action_extra_forbid(): ...                # 多余字段被拒
def test_react_action_call_tool_minimal(): ...           # call_tool 至少包含 tool_name
def test_react_observation_kind_enumerable(): ...        # 6 种 kind 全可构造
def test_react_terminal_reason_enumerable(): ...         # 8 种 reason 全可构造
def test_business_status_includes_tool_unavailable(): ...
def test_react_step_decision_default_confidence(): ...
```

执行 `pytest test/contract/test_react_protocol.py -v` 全部通过。

---

## 4. Phase 2：CapabilityPolicy 扩展

**目标**：在 capability 上声明 `fast_path_eligible` 等元信息，driver runtime 路由策略。

### 4.1 修改 `src/xinyidai_agent/capabilities/base.py`

在 `CapabilityPolicy` 末尾追加 3 个字段 + 1 个方法：

```python
@dataclass(frozen=True)
class CapabilityPolicy:
    capability_id: str
    scene: RouteScene
    standard_intent: str
    description: str
    required_slots: list[str] = field(default_factory=list)
    optional_slots: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    allowed_tool_categories: list[ToolCategory] = field(default_factory=list)
    risk_level: RiskLevel = "read_only"
    confirmation_required: bool = False
    min_confidence: float = 0.7

    # —— Phase 2 新增字段 ——
    fast_path_eligible: bool = False
    """单工具只读快速通道：满足条件时走 fast_path，跳过 ReAct 多轮。"""

    requires_evidence: bool = False
    """answer 必须基于工具检索到的 sources（如 KNOWLEDGE_QA），否则视为幻觉。"""

    max_react_steps: int = 4
    """该 capability 单次 ReAct 循环允许的最大步数（不含系统重试）。"""

    def is_fast_path_eligible_for_route(self, route_filled_slots: dict) -> bool:
        """运行时判断：fast_path_eligible 且必填 slots 全部填齐。"""
        if not self.fast_path_eligible:
            return False
        if len(self.allowed_tools) != 1:
            return False
        if self.risk_level != "read_only":
            return False
        for slot in self.required_slots:
            value = route_filled_slots.get(slot)
            if value is None or (isinstance(value, str) and not value.strip()):
                return False
        return True
```

### 4.2 修改 `src/xinyidai_agent/capabilities/catalog.py`

为每个 capability 显式声明三个新字段：

| capability_id | fast_path_eligible | requires_evidence | max_react_steps |
|---|---|---|---|
| `knowledge.policy.read` | True | True | 2 |
| `credit.limit.read` | False | False | 4 |
| `product.terms.read` | False | False | 4 |
| `authorization.link.create` | False | False | 4 |
| `application.draft.create` | False | False | 6 |
| `application.status.read` | True | False | 2 |
| `smalltalk.respond` | False | False | 1 |
| `unknown.clarify` | False | False | 1 |

**禁止**使用默认值兜底——必须在 catalog 中**逐项显式声明**，方便后续审计。

### 4.3 验收

新增 `test/contract/test_capability_policy.py`：

```python
def test_all_capabilities_load_with_required_fields():
    """全部 8 个 capability 都能加载且新字段非默认。"""

def test_fast_path_eligible_when_slots_filled():
    """knowledge.policy.read：slots 填齐返回 True。"""

def test_fast_path_blocked_when_slots_missing():
    """application.status.read：缺 company_name 返回 False。"""

def test_fast_path_blocked_when_capability_disabled():
    """credit.limit.read：fast_path_eligible=False 永远返回 False。"""

def test_fast_path_blocked_when_multi_tools():
    """allowed_tools 长度 > 1 时不允许 fast_path（防误配置）。"""

def test_fast_path_blocked_when_not_read_only():
    """risk_level != read_only 时不允许 fast_path。"""

def test_capability_max_react_steps_positive():
    """max_react_steps 必须 ≥ 1。"""
```

执行 `pytest test/contract/test_capability_policy.py -v` 全部通过。

---

## 5. Phase 3：ToolSpec fail-closed

**目标**：把 claude-code 的 `isReadOnly` / `isConcurrencySafe` / `isDestructive` 引入 `ToolSpec`，所有现存工具显式声明。

### 5.1 修改 `src/xinyidai_agent/tools/base.py`

```python
from typing import Literal

@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: ToolCategory
    risk_level: RiskLevel
    description: str
    requires_confirmation: bool = False
    input_slots: list[SlotSpec] = field(default_factory=list)
    output_slots: list[SlotSpec] = field(default_factory=list)
    input_schema: dict[str, object] = field(default_factory=dict)

    # —— Phase 3 新增 fail-closed 字段 ——
    is_read_only: bool = False
    """是否只读。默认 False（fail-closed），只读工具必须显式声明 True。"""

    is_idempotent: bool = False
    """是否幂等。同 (name, args) 多次调用结果一致才能声明 True。"""

    is_concurrency_safe: bool = False
    """是否可并发。涉及共享状态的工具必须保留 False。"""

    cost_class: Literal["cheap", "medium", "expensive"] = "medium"
    """单次调用成本档位，供调度器与重试策略参考。"""

    max_duration_ms: int = 5000
    """单次调用硬超时，超出由 runtime 强制截断。"""

    def required_argument_names(self) -> list[str]:
        """返回必填入参名称列表。"""
        return [s.name for s in self.input_slots if s.required]
```

### 5.2 现有工具显式声明新字段

修改 `tools/rag_search.py` 的 `RagSearchTool.spec()`：

```python
return ToolSpec(
    name="rag_search",
    category="knowledge",
    risk_level="read_only",
    description="检索信易贷政策、产品、规则知识库。",
    is_read_only=True,
    is_idempotent=True,
    is_concurrency_safe=True,
    cost_class="cheap",
    max_duration_ms=3000,
    input_slots=[
        SlotSpec(name="query", value_type="string", required=True, description="检索 query"),
        SlotSpec(name="top_k", value_type="integer", required=False, description="返回条数"),
    ],
    output_slots=[...],
)
```

修改 `tools/mock_credit.py` 的 `MockCreditAmountTool.spec()`：

```python
return ToolSpec(
    name="query_credit_amount",
    category="data_query",
    risk_level="read_only",
    description="查询企业授信额度（当前为本地占位实现）。",
    is_read_only=True,
    is_idempotent=True,
    is_concurrency_safe=True,
    cost_class="medium",
    max_duration_ms=5000,
    input_slots=[
        SlotSpec(name="company_name", value_type="string", required=True),
    ],
    output_slots=[...],
)
```

### 5.3 验收

新增 `test/contract/test_tool_spec_fail_closed.py`：

```python
def test_tool_spec_defaults_are_fail_closed():
    spec = ToolSpec(name="x", category="utility", risk_level="read_only", description="")
    assert spec.is_read_only is False
    assert spec.is_idempotent is False
    assert spec.is_concurrency_safe is False
    assert spec.cost_class == "medium"
    assert spec.max_duration_ms == 5000

def test_registered_tools_declare_metadata_explicitly():
    """default_tool_registry 加载的工具的 spec 都显式声明了新字段（断言 True）。"""

def test_required_argument_names_filters_optional():
    ...

def test_tool_spec_is_frozen():
    """ToolSpec 是 frozen dataclass。"""
```

---

## 6. Phase 4：ToolRegistry 输出 ToolExecutionObservation

**目标**：新增 `execute_observed()` 方法，永远返回 `(ToolExecution, ToolExecutionObservation)` 二元组。原 `execute()` 保留不变（confirm flow 还在用）。

### 6.1 修改 `src/xinyidai_agent/tools/registry.py`

**步骤 A**：文件头新增导入：

```python
import time

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.protocol import (
    ...,
    ReactObservationKind,
    ToolExecutionObservation,
)
```

**步骤 B**：新增 `execute_observed` 方法（在 `execute` 之后）：

```python
def execute_observed(
    self,
    request: ChatRequest,
    route: RouteDecision,
    tool_call: ToolCall,
    *,
    capability: CapabilityPolicy,
    session_state: SessionStateSnapshot | None = None,
    confirmed_action: PendingAction | None = None,
    pending_action_validated: bool = False,
) -> tuple[ToolExecution, ToolExecutionObservation]:
    """执行工具并产出结构化 observation。

    永远返回 (ToolExecution, ToolExecutionObservation)，即便工具未注册。
    内部职责：
    1. 工具未注册 → kind=unavailable，构造空 ToolExecution + alt_tools + hint
    2. 走 execute() 拿 ToolResult
    3. 把 ToolResult.status / business_status 转成 ReactObservationKind
    """
    started = time.perf_counter()

    if tool_call.tool_name not in self._tools:
        observation = ToolExecutionObservation(
            kind="unavailable",
            tool_name=tool_call.tool_name,
            arguments=dict(tool_call.arguments),
            reason=f"{tool_call.tool_name} 未在工具注册表接入。",
            alternative_tools=[
                t for t in capability.allowed_tools
                if t in self._tools and t != tool_call.tool_name
            ],
            hint=self._build_unavailable_hint(capability, tool_call.tool_name),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        empty_execution = ToolExecution(result=self._unavailable_result(tool_call))
        return empty_execution, observation

    execution = self.execute(
        request, route, tool_call,
        session_state=session_state,
        confirmed_action=confirmed_action,
        pending_action_validated=pending_action_validated,
    )
    observation = self._build_observation_from_execution(
        tool_call, execution, started_at=started,
    )
    return execution, observation
```

**步骤 C**：新增 4 个辅助方法：

```python
def _build_observation_from_execution(
    self,
    tool_call: ToolCall,
    execution: ToolExecution,
    started_at: float,
) -> ToolExecutionObservation:
    """把 ToolResult 翻译成 ToolExecutionObservation。"""
    result = execution.result
    duration_ms = (time.perf_counter() - started_at) * 1000

    if result.status == "blocked":
        return ToolExecutionObservation(
            kind="blocked",
            tool_name=tool_call.tool_name,
            arguments=dict(tool_call.arguments),
            reason=result.error_message or result.message or "工具被风险策略阻断",
            hint="工具被白名单或风险策略拦截，请重新选择 action。",
            duration_ms=duration_ms,
        )

    if result.status == "failed":
        kind: ReactObservationKind = (
            "schema_error"
            if result.business_status in {"TOOL_SCHEMA_ERROR"}
            else "failed"
        )
        return ToolExecutionObservation(
            kind=kind,
            tool_name=tool_call.tool_name,
            arguments=dict(tool_call.arguments),
            reason=result.error_message or result.message or "工具执行失败",
            hint="工具失败，可重试一次或改用 ask_user / handoff。",
            duration_ms=duration_ms,
        )

    # status == "success"
    if result.business_status in {"NOT_FOUND", "PARTIAL_DATA", "EMPTY"}:
        return ToolExecutionObservation(
            kind="empty",
            tool_name=tool_call.tool_name,
            arguments=dict(tool_call.arguments),
            summary=self._summarize_output(result),
            business_status=result.business_status,
            sources_count=len(execution.sources),
            hint="结果为空或不完整，请基于现有信息回答或追问用户。",
            duration_ms=duration_ms,
        )

    return ToolExecutionObservation(
        kind="success",
        tool_name=tool_call.tool_name,
        arguments=dict(tool_call.arguments),
        summary=self._summarize_output(result),
        business_status=result.business_status,
        sources_count=len(execution.sources),
        duration_ms=duration_ms,
    )

def _summarize_output(self, result: ToolResult) -> str:
    """生成 ≤ 500 字符的工具结果摘要喂给模型。

    规则：
    - 优先 result.envelope.message + 关键 data 字段
    - 不调用 LLM
    - 不截断关键 ID（如 application_id / authorization_id）
    """
    parts: list[str] = []
    if result.envelope and result.envelope.message:
        parts.append(result.envelope.message[:200])
    if result.envelope and result.envelope.data:
        for key in ("amount", "limit", "status", "url", "application_id", "authorization_id"):
            if key in result.envelope.data:
                parts.append(f"{key}={result.envelope.data[key]}")
    if not parts:
        parts.append(result.message or "工具已成功返回")
    summary = "; ".join(parts)
    return summary[:500]

def _unavailable_result(self, tool_call: ToolCall) -> ToolResult:
    """工具未注册时构造占位 ToolResult。"""
    msg = f"工具 {tool_call.tool_name} 暂未接入"
    return ToolResult(
        tool_call_id=tool_call.tool_call_id,
        tool_name=tool_call.tool_name,
        tool_category=tool_call.tool_category,
        status="failed",
        envelope=ToolResultEnvelope(
            success=False,
            status="TOOL_UNAVAILABLE",
            code=503,
            message=msg,
        ),
        error_message=msg,
        business_status="TOOL_UNAVAILABLE",
        code=503,
        message=msg,
        terminal=False,  # 关键：unavailable 不是 terminal，让模型下一轮重新决策
        user_visible_message="该数据查询暂未上线",
        model_observation=msg,
    )

def _build_unavailable_hint(self, capability: CapabilityPolicy, tool_name: str) -> str:
    """根据 capability 给出针对性提示。

    每个已知 capability 必须有硬编码 hint，不允许通用模板。
    """
    hints: dict[str, str] = {
        "product.terms.read": (
            "产品参数查询后端尚未接入。可改用 rag_search 在政策知识库中查询利率范围、期限范围等政策类信息，"
            "或用 ask_user 让用户改述为政策类问题。禁止编造具体数值。"
        ),
        "credit.limit.read": (
            "授信额度查询后端尚未接入。请用 answer 明确告知用户该数据查询暂未上线，"
            "并建议用户走线下渠道或人工咨询。禁止编造金额。"
        ),
        "application.draft.create": (
            "申请草稿后端尚未接入。请用 handoff 转人工受理。"
        ),
    }
    return hints.get(
        capability.capability_id,
        f"{tool_name} 暂未接入，请改用 alternative_tools 或选择 answer / ask_user / handoff。",
    )
```

### 6.2 验收

新增 `test/contract/test_tool_registry_observation.py`：

```python
def test_unavailable_tool_returns_observation():
    """工具未注册 → kind=unavailable + alternative_tools 不含自己 + hint 非空。"""

def test_blocked_tool_returns_observation():
    """工具不在白名单 → kind=blocked。"""

def test_schema_error_returns_observation():
    """输入 schema 错 → kind=schema_error。"""

def test_failed_tool_returns_observation():
    """工具抛异常（如 RAG 后端 500）→ kind=failed。"""

def test_empty_result_returns_observation():
    """工具成功但 NOT_FOUND → kind=empty + sources_count=0。"""

def test_success_result_returns_observation():
    """工具成功 + 有数据 → kind=success + summary ≤ 500 字符。"""

def test_summary_preserves_critical_ids():
    """summary 不会截断 application_id / authorization_id。"""

def test_unavailable_hint_per_capability():
    """product.terms.read / credit.limit.read 都有专属 hint。"""
```

---

## 7. Phase 5：L0 + L1 入口短路

**目标**：把"特定输入直接走对应操作"做实，0 模型调用解决高频确定性输入。

### 7.1 新增 `src/xinyidai_agent/router/rules_guard.py`

```python
"""L0 守卫：纯规则前置过滤，命中立刻返回，不走任何模型调用。

设计准则：
- 宁可漏，不可错。
- 规则只命中"非业务请求"或"100% 确定的业务请求"。
- 任何边界情况都让 L1 / L2 接管。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from xinyidai_agent.protocol import ChatRequest


GuardKind = Literal["pass", "reject_empty", "reject_invalid", "continuation_confirm"]


@dataclass(frozen=True)
class GuardResult:
    kind: GuardKind
    user_visible_message: str = ""
    reason: str = ""


class RulesGuard:
    """L0 规则守卫。"""

    _CONFIRM_REPLIES = frozenset({"确认", "好", "好的", "ok", "yes", "y", "是", "对", "同意"})
    _CANCEL_REPLIES = frozenset({"取消", "不", "否", "no", "n", "不要", "不同意"})

    def guard(self, request: ChatRequest, *, has_pending_action: bool) -> GuardResult:
        """守卫入口。

        :param request: 用户请求
        :param has_pending_action: 当前 session 是否存在待确认动作
        """
        msg = (request.user_message or "").strip()
        if not msg:
            return GuardResult(
                kind="reject_empty",
                user_visible_message="请发送您的问题或诉求，我会为您解答。",
                reason="empty_message",
            )
        if not self._has_meaningful_content(msg):
            return GuardResult(
                kind="reject_invalid",
                user_visible_message="您的问题我没看明白，能详细描述一下吗？",
                reason="no_chinese_or_letter",
            )
        if has_pending_action and self._is_confirmation_reply(msg):
            return GuardResult(
                kind="continuation_confirm",
                reason="user_confirmed_pending_action",
            )
        return GuardResult(kind="pass")

    @staticmethod
    def _has_meaningful_content(msg: str) -> bool:
        """判断是否包含中文字符或字母数字。"""
        for ch in msg:
            if "\u4e00" <= ch <= "\u9fff" or ch.isalnum():
                return True
        return False

    @classmethod
    def _is_confirmation_reply(cls, msg: str) -> bool:
        """判断是否是肯定确认。"""
        return msg.strip().lower() in cls._CONFIRM_REPLIES
```

### 7.2 新增 `src/xinyidai_agent/router/scene_direct.py`

```python
"""L1 场景直达：高确定性的输入命中预设 handler，跳过 L2 模型路由。

handler 必须：
- 0 模型调用
- 输出 final_answer 直接返回
- 不依赖 session_state
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from xinyidai_agent.capabilities.catalog import CapabilityCatalog
from xinyidai_agent.protocol import ChatRequest


@dataclass(frozen=True)
class SceneDirectHit:
    handler_name: str
    final_answer: str
    stop_reason: str = "answered_without_tool"


class SceneDirectDispatcher:
    """L1 直达分发器。"""

    def __init__(self, catalog: CapabilityCatalog) -> None:
        self._catalog = catalog
        self._rules: list[tuple[re.Pattern[str], Callable[[ChatRequest], SceneDirectHit]]] = [
            (re.compile(r"^(你是(谁|什么)|介绍.*你|你能(做|干).*什么)\s*[？?。.！!]*$"),
             self._handle_introduction),
            (re.compile(r"(支持|能办|有什么|有哪些).*(业务|功能|服务|能力)"),
             self._handle_capability_list),
            (re.compile(r"^(ping|测试|在吗|你好|您好|hi|hello|嗨)[\s！!？?。.~]*$", re.I),
             self._handle_greeting),
            (re.compile(r"^(谢谢|多谢|thanks?|thx|辛苦了|麻烦了|感谢)[\s！!。.~]*$", re.I),
             self._handle_thanks),
        ]

    def dispatch(self, request: ChatRequest) -> SceneDirectHit | None:
        """命中即返回 SceneDirectHit；未命中返回 None。"""
        msg = (request.user_message or "").strip()
        if not msg:
            return None
        for pattern, handler in self._rules:
            if pattern.search(msg):
                return handler(request)
        return None

    def _handle_introduction(self, request: ChatRequest) -> SceneDirectHit:
        cap_lines = self._render_capability_brief()
        answer = (
            "我是信易贷智能助手，可以帮您：\n"
            f"{cap_lines}\n"
            "您可以直接告诉我业务诉求，例如：信易贷的准入条件是什么？"
        )
        return SceneDirectHit(handler_name="introduction", final_answer=answer)

    def _handle_capability_list(self, request: ChatRequest) -> SceneDirectHit:
        return SceneDirectHit(
            handler_name="capability_list",
            final_answer="我目前支持以下业务能力：\n" + self._render_capability_brief(),
        )

    def _handle_greeting(self, request: ChatRequest) -> SceneDirectHit:
        return SceneDirectHit(
            handler_name="greeting",
            final_answer="您好，我是信易贷智能助手。请问有什么可以帮您？",
        )

    def _handle_thanks(self, request: ChatRequest) -> SceneDirectHit:
        return SceneDirectHit(
            handler_name="thanks",
            final_answer="不客气，有任何信易贷相关的问题都可以问我。",
        )

    def _render_capability_brief(self) -> str:
        """从 catalog 动态生成能力清单文本。"""
        lines: list[str] = []
        for cap in self._catalog.user_visible_capabilities():
            lines.append(f"- {cap.description}")
        return "\n".join(lines)
```

> **注意**：`CapabilityCatalog` 需要新增 `user_visible_capabilities()` 方法，返回除 `smalltalk.respond` / `unknown.clarify` 外的能力。

### 7.3 修改 `src/xinyidai_agent/router/service.py`

`ControlledIntentRouter` 新增 `pre_route` 方法，并在构造函数注入 guard / dispatcher：

```python
from pydantic import BaseModel, ConfigDict
from typing import Literal

from xinyidai_agent.router.rules_guard import RulesGuard
from xinyidai_agent.router.scene_direct import SceneDirectDispatcher


class PreRouteOutcome(BaseModel):
    """L0 + L1 短路的输出，由 runtime 直接转发或继续走 L2。"""
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["guard_reject", "scene_direct", "continue"]
    final_answer: str | None = None
    stop_reason: str | None = None
    handler_name: str | None = None


class ControlledIntentRouter:
    def __init__(
        self,
        model: ChatModel | None = None,
        rules: RuleBasedRouter | None = None,
        policy: RoutePolicy | None = None,
        guard: RulesGuard | None = None,
        scene_direct: SceneDirectDispatcher | None = None,
    ) -> None:
        self._rules = rules or RuleBasedRouter()
        self._model_router = ModelIntentRouter(model) if model else None
        self._policy = policy or RoutePolicy()
        self._guard = guard or RulesGuard()
        self._scene_direct = scene_direct  # 必须由调用方传入（依赖 catalog）

    def pre_route(
        self,
        request: ChatRequest,
        *,
        has_pending_action: bool,
    ) -> PreRouteOutcome:
        """L0 + L1 前置短路。"""
        guard_result = self._guard.guard(request, has_pending_action=has_pending_action)
        if guard_result.kind in {"reject_empty", "reject_invalid"}:
            return PreRouteOutcome(
                kind="guard_reject",
                final_answer=guard_result.user_visible_message,
                stop_reason="rules_guard_rejected",
                handler_name=guard_result.reason,
            )
        if guard_result.kind == "continuation_confirm":
            # 让 runtime 知道这是确认回复，但仍继续走 L2/L3 流程
            return PreRouteOutcome(kind="continue", handler_name="continuation_confirm")
        if self._scene_direct is not None:
            hit = self._scene_direct.dispatch(request)
            if hit is not None:
                return PreRouteOutcome(
                    kind="scene_direct",
                    final_answer=hit.final_answer,
                    stop_reason=hit.stop_reason,
                    handler_name=hit.handler_name,
                )
        return PreRouteOutcome(kind="continue")

    def route(self, request: ChatRequest) -> RouteDecision:
        # 保持原有逻辑不变，runtime 在 pre_route 决定不短路时才调用
        ...
```

### 7.4 修改 `src/xinyidai_agent/router/__init__.py`

导出 `RulesGuard` / `SceneDirectDispatcher` / `PreRouteOutcome`。

### 7.5 验收

新增 `test/contract/test_router_pre_route.py`：

```python
def test_empty_message_rejected_by_guard(): ...
def test_pure_punctuation_rejected_by_guard(): ...
def test_confirmation_reply_passes_to_continue(): ...
def test_introduction_hits_scene_direct(): ...
def test_capability_list_hits_scene_direct(): ...
def test_greeting_hits_scene_direct(): ...
def test_thanks_hits_scene_direct(): ...
def test_business_question_falls_through_to_continue(): ...
def test_pre_route_does_not_call_model():
    """断言 pre_route 调用过程中 model.complete 调用次数为 0。"""
```

9 个用例全部通过。

---

## 8. Phase 6：L3-fast 单工具快速通道

**目标**：`knowledge.policy.read` 和 `application.status.read` 跳过 ReAct 多轮，1 路由 + 1 工具 + 1 答案。

### 8.1 新增 `src/xinyidai_agent/runtime/__init__.py`

```python
"""ReAct 主循环及其支撑组件。"""
```

### 8.2 新增 `src/xinyidai_agent/runtime/fast_path.py`

```python
"""单工具快速通道：fast_path_eligible 的 capability 跳过 ReAct 多轮。

适用条件（CapabilityPolicy.is_fast_path_eligible_for_route）：
- capability.fast_path_eligible == True
- 必填 slots 全部填齐
- allowed_tools 长度 == 1
- risk_level == "read_only"

失败回退：observation.kind != "success" → 升级到 ReAct 多轮。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.evidence_policy import EvidencePolicy
from xinyidai_agent.llm import ChatModel
from xinyidai_agent.protocol import (
    ChatRequest,
    RouteDecision,
    SessionStateSnapshot,
    SourceDocument,
    ToolCall,
    ToolExecutionObservation,
    ToolResult,
)
from xinyidai_agent.tools.base import ToolExecution
from xinyidai_agent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class FastPathOutcome:
    """fast_path 执行结果。"""
    succeeded: bool
    final_answer: str | None
    sources: list[SourceDocument]
    tool_result: ToolResult | None
    tool_execution: ToolExecution | None
    observation: ToolExecutionObservation | None
    fallback_reason: str | None = None


class FastPathRunner:
    """单工具快速通道执行器。"""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        model: ChatModel,
        evidence_policy: EvidencePolicy | None = None,
    ) -> None:
        self._tools = tool_registry
        self._model = model
        self._evidence = evidence_policy or EvidencePolicy()

    def run(
        self,
        request: ChatRequest,
        route: RouteDecision,
        capability: CapabilityPolicy,
        session_state: SessionStateSnapshot,
    ) -> FastPathOutcome:
        """执行快速通道。

        失败时返回 succeeded=False，调用方需升级到 ReAct 多轮。
        """
        if not capability.is_fast_path_eligible_for_route(route.filled_slots):
            return FastPathOutcome(
                succeeded=False,
                final_answer=None,
                sources=[],
                tool_result=None,
                tool_execution=None,
                observation=None,
                fallback_reason="not_fast_path_eligible",
            )

        tool_name = capability.allowed_tools[0]
        tool_call = ToolCall(
            tool_call_id=str(uuid.uuid4()),
            tool_name=tool_name,
            tool_category=route.allowed_tool_categories[0] if route.allowed_tool_categories else None,
            arguments=self._build_arguments(capability, route, request),
            risk_level=capability.risk_level,
            confirmation_required=False,
            reason="fast_path",
        )

        execution, observation = self._tools.execute_observed(
            request, route, tool_call,
            capability=capability,
            session_state=session_state,
        )

        if observation.kind != "success":
            return FastPathOutcome(
                succeeded=False,
                final_answer=None,
                sources=list(execution.sources),
                tool_result=execution.result,
                tool_execution=execution,
                observation=observation,
                fallback_reason=f"observation_kind_{observation.kind}",
            )

        answer = self._compose_answer(
            request, route, capability, execution.result, execution.sources,
        )
        return FastPathOutcome(
            succeeded=True,
            final_answer=answer,
            sources=list(execution.sources),
            tool_result=execution.result,
            tool_execution=execution,
            observation=observation,
        )

    def _build_arguments(
        self,
        capability: CapabilityPolicy,
        route: RouteDecision,
        request: ChatRequest,
    ) -> dict:
        """根据 capability 构造工具参数（每个 capability 显式映射）。"""
        if capability.capability_id == "knowledge.policy.read":
            return {"query": request.user_message, "top_k": request.top_k or 5}
        if capability.capability_id == "application.status.read":
            return {"company_name": route.filled_slots.get("company_name", "")}
        raise ValueError(f"未知 fast_path capability: {capability.capability_id}")

    def _compose_answer(
        self,
        request: ChatRequest,
        route: RouteDecision,
        capability: CapabilityPolicy,
        result: ToolResult,
        sources: list[SourceDocument],
    ) -> str:
        """基于 sources 生成最终回答；KNOWLEDGE_QA 强制带证据引用。"""
        # 必须复用 runtime.py 现有的 _summarize_tool_result + EvidencePolicy.attach_citations
        # 不允许重写一份，避免格式漂移。
        ...
```

### 8.3 验收

新增 `test/contract/test_fast_path.py`：

```python
def test_knowledge_qa_fast_path_succeeds_with_sources(): ...
def test_knowledge_qa_fast_path_fallback_on_empty(): ...
def test_application_status_fast_path_requires_company_name(): ...
def test_data_query_capability_not_eligible(): ...
def test_fast_path_argument_mapping_per_capability(): ...
def test_fast_path_does_not_call_model_for_arguments():
    """fast_path 构造参数时不调用 LLM。"""
```

---

## 9. Phase 7：ReactStepEngine + Prompt + ObservationBuilder

**目标**：实现 ReAct 主循环的"模型 reasoning"和"guardrails"两层。

### 9.1 新增 `src/xinyidai_agent/runtime/react_observation_builder.py`

```python
"""根据 runtime 当前状态构造 ReactObservation 喂给模型。"""

from __future__ import annotations

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.protocol import (
    ChatRequest,
    RouteDecision,
    SessionStateSnapshot,
    ToolExecutionObservation,
)
from xinyidai_agent.react_observation import ReactObservation, ToolSpecBrief
from xinyidai_agent.tools.registry import ToolRegistry


class ReactObservationBuilder:
    """ReactObservation 构造器。"""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tools = tool_registry

    def build(
        self,
        *,
        step: int,
        max_steps: int,
        request: ChatRequest,
        route: RouteDecision,
        capability: CapabilityPolicy,
        session_state: SessionStateSnapshot,
        executed_tools: list[ToolExecutionObservation],
        last_observation: ToolExecutionObservation | None,
        invalid_action_hint: str | None = None,
    ) -> ReactObservation:
        briefs = self._render_tool_briefs(capability, executed_tools)
        return ReactObservation(
            step=step,
            max_steps=max_steps,
            request=request,
            route=route,
            capability=capability,
            session_state=session_state,
            available_tools=briefs,
            executed_tools=executed_tools,
            last_observation=last_observation,
            invalid_action_hint=invalid_action_hint,
        )

    def _render_tool_briefs(
        self,
        capability: CapabilityPolicy,
        executed_tools: list[ToolExecutionObservation],
    ) -> list[ToolSpecBrief]:
        """暴露 capability.allowed_tools 中的工具。

        未注册的工具仍要在列表里露名（标 already_executed_count=0），
        让模型知道理论上有这个工具但当前不可调。
        """
        briefs: list[ToolSpecBrief] = []
        for tool_name in capability.allowed_tools:
            spec = self._tools.spec_for_name(tool_name)
            executed = sum(1 for e in executed_tools if e.tool_name == tool_name)
            if spec is None:
                briefs.append(ToolSpecBrief(
                    name=tool_name,
                    description=f"（{tool_name} 当前未在工具注册表接入，调用会返回 unavailable）",
                    risk_level="read_only",
                    is_read_only=True,
                    input_schema={},
                    required_arguments=[],
                    already_executed_count=executed,
                ))
                continue
            briefs.append(ToolSpecBrief(
                name=spec.name,
                description=spec.description,
                risk_level=spec.risk_level,
                is_read_only=spec.is_read_only,
                input_schema=dict(spec.input_schema),
                required_arguments=spec.required_argument_names(),
                already_executed_count=executed,
            ))
        return briefs
```

### 9.2 新增 `src/xinyidai_agent/runtime/react_prompt.py`

```python
"""构造 ReAct 单步的 system + user prompt。

强约束：
- system prompt 注入完整 ReactAction JSON Schema
- user prompt 注入 capability 描述、可用工具、历史执行、最近 observation
- response_format 强制 JSON 输出
"""

from __future__ import annotations

import json
from typing import Any

from xinyidai_agent.protocol import ToolExecutionObservation
from xinyidai_agent.react_observation import ReactObservation


REACT_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {"type": "string", "maxLength": 500},
        "action": {
            "type": "object",
            "properties": {
                "type": {"enum": ["call_tool", "answer", "ask_user", "handoff"]},
                "tool_name": {"type": ["string", "null"]},
                "arguments": {"type": "object"},
                "final_answer": {"type": ["string", "null"]},
                "evidence_used": {"type": "array", "items": {"type": "string"}},
                "message": {"type": ["string", "null"]},
            },
            "required": ["type"],
            "additionalProperties": False,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["thought", "action", "confidence"],
    "additionalProperties": False,
}


class ReactPromptBuilder:
    """ReAct 单步 prompt 构造器。"""

    def build_messages(self, observation: ReactObservation) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._build_system_prompt(observation)},
            {"role": "user", "content": self._build_user_prompt(observation)},
        ]

    def _build_system_prompt(self, obs: ReactObservation) -> str:
        return (
            "# 你的角色\n"
            "你是信易贷业务智能助手的 ReAct 推理器。每一轮你只能输出一个 action，"
            "且 action.type 必须是 call_tool / answer / ask_user / handoff 之一。\n\n"
            "# 严格约束\n"
            f"- 当前能力 {obs.capability.capability_id}：{obs.capability.description}\n"
            f"- 当前已执行 {len(obs.executed_tools)} 次工具，最大允许 {obs.max_steps} 步。\n"
            "- 仅能调用 available_tools 中列出的工具。\n"
            "- 同一 (tool_name, arguments) 已经执行过 2 次以上时，禁止再次调用，"
            "请改用 answer / ask_user / handoff。\n"
            f"- 当前能力 requires_evidence={obs.capability.requires_evidence}："
            "为 True 时 answer 必须基于工具结果且填 evidence_used。\n"
            "- final_answer 必须基于已成功的工具结果或闲聊场景，"
            "禁止编造数据、链接、利率、额度。\n"
            "- 当所需工具未注册（last_observation.kind=='unavailable'）时：\n"
            "  - 优先选择 alternative_tools 中的工具\n"
            "  - 否则用 answer 明确告知用户该数据未上线\n"
            "  - 或用 ask_user 让用户改述\n"
            "- 风险动作（state_create / state_update / final_submit）由系统转 pending_action 确认，"
            "你只需正常输出 call_tool。\n\n"
            "# 输出格式\n"
            "只输出 JSON 对象，结构必须严格符合：\n"
            f"{json.dumps(REACT_ACTION_SCHEMA, ensure_ascii=False, indent=2)}\n"
            "禁止输出任何 JSON 之外的文字、注释、Markdown 包裹。\n"
        )

    def _build_user_prompt(self, obs: ReactObservation) -> str:
        parts: list[str] = []
        parts.append(f"# 用户问题\n{obs.request.user_message}\n")
        parts.append(f"# 当前路由\nscene={obs.route.scene}, "
                     f"capability={obs.capability.capability_id}, "
                     f"confidence={obs.route.confidence:.2f}\n")
        parts.append(f"# 已填槽位\n{json.dumps(dict(obs.route.filled_slots), ensure_ascii=False)}\n")
        if obs.route.missing_slots:
            parts.append(f"# 缺失槽位\n{obs.route.missing_slots}\n")
        parts.append("# 可用工具\n" + self._render_tools(obs))
        if obs.executed_tools:
            parts.append("# 已执行工具历史\n" + self._render_executed(obs.executed_tools))
        if obs.last_observation is not None:
            parts.append("# 你上一步的执行观察\n" + self._render_observation(obs.last_observation))
        if obs.invalid_action_hint:
            parts.append(f"# 上次输出被规则拒绝\n{obs.invalid_action_hint}\n请重新选择 action。")
        parts.append(f"# 当前步数\nstep={obs.step}/{obs.max_steps}")
        return "\n".join(parts)

    @staticmethod
    def _render_tools(obs: ReactObservation) -> str:
        lines = []
        for t in obs.available_tools:
            lines.append(
                f"- `{t.name}`（risk={t.risk_level}, read_only={t.is_read_only}, "
                f"已执行={t.already_executed_count} 次）：{t.description}"
                f"\n  required_arguments: {t.required_arguments}"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_executed(executed: list[ToolExecutionObservation]) -> str:
        lines = []
        for i, e in enumerate(executed, 1):
            arg_brief = json.dumps(dict(e.arguments), ensure_ascii=False)[:120]
            detail = e.summary or e.reason or ""
            lines.append(f"{i}. [{e.kind}] {e.tool_name}({arg_brief}) → {detail[:200]}")
        return "\n".join(lines)

    @staticmethod
    def _render_observation(obs: ToolExecutionObservation) -> str:
        return (
            f"kind={obs.kind}\n"
            f"tool={obs.tool_name}\n"
            f"summary={obs.summary}\n"
            f"reason={obs.reason}\n"
            f"hint={obs.hint}\n"
            f"alternative_tools={obs.alternative_tools}\n"
        )
```

### 9.3 新增 `src/xinyidai_agent/runtime/react_engine.py`

```python
"""ReAct 单步执行器：模型 reasoning + 规则 guardrails + 工具执行。

不是循环，只是单步。循环由 ReactRuntime 控制。
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Literal

from xinyidai_agent.llm import JSON_OBJECT_RESPONSE_FORMAT, ChatModel
from xinyidai_agent.policies import RiskPolicy
from xinyidai_agent.protocol import (
    ReactAction,
    ReactStepDecision,
    ToolCall,
    ToolExecutionObservation,
)
from xinyidai_agent.react_observation import ReactObservation
from xinyidai_agent.runtime.react_prompt import ReactPromptBuilder
from xinyidai_agent.tools.base import ToolExecution, ToolSpec
from xinyidai_agent.tools.registry import ToolRegistry


StepKind = Literal["call_tool_executed", "terminal", "invalid_action"]


@dataclass(frozen=True)
class StepOutcome:
    """ReactStepEngine 单步执行结果。"""
    kind: StepKind
    decision: ReactStepDecision
    tool_observation: ToolExecutionObservation | None = None
    tool_execution: ToolExecution | None = None
    invalid_hint: str | None = None
    pending_action_required: bool = False
    pending_tool_call: ToolCall | None = None


class ReactStepEngine:
    """单步 ReAct 执行：reason → guardrail → execute。"""

    _REPEAT_LIMIT = 2  # 同 (name, args) 已执行 ≥ 此数则硬阻断

    def __init__(
        self,
        model: ChatModel,
        tool_registry: ToolRegistry,
        prompt_builder: ReactPromptBuilder | None = None,
        risk_policy: RiskPolicy | None = None,
        max_json_retries: int = 1,
    ) -> None:
        self._model = model
        self._tools = tool_registry
        self._prompt = prompt_builder or ReactPromptBuilder()
        self._risk = risk_policy or RiskPolicy()
        self._max_json_retries = max_json_retries

    def step(self, observation: ReactObservation) -> StepOutcome:
        """执行单步。"""
        decision = self._reason(observation)
        action = decision.action

        if action.type == "call_tool":
            return self._handle_call_tool(observation, decision)
        if action.type == "answer":
            return self._validate_answer(observation, decision)
        if action.type == "ask_user":
            return self._validate_ask_user(decision)
        if action.type == "handoff":
            return self._validate_handoff(decision)
        return StepOutcome(
            kind="invalid_action",
            decision=decision,
            invalid_hint="action.type 必须是 call_tool / answer / ask_user / handoff。",
        )

    # ---------- 模型 reasoning（含 1 次 JSON 修复重试） ----------
    def _reason(self, observation: ReactObservation) -> ReactStepDecision:
        messages = self._prompt.build_messages(observation)
        last_error = ""
        for attempt in range(self._max_json_retries + 1):
            raw = self._model.complete(messages, response_format=JSON_OBJECT_RESPONSE_FORMAT)
            try:
                payload = self._parse_json(raw)
                return ReactStepDecision.model_validate(payload)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                last_error = str(exc)
                if attempt < self._max_json_retries:
                    messages = self._prompt.build_messages(
                        observation.model_copy(update={
                            "invalid_action_hint": (
                                f"上一次输出不是合法 JSON：{last_error}，"
                                "请严格按 schema 输出。"
                            ),
                        })
                    )
        raise RuntimeError(f"模型 JSON 输出无法解析：{last_error}")

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"未找到 JSON 对象：{raw[:200]}")
        return json.loads(text[start:end + 1])

    # ---------- guardrails ----------
    def _handle_call_tool(self, obs: ReactObservation, decision: ReactStepDecision) -> StepOutcome:
        action = decision.action
        if not action.tool_name:
            return self._reject(decision, "选择 call_tool 时必须给出 tool_name。")
        if action.tool_name not in obs.capability.allowed_tools:
            return self._reject(decision, (
                f"{action.tool_name} 不在当前能力 {obs.capability.capability_id} 的工具白名单："
                f"{obs.capability.allowed_tools}"
            ))
        repeat = self._count_same_calls(obs.executed_tools, action)
        if repeat >= self._REPEAT_LIMIT:
            return self._reject(decision, (
                f"同 (tool_name, arguments) 已执行 {repeat} 次，"
                "请改用 answer / ask_user / handoff。"
            ))

        spec = self._tools.spec_for_name(action.tool_name)
        # 工具未注册：仍走 execute_observed，让 registry 产出 unavailable observation
        if spec is None:
            tool_call = self._build_tool_call(action, None, obs)
            execution, tool_observation = self._tools.execute_observed(
                obs.request, obs.route, tool_call,
                capability=obs.capability,
                session_state=obs.session_state,
            )
            return StepOutcome(
                kind="call_tool_executed",
                decision=decision,
                tool_observation=tool_observation,
                tool_execution=execution,
            )

        # 高风险（state_create / state_update / final_submit）→ 转 pending_action
        if self._risk.requires_handoff(spec.risk_level):
            return self._reject(decision, (
                f"{action.tool_name} 风险级别 {spec.risk_level} 必须人工处理，请改用 handoff。"
            ))
        if self._risk.requires_confirmation(spec.risk_level, spec.requires_confirmation):
            tool_call = self._build_tool_call(action, spec, obs)
            return StepOutcome(
                kind="terminal",
                decision=decision,
                pending_action_required=True,
                pending_tool_call=tool_call,
            )

        tool_call = self._build_tool_call(action, spec, obs)
        execution, tool_observation = self._tools.execute_observed(
            obs.request, obs.route, tool_call,
            capability=obs.capability,
            session_state=obs.session_state,
        )
        return StepOutcome(
            kind="call_tool_executed",
            decision=decision,
            tool_observation=tool_observation,
            tool_execution=execution,
        )

    def _validate_answer(self, obs: ReactObservation, decision: ReactStepDecision) -> StepOutcome:
        action = decision.action
        if not action.final_answer or not action.final_answer.strip():
            return self._reject(decision, "选择 answer 时 final_answer 不能为空。")
        if obs.capability.requires_evidence:
            success_obs = [
                e for e in obs.executed_tools
                if e.kind == "success" and e.sources_count > 0
            ]
            if not success_obs:
                return self._reject(decision, (
                    "当前能力要求 answer 必须基于工具检索到的证据，"
                    "请先 call_tool 拿到 sources。"
                ))
            if not action.evidence_used:
                return self._reject(decision, "answer 必须填 evidence_used，标明引用了哪些 source_id。")
        return StepOutcome(kind="terminal", decision=decision)

    def _validate_ask_user(self, decision: ReactStepDecision) -> StepOutcome:
        if not decision.action.message:
            return self._reject(decision, "选择 ask_user 时 message 不能为空。")
        return StepOutcome(kind="terminal", decision=decision)

    def _validate_handoff(self, decision: ReactStepDecision) -> StepOutcome:
        if not decision.action.message:
            return self._reject(decision, "选择 handoff 时 message 不能为空。")
        return StepOutcome(kind="terminal", decision=decision)

    # ---------- helpers ----------
    @staticmethod
    def _reject(decision: ReactStepDecision, hint: str) -> StepOutcome:
        return StepOutcome(kind="invalid_action", decision=decision, invalid_hint=hint)

    @staticmethod
    def _count_same_calls(
        executed: list[ToolExecutionObservation],
        action: ReactAction,
    ) -> int:
        return sum(
            1 for e in executed
            if e.tool_name == action.tool_name and e.arguments == action.arguments
        )

    def _build_tool_call(
        self,
        action: ReactAction,
        spec: ToolSpec | None,
        obs: ReactObservation,
    ) -> ToolCall:
        category = spec.category if spec is not None else None
        risk = spec.risk_level if spec is not None else "read_only"
        confirmation = spec.requires_confirmation if spec is not None else False
        return ToolCall(
            tool_call_id=str(uuid.uuid4()),
            tool_name=action.tool_name or "",
            tool_category=category,
            arguments=dict(action.arguments),
            risk_level=risk,
            confirmation_required=confirmation,
            reason=f"react_step_{obs.step}",
        )
```

### 9.4 验收

新增 `test/contract/test_react_step_engine.py`：

```python
def test_call_tool_executed_when_valid(): ...
def test_call_tool_unavailable_returns_call_tool_executed_with_obs():
    """工具未注册仍返回 call_tool_executed + observation.kind=unavailable。"""
def test_call_tool_blocked_when_not_in_whitelist(): ...
def test_call_tool_blocked_when_repeat_limit_reached(): ...
def test_answer_requires_evidence_when_capability_requires_it(): ...
def test_answer_requires_non_empty_final_answer(): ...
def test_answer_requires_evidence_used_when_required(): ...
def test_ask_user_requires_non_empty_message(): ...
def test_handoff_requires_non_empty_message(): ...
def test_invalid_json_retried_once_then_raises(): ...
def test_call_tool_state_create_sets_pending_action_required(): ...
def test_call_tool_final_submit_rejected_to_handoff(): ...
```

---

## 10. Phase 8：主循环切换

**目标**：用 `ReactRuntime` 替换 `ControlledAgentLoop` 的核心，**整体删除** 旧的 `RuntimeStepPolicy` / `_plan_tool` / `_answer_without_tool`。

### 10.1 新增 `src/xinyidai_agent/runtime/react_loop.py`

```python
"""ReAct 主循环。

入口：ReactRuntime.run(request) -> Iterator[AgentEvent]

循环不变量：
- invalid_action 兜底重试上限 = 2，超过转 runtime_error，不消耗 step
- call_tool_executed 算消耗 1 step
- terminal 立即结束循环
- step > max_react_steps 时强制构造 ReactTerminal(reason="max_steps")，
  final_answer 是 ask_user 兜底文案（不允许 LLM 自由生成）
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.capabilities.catalog import CapabilityCatalog
from xinyidai_agent.evidence_policy import EvidencePolicy
from xinyidai_agent.llm import ChatModel
from xinyidai_agent.memory import MemoryManager
from xinyidai_agent.pending_actions import PendingActionFactory
from xinyidai_agent.protocol import (
    AgentEvent,
    ChatRequest,
    PendingAction,
    ReactTerminal,
    SessionStateSnapshot,
    SourceDocument,
    ToolCall,
    ToolExecutionObservation,
)
from xinyidai_agent.router.service import ControlledIntentRouter
from xinyidai_agent.runtime.fast_path import FastPathRunner
from xinyidai_agent.runtime.react_engine import ReactStepEngine, StepOutcome
from xinyidai_agent.runtime.react_observation_builder import ReactObservationBuilder
from xinyidai_agent.tools.base import ToolExecution
from xinyidai_agent.tools.registry import ToolRegistry


class ReactRuntime:
    """ReAct 主循环。"""

    _MAX_INVALID_ACTION_RETRIES = 2

    def __init__(
        self,
        model: ChatModel,
        router: ControlledIntentRouter,
        tool_registry: ToolRegistry,
        catalog: CapabilityCatalog,
        memory: MemoryManager,
        fast_path: FastPathRunner,
        step_engine: ReactStepEngine,
        observation_builder: ReactObservationBuilder,
        evidence_policy: EvidencePolicy,
        pending_action_factory: PendingActionFactory,
    ) -> None:
        self._model = model
        self._router = router
        self._tools = tool_registry
        self._catalog = catalog
        self._memory = memory
        self._fast_path = fast_path
        self._step_engine = step_engine
        self._observation_builder = observation_builder
        self._evidence = evidence_policy
        self._pending_factory = pending_action_factory

    def run(self, request: ChatRequest) -> Iterator[AgentEvent]:
        """主循环入口。"""
        # 关键路径骨架（详细事件发送由 EventEmitter 上下文管理）：
        # 1. 加载 session_state
        # 2. router.pre_route()  →  L0/L1 短路
        # 3. router.route()      →  L2 路由
        # 4. 找 capability，未找到则 ask_user
        # 5. fast_path 尝试      →  L3-fast
        # 6. ReAct 主循环        →  L3-react
        # 7. 落库 + emit final
        ...

    def _react_loop(
        self,
        request: ChatRequest,
        route,
        capability: CapabilityPolicy,
        session_state: SessionStateSnapshot,
    ) -> tuple[ReactTerminal, list[ToolExecution]]:
        """ReAct 主循环本体。

        返回 (terminal, all_executions)，executions 用于落库 tool_trace。
        """
        executed_observations: list[ToolExecutionObservation] = []
        all_executions: list[ToolExecution] = []
        last_obs: ToolExecutionObservation | None = None
        invalid_hint: str | None = None
        invalid_retries = 0
        step = 0

        while step < capability.max_react_steps:
            observation = self._observation_builder.build(
                step=step + 1,
                max_steps=capability.max_react_steps,
                request=request,
                route=route,
                capability=capability,
                session_state=session_state,
                executed_tools=executed_observations,
                last_observation=last_obs,
                invalid_action_hint=invalid_hint,
            )

            try:
                outcome = self._step_engine.step(observation)
            except RuntimeError as exc:
                return self._build_runtime_error_terminal(exc), all_executions

            if outcome.kind == "invalid_action":
                invalid_retries += 1
                if invalid_retries > self._MAX_INVALID_ACTION_RETRIES:
                    return (
                        self._build_runtime_error_terminal(
                            RuntimeError(f"模型连续 {invalid_retries} 次输出非法 action：{outcome.invalid_hint}")
                        ),
                        all_executions,
                    )
                invalid_hint = outcome.invalid_hint
                # 不消耗 step
                continue

            # 走到这里说明本步合法
            invalid_hint = None
            invalid_retries = 0

            if outcome.kind == "call_tool_executed":
                executed_observations.append(outcome.tool_observation)  # type: ignore[arg-type]
                all_executions.append(outcome.tool_execution)  # type: ignore[arg-type]
                last_obs = outcome.tool_observation
                step += 1
                continue

            # outcome.kind == "terminal"
            return self._build_terminal(
                outcome=outcome,
                executed_observations=executed_observations,
                all_executions=all_executions,
                capability=capability,
                session_state=session_state,
            ), all_executions

        # 达到 max_steps
        return self._build_max_steps_terminal(executed_observations, capability), all_executions

    def _build_terminal(
        self,
        *,
        outcome: StepOutcome,
        executed_observations: list[ToolExecutionObservation],
        all_executions: list[ToolExecution],
        capability: CapabilityPolicy,
        session_state: SessionStateSnapshot,
    ) -> ReactTerminal:
        """根据 StepOutcome 构造终止决策。"""
        action = outcome.decision.action
        sources = self._collect_sources_from_executions(all_executions, action.evidence_used)

        if outcome.pending_action_required:
            assert outcome.pending_tool_call is not None
            pending = self._pending_factory.build(
                tool_call=outcome.pending_tool_call,
                session_state=session_state,
                capability=capability,
            )
            return ReactTerminal(
                reason="wait_confirmation",
                final_answer=pending.summary,
                pending_action=pending,
            )

        if action.type == "answer":
            answer = self._evidence.attach_citations(action.final_answer or "", sources) \
                if capability.requires_evidence else (action.final_answer or "")
            return ReactTerminal(
                reason="completed" if all_executions else "answered_no_tool",
                final_answer=answer,
                sources=sources,
            )
        if action.type == "ask_user":
            return ReactTerminal(
                reason="ask_user",
                final_answer=action.message or "请补充更多信息以便继续。",
                missing_slots=list(self._guess_missing_slots(executed_observations, capability)),
            )
        if action.type == "handoff":
            return ReactTerminal(
                reason="handoff",
                final_answer=action.message or "需要人工接入处理。",
                handoff_reason=action.message,
            )
        # 不应到达
        raise RuntimeError(f"未知 action.type：{action.type}")

    def _build_max_steps_terminal(
        self,
        executed_observations: list[ToolExecutionObservation],
        capability: CapabilityPolicy,
    ) -> ReactTerminal:
        """max_steps 兜底：强制 ask_user，禁止 LLM 自由生成完整回答。"""
        executed_brief = ", ".join(
            f"{e.tool_name}({e.kind})" for e in executed_observations
        ) or "无"
        return ReactTerminal(
            reason="max_steps",
            final_answer=(
                f"我已经尝试调用了 {len(executed_observations)} 次工具（{executed_brief}），"
                "但还没办法给出完整回答。请您再补充一些关键信息，例如企业全称、产品名称或具体诉求，"
                "我会重新为您处理。"
            ),
        )

    def _build_runtime_error_terminal(self, exc: BaseException) -> ReactTerminal:
        return ReactTerminal(
            reason="runtime_error",
            final_answer="系统暂时无法处理您的请求，请稍后再试或联系人工客服。",
            error_class=type(exc).__name__,
            error_message=str(exc)[:500],
        )

    def _collect_sources_from_executions(
        self,
        all_executions: list[ToolExecution],
        evidence_used: list[str],
    ) -> list[SourceDocument]:
        """从已执行工具的 sources 中收集；evidence_used 为空时返回全量。"""
        all_sources: list[SourceDocument] = []
        for e in all_executions:
            all_sources.extend(e.sources)
        if not evidence_used:
            return all_sources
        keep = {sid for sid in evidence_used}
        return [s for s in all_sources if s.source_id in keep]

    @staticmethod
    def _guess_missing_slots(
        executed_observations: list[ToolExecutionObservation],
        capability: CapabilityPolicy,
    ) -> list[str]:
        """ask_user 时尝试从 capability.required_slots 推断仍缺失的字段。"""
        return list(capability.required_slots)
```

### 10.2 修改 `src/xinyidai_agent/runtime.py`

**步骤 A**：`ControlledAgentLoop` 构造函数注入 `react_runtime: ReactRuntime`。

**步骤 B**：`run` 方法替换为：

```python
def run(self, request: ChatRequest) -> Iterator[AgentEvent]:
    yield from self._react_runtime.run(request)
```

**步骤 C**：删除以下方法（不允许保留为 deprecated）：

- `_plan_tool`
- `_parse_tool_plan`
- `_build_deterministic_tool_call`
- `_answer_without_tool`

保留方法（被 react_loop 内部 _build_terminal / fast_path 复用）：

- `_summarize_tool_result`
- `_build_messages`
- `run_confirmation`（confirm flow）

### 10.3 删除文件 / 类型

- 删除文件 `src/xinyidai_agent/runtime_policy.py`
- 删除 `protocol.py` 中的 `RuntimeStepDecision` 类、`RuntimeStepType` Literal
- 全局删除 `from xinyidai_agent.runtime_policy import ...` 引用

### 10.4 修改 `src/xinyidai_agent/api.py`

`build_agent()` 函数改为构造完整依赖图：

```python
def build_agent() -> ControlledAgentLoop:
    model = build_chat_model()
    catalog = default_capability_catalog()
    tool_registry = default_tool_registry()
    memory = MemoryManager()

    guard = RulesGuard()
    scene_direct = SceneDirectDispatcher(catalog)
    router = ControlledIntentRouter(
        model=model,
        guard=guard,
        scene_direct=scene_direct,
    )

    evidence_policy = EvidencePolicy()
    fast_path = FastPathRunner(tool_registry, model, evidence_policy)
    prompt_builder = ReactPromptBuilder()
    step_engine = ReactStepEngine(model, tool_registry, prompt_builder)
    observation_builder = ReactObservationBuilder(tool_registry)
    pending_factory = PendingActionFactory()

    react_runtime = ReactRuntime(
        model=model,
        router=router,
        tool_registry=tool_registry,
        catalog=catalog,
        memory=memory,
        fast_path=fast_path,
        step_engine=step_engine,
        observation_builder=observation_builder,
        evidence_policy=evidence_policy,
        pending_action_factory=pending_factory,
    )

    return ControlledAgentLoop(
        model=model,
        router=router,
        tool_registry=tool_registry,
        memory_manager=memory,
        react_runtime=react_runtime,
    )
```

### 10.5 验收

- 删除 `test/contract/test_runtime_step_*.py`（全文件）
- 修改 `test/contract/test_*.py` 中检查 `RuntimeStepType` / `RuntimeStepDecision` 的断言
- 全量 pytest 必须 0 failed：

```powershell
pytest test/ -x -v
```

---

## 11. Phase 9：全量测试矩阵

### 11.1 端到端 query 矩阵

新增 `test/e2e/test_react_loop_matrix.py`：

```python
import pytest

from xinyidai_agent.protocol import ChatRequest

# (id, user_message, expected_stop_reason, has_sources, expected_path)
CASES: list[tuple[str, str, str | tuple[str, ...], bool, str]] = [
    ("empty",      "",                    "rules_guard_rejected",    False, "L0"),
    ("punct",      "。。。",                "rules_guard_rejected",    False, "L0"),
    ("intro",      "你是谁",                "answered_without_tool",   False, "L1"),
    ("cap_list",   "你能办什么业务",         "answered_without_tool",   False, "L1"),
    ("greeting",   "你好",                  "answered_without_tool",   False, "L1"),
    ("thanks",     "谢谢",                  "answered_without_tool",   False, "L1"),
    ("policy_qa",  "信易贷的准入条件是什么",  "completed",               True,  "L3-fast"),
    ("risk_tags",  "高风险标签都包括哪些情况", "completed",               True,  "L3-fast"),
    ("credit_amt", "查一下 ABC 公司的授信额度", "completed",              False, "L3-react"),
    ("rate",       "小微税贷的利率是多少",
                   ("answered_without_tool", "missing_slots", "completed"),
                                                                       False, "L3-react"),
    ("auth_link",  "帮我给 ABC 公司生成授权链接",
                   "waiting_confirmation",                              False, "L3-react"),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_react_loop_matrix(case, agent):
    cid, msg, expected, has_sources, _path = case
    resp = agent.answer(ChatRequest(user_message=msg, session_id=f"e2e_{cid}"))

    if isinstance(expected, tuple):
        assert resp.stop_reason in expected, f"{cid}: stop_reason={resp.stop_reason}"
    else:
        assert resp.stop_reason == expected, f"{cid}: stop_reason={resp.stop_reason}"

    if has_sources:
        assert len(resp.sources) >= 1, f"{cid}: 缺少证据"

    # 关键反幻觉断言：rate 用例不允许编造数值
    if cid == "rate":
        answer_lower = resp.answer.lower()
        has_fabricated_number = any(
            kw in resp.answer for kw in ("3.5%", "4.0%", "4.5%", "5.0%", "年化")
        )
        has_disclaimer = any(
            kw in resp.answer for kw in ("未上线", "暂未", "无法获取", "请咨询", "联系")
        )
        if has_fabricated_number:
            assert has_disclaimer, f"rate 用例疑似幻觉数值：{resp.answer}"
```

### 11.2 反幻觉专项测试

新增 `test/e2e/test_no_hallucination.py`：

```python
def test_unavailable_tool_does_not_yield_concrete_numbers():
    """product.terms.read 工具未注册时，回答中不能出现 % / 万元 / 月数等具体数值，
    或必须伴随明确的 disclaimer。"""

def test_credit_amount_without_data_does_not_fabricate():
    """credit.limit.read 拿不到数据时不能编造金额。"""

def test_authorization_link_does_not_yield_fake_url():
    """authorization.link.create 没真正执行时不能在 answer 里给出假 URL。"""
```

### 11.3 性能基线

新增 `test/smoke/test_react_perf_baseline.py`：

```python
def test_l0_l1_under_50ms():
    """L0/L1 路径：p99 < 50ms（0 模型调用）。"""

def test_fast_path_under_6s():
    """L3-fast 路径：p99 < 6s（1 路由 + 1 RAG + 1 答案）。"""

def test_react_step_under_8s():
    """L3-react 单步：p99 < 8s。"""
```

### 11.4 总验收命令

```powershell
conda activate xinyidai-agent

pytest test/ -x -v --tb=short
python scripts/diag_intent_router.py    # 9/9 路由正确
python scripts/smoke_chat_endpoint.py   # 11/11 query 路径正确
```

`progress.txt` 追加：

```
[REACT_REFAC_DONE] <UTC ISO 时间>
pytest: <N> passed, 0 failed
diag_intent_router: 9/9
smoke_chat_endpoint: 11/11
旧文件已删除：runtime_policy.py
旧方法已删除：_plan_tool / _parse_tool_plan / _build_deterministic_tool_call / _answer_without_tool
```

---

## 12. 关键不变量自检清单

每个 Phase 完成后用这张清单核对：

| # | 不变量 | 检查方式 |
|---|---|---|
| 1 | `protocol.py` 不含未使用的 mock 子串（除 `RetrievalTrace.mock` 字段） | `grep -i mock src/xinyidai_agent/protocol.py` |
| 2 | 新增文件不含 `pass` / `TODO` / `FIXME` / 仅 `...` 占位 | `grep -E "^\s*(pass|\.\.\.|TODO\|FIXME)\s*$" src/xinyidai_agent/runtime/*.py` |
| 3 | `RuntimeStepPolicy` 全局零引用 | `grep -r "RuntimeStepPolicy" src/ test/` 应 0 行 |
| 4 | `_answer_without_tool` 全局零引用 | `grep -r "_answer_without_tool" src/` 应 0 行 |
| 5 | `default_tool_registry()` 加载的工具全部 `is_read_only=True`（当前阶段唯二工具都是只读） | 单测断言 |
| 6 | 任何 `requires_evidence=True` 的 capability 都对应 `allowed_tools` 非空 | catalog 加载时自检 |
| 7 | `ReactStepEngine.step` 不抛 ValueError / KeyError / AttributeError | fuzz 测试 |
| 8 | `ReactRuntime.run` 在所有终止路径都发出 `final_answer` + `turn_finished` 事件 | e2e 事件序列断言 |
| 9 | 工具未注册不会出现在 `ChatResponse.answer` 中有具体数值（无 disclaimer） | rate 用例断言 |
| 10 | `max_react_steps` 后 `stop_reason="max_steps"` 且 answer 是 ask_user 兜底文案 | 强制循环超限测试 |
| 11 | `pre_route` 调用不触发 LLM | mock 模型调用计数器断言 |
| 12 | `fast_path._build_arguments` 对未知 capability 抛 ValueError（fail-loudly） | 单测 |

---

## 13. 落地节奏（Commit 记录建议）

按 Phase 顺序执行，每完成一个 Phase 提交一次 commit，commit message 中文：

```
Phase 1：补齐 ReAct 协议（ReactAction/ReactStepDecision/ToolExecutionObservation/ReactTerminal）
Phase 2：CapabilityPolicy 增加 fast_path_eligible / requires_evidence / max_react_steps
Phase 3：ToolSpec fail-closed（is_read_only / is_idempotent / is_concurrency_safe）
Phase 4：ToolRegistry.execute_observed 输出结构化 ToolExecutionObservation
Phase 5：L0 RulesGuard + L1 SceneDirectDispatcher 入口短路
Phase 6：FastPathRunner 单工具快速通道
Phase 7：ReactStepEngine + ReactPromptBuilder + ReactObservationBuilder
Phase 8：主循环切换为 ReactRuntime，删除 RuntimeStepPolicy 与幻觉路径
Phase 9：端到端测试矩阵（11 用例）+ 反幻觉专项 + 性能基线
```

每个 commit 在 commit body 给出该 Phase 的：

- 改动文件清单
- 验收命令输出（passed 数 / failed 数）
- 已删除的旧路径

---

## 14. 附录 A：Phase 间依赖图

```text
Phase 0（基线）
    │
    ▼
Phase 1（协议）
    │
    ├─► Phase 2（Capability）
    │
    └─► Phase 3（ToolSpec）
              │
              ▼
         Phase 4（Registry.execute_observed）   ◄── 依赖 1 + 3
              │
              ├─► Phase 5（L0 + L1）            ◄── 依赖 1
              │
              └─► Phase 6（fast_path）          ◄── 依赖 1 + 2 + 4
                        │
                        ▼
                   Phase 7（ReactEngine）       ◄── 依赖 1 + 2 + 4
                        │
                        ▼
                   Phase 8（主循环切换）         ◄── 依赖 5 + 6 + 7
                        │
                        ▼
                   Phase 9（全量验收）
```

## 15. 附录 B：禁忌清单

Codex 必须遵守，违反任何一条都视为本次改造失败：

- ❌ 不允许新建任何带 `_v2` / `_new` / `_legacy` / `_old` 后缀的文件
- ❌ 不允许保留 `RuntimeStepPolicy` 作为 fallback
- ❌ 不允许在 `react_engine.py` 中调用 `_plan_tool` 等已删除的旧方法
- ❌ 不允许 `ToolExecutionObservation` 字段使用未在 schema 中定义的 `Any` 弱类型
- ❌ 不允许在 prompt 里硬编码"你必须基于 RAG 的内容回答"等模糊指令——必须明确"requires_evidence=True 时 answer 必须填 evidence_used"
- ❌ 不允许 `MockXxx` 命名的新增类（已存在的 `MockCreditAmountTool` 不动）
- ❌ 不允许 `react_engine.py` / `react_loop.py` / `fast_path.py` 内部出现裸 `print` 语句（必须用 `logging`）
- ❌ 不允许通过捕获 `Exception` 静默吞掉异常——必须分类捕获并构造 ToolExecutionObservation 或 ReactTerminal
- ❌ 不允许在 catalog 里使用默认值兜底（必须逐项显式声明 `fast_path_eligible` / `requires_evidence` / `max_react_steps`）

## 16. 附录 C：已知遗留与后续工作

**本次改造不处理**（保留为后续 Phase）：

- 工具并发执行（`is_concurrency_safe` 字段已加但未启用并发调度器）
- 上下文压缩（claude-code 的 reactive_compact / collapse；我们的 turn 较短不急需）
- 流式工具执行（`StreamingToolExecutor` 等价实现）
- MCP 工具支持
- Token budget 续推（`tokenBudget`）
- 工具结果缓存（同 `(tool_name, arguments)` 同 session 内复用上次结果）

这些项进入 `progress.txt` 的 `BACKLOG` 区，不阻塞本次发版。

---

## 修订记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-05-09 | 初版：Phase 0-9 完整清单 |
