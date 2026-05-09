# Agent Runtime 设计：ReAct 主循环 + 分层入口短路

> 版本：v1.0  
> 适用项目：信易贷 Agent + RAG 系统  
> 参考实现：`d:\companyCode\claude-code-cli-master\query.ts`（主循环 1730 行）

---

## 0. 文档目的

本文档分两部分：

1. **第 1～2 章：claude-code 主循环精读**——把 claude-code-cli 的 runtime 设计拆开，提炼出可被业务 Agent 借鉴的 7 条最佳实践。
2. **第 3～11 章：信易贷场景的实现方案**——结合我们的封闭业务域、强受控、强证据、可信回答要求，给出具体的循环重构方案、工具契约、结束规则、失败处理路径和落地步骤。

设计原则贯穿全文：

- **模型主判，规则实操**——模型决定"该做什么"，规则决定"能不能做、怎么做、做完之后怎么走"。
- **失败即观察**——任何工具失败、未注册、参数错误，都不被规则吞掉、不变成幻觉答；统一以结构化 observation 回喂给模型，让模型在下一轮重新决策。
- **fail-closed**——能力、并发、权限默认关闭；新工具上线必须显式声明白名单字段。
- **能直达就不入 ReAct**——可枚举、零歧义的输入走规则直达；多步、需要工具串联或失败回退的才进入 ReAct 多轮。

---

## 1. claude-code 主循环精读

源码位置：`claude-code-cli-master/query.ts:241-1729`，导出 `query()` 单一入口。

### 1.1 唯一推进信号：`needsFollowUp`

```ts
// query.ts:557-558
const toolUseBlocks: ToolUseBlock[] = []
let needsFollowUp = false
// ...
if (msgToolUseBlocks.length > 0) {
  toolUseBlocks.push(...msgToolUseBlocks)
  needsFollowUp = true
}
// ...
if (!needsFollowUp) { return { reason: 'completed' } }   // 1062 行
```

**关键点**：循环只有一个进入下一轮的判定——上一轮 assistant 消息里**是否包含 `tool_use` block**。模型不再请求工具就自然结束，**没有规则去"决定模型该不该停"**。

这就是 ReAct 的标准约束：reasoning 步骤产出 action，action 集合只有两类（call_tool / final_answer），final_answer 即终止。

### 1.2 11 种 Terminal reason

```ts
// query.ts 各 return 站点提取
type Terminal =
  | { reason: 'completed' }                   // 模型不再 call_tool，正常结束
  | { reason: 'max_turns' }                   // 达到 maxTurns 限
  | { reason: 'model_error', error }          // 模型 API 异常
  | { reason: 'image_error' }                 // 图片处理异常
  | { reason: 'prompt_too_long' }             // 上下文超限恢复失败
  | { reason: 'blocking_limit' }              // 上下文超限硬阻断
  | { reason: 'aborted_streaming' }           // 用户中断流式
  | { reason: 'aborted_tools' }               // 用户中断工具
  | { reason: 'stop_hook_prevented' }         // stop hook 阻止继续
  | { reason: 'hook_stopped' }                // 工具中 hook 阻止继续
```

**关键点**：把"循环为什么停"拆成 11 种枚举，**调用方根据 reason 走不同的善后路径**（落库、重试、回滚、上抛）。不是一个统一的"OK / Error" 二态。

### 1.3 continue 站点：系统重试 vs ReAct 推进

`query.ts` 的 `while(true)` 内部有 7 个 `continue` 站点，但**只有 1 个真正消耗 turnCount**：

| 站点 | reason | 是否计入 turnCount | 性质 |
|---|---|---|---|
| 模型 fallback | `tengu_model_fallback_triggered` | ❌ | 系统重试 |
| 上下文 collapse drain | `collapse_drain_retry` | ❌ | 上下文压缩 |
| 上下文 reactive compact | `reactive_compact_retry` | ❌ | 上下文压缩 |
| max_output_tokens 提额 | `max_output_tokens_escalate` | ❌ | 输出 token 不够 |
| max_output_tokens 续写 | `max_output_tokens_recovery` | ❌ | 最多 3 次 |
| stop_hook 阻断 | `stop_hook_blocking` | ❌ | hook 注入提示 |
| token_budget 续推 | `token_budget_continuation` | ❌ | 预算允许 |
| **下一轮 ReAct** | `next_turn` | ✅ | **唯一计入** |

**关键点**：**系统级容错重试不和业务步数共享上限**。`max_turns` 只圈住 ReAct 的"看-想-做"循环次数，不会被压缩重试、模型 fallback 这些幕后修复消耗掉。

### 1.4 工具结果回喂的形态

```ts
// query.ts:1715-1716
const next: State = {
  messages: [...messagesForQuery, ...assistantMessages, ...toolResults],
  ...
}
```

工具结果（包括失败结果）**统一以 `user` 角色 + `tool_result` content block** 拼回 messages 数组：

```json
{
  "role": "user",
  "content": [
    { "type": "tool_result", "tool_use_id": "toolu_xxx", "content": "...", "is_error": false }
  ]
}
```

**关键点**：messages 数组就是模型的"记忆"，所有观察都用同一种结构。模型看上一轮自己说了什么（assistant），看工具返回了什么（user/tool_result），统一在一个上下文里推理。

### 1.5 失败统一以 `tool_result` 回喂

源码：`services/tools/toolExecution.ts:337-490`。任何错误都不抛出循环外：

| 错误来源 | 处理 |
|---|---|
| 工具不存在 | `<tool_use_error>Error: No such tool available: xxx</tool_use_error>` + `is_error: true` |
| 输入 schema 校验失败 | `<tool_use_error>InputValidationError: ...</tool_use_error>` |
| 工具执行抛错 | `<tool_use_error>Error calling tool (xxx): ...</tool_use_error>` |
| 用户拒绝权限 | `<tool_use_error>...</tool_use_error>` |
| 模型 fallback 中途 | `yieldMissingToolResultBlocks` 给每个孤儿 tool_use 补 `is_error` 占位 |

**关键点**：**永远不让 messages 出现"有 tool_use 但没有对应 tool_result"的状态**——这是 Anthropic API 强约束。即便工具压根不存在，也要伪造一个 `is_error: true` 的占位结果。

这一条直接对应用户提的设计："工具未注册应该走下一轮模型输入"——claude-code 就是这么做的。

### 1.6 重复调用：完全交给模型

`runTools` / `StreamingToolExecutor` **不做任何**"上次已经调过同一工具/同样参数"的检测。模型自己看历史 messages，自己决定要不要重调。

但有 3 道天花板兜底：
- `maxTurns` —— 直接终止
- `MAX_OUTPUT_TOKENS_RECOVERY_LIMIT = 3` —— 续写不超过 3 次
- `taskBudget` / `tokenBudget` —— token 预算耗尽提示模型收尾

**关键点**：业务工具的"是否重复调用"由模型基于上下文判断，规则只兜上限。

### 1.7 工具自描述 + fail-closed 默认

`Tool.ts:380-525` 定义工具元数据契约：

```ts
type Tool = {
  name: string
  description: string
  inputSchema: ZodSchema
  isEnabled(): boolean                       // 默认 true
  isConcurrencySafe(input): boolean          // 默认 false（不安全）
  isReadOnly(input): boolean                 // 默认 false（写）
  isDestructive?(input): boolean             // 默认 false
  validateInput?(input, ctx): ValidationResult
  checkPermissions(input, ctx): PermissionResult
  call(input, ctx): AsyncGenerator<...>
}

// Tool.ts:757-769
const TOOL_DEFAULTS = {
  isConcurrencySafe: () => false,    // fail-closed
  isReadOnly: () => false,           // fail-closed
  isDestructive: () => false,
  // ...
}
```

并发执行时，编排层根据 `isConcurrencySafe` 把 tool_use blocks 切成 batch：连续 safe 的并行，遇到 unsafe 必须串行。

### 1.8 特定输入直接短路（不调模型）

入口 `processUserInput.ts:466-551` 的分流：

```ts
// 1. 关键词改写（如 ultraplan）→ 转到 slash command
if (hasUltraplanKeyword(input)) { ... }

// 2. bash 模式（用户输入以 ! 开头）→ processBashCommand
if (mode === 'bash') { ... }

// 3. 斜杠命令（/clear、/cost、/help）→ processSlashCommand
if (input.startsWith('/')) { ... }

// 4. 普通 prompt → 进 query loop
return processTextPrompt(...)
```

`processSlashCommand.tsx` 三种 command.type：

| type | 是否调模型 | 用途 |
|---|---|---|
| `local` | ❌ `shouldQuery: false` | 完全本地副作用（清屏、看成本、退出） |
| `local-jsx` | ❌ `shouldQuery: false` | 本地 UI 交互（登录、设置） |
| `prompt` | ✅ `shouldQuery: true` | 注入预设 prompt 模板，让模型按模板工作（`/review`、`/security-review`） |

**关键点**：高确定性的输入根本不进 query loop。这就是用户提的"特定输入直接走对应操作，不用走模型判断"。

---

## 2. 7 条可借鉴的最佳实践

把上述拆解归纳成 7 条原则，作为我们设计的依据：

| # | 原则 | 在 claude-code 的体现 | 我们应该怎样落地 |
|---|---|---|---|
| 1 | **唯一推进信号** | `needsFollowUp` = 上轮有 `tool_use` | 模型 action ∈ {call_tool, answer, ask_user, handoff}，只有 call_tool 进入下一轮 |
| 2 | **失败即观察** | 任何错误以 `is_error: true` 的 tool_result 回喂 | 工具未注册、参数错、风险拦截、RAG 空集，都构造 Observation 喂回模型 |
| 3 | **结束原因枚举化** | 11 种 terminal reason | 我们至少 8 种：`completed` / `answered_no_tool` / `ask_user` / `wait_confirmation` / `handoff` / `max_steps` / `tool_blocked` / `runtime_error` |
| 4 | **系统重试 ≠ 业务步数** | 7 个 continue 中只有 `next_turn` 计 turnCount | `max_steps` 只算 ReAct 推进；模型重试、JSON 解析失败重生成不计入 |
| 5 | **工具自描述 + fail-closed** | `isReadOnly` / `isConcurrencySafe` 默认 false | `ToolSpec` 已有 risk_level / required_slots，再补 `is_idempotent` / `allow_repeat` / `cost_class` |
| 6 | **重复调用交给模型** | 完全不做去重 | 不在 runtime 强行去重；但通过 observation 把"已执行工具+结果摘要"喂回模型，触发其自主收敛 |
| 7 | **高确定输入直接短路** | slash command 不进 query loop | 三层入口：rules → scene-direct → ReAct，前两层完全不进 ReAct |

---

## 3. 信易贷场景的差异与约束

claude-code 的 query loop 是**面向开放域、模型主导**的代码助手。我们不一样：

| 维度 | claude-code | 信易贷 Agent |
|---|---|---|
| 域 | 开放（任意 IDE 任务） | 封闭（8 类 capability） |
| 工具集 | 60+ 通用工具 + MCP | ~7 个业务工具 + RAG |
| 路由 | 无（模型自己决定调哪个） | 有（standard_intent → capability → allowed_tools） |
| 答案约束 | 文本/代码即可 | KNOWLEDGE_QA 必须有 sources，DATA_QUERY 必须工具调用，AUTHORIZATION 必须人工确认 |
| 风险 | 文件破坏 | 资金风险 + 合规风险（reason: 答错可能涉及金融建议责任） |
| 性能 | 单用户 | 多租户 + 并发 |
| 兜底 | 模型自由生成 | **必须避免幻觉**——降级 ask_user 或 handoff |

因此我们不能照搬"模型主导循环"。我们要的是：**封闭域内的受控 ReAct**。

---

## 4. 分层入口设计

总入口 `ChatRequest` 进来后，按命中优先级路由到 4 层：

```text
┌─────────────────────────────────────────────────────┐
│  L0  rules.guard            （规则强匹配，0 模型调用）  │ 敏感词、空消息、续接
│      ↓ 未命中                                          │
│  L1  scene_direct.dispatch  （高确定场景直达，0 模型）  │ 自我介绍、能力说明、纯闲聊
│      ↓ 未命中                                          │
│  L2  intent_router.classify （1 次模型调用：路由）      │ standard_intent + scene + slots
│      ↓                                                │
│  L3  fast_path 或 ReAct loop（按 capability 选）       │
│      ├─ KNOWLEDGE_QA → fast_path（直调 rag_search）    │ 1 次工具 + 1 次答案
│      └─ 其他 capability → react_loop（多轮 ReAct）     │ 模型主判 + 规则实操
└─────────────────────────────────────────────────────┘
```

### 4.1 L0：规则守卫

**只做"非业务请求"的过滤**，不做意图识别。命中即返回，**不调模型、不调路由**。

```python
# router/rules.py 已有的能力，需要补几条
RULE_GUARDS = [
    ("空消息或纯空白",      lambda req: not req.user_message.strip()),
    ("纯标点 / 表情",       lambda req: not has_chinese_or_letter(req.user_message)),
    ("续接确认",            lambda req: is_continuation(req, last_route)),
    ("敏感词命中黑名单",    lambda req: contains_blacklist(req.user_message)),
]
```

### 4.2 L1：场景直达（scene_direct）

**已知确定性的输入直接命中预设 handler，不进入意图路由**。

| 输入类型 | 命中规则 | Handler | 输出 |
|---|---|---|---|
| 自我介绍类 | `re.search(r'你是(谁\|什么)\|介绍.*你\|你能(做\|干).*什么', msg)` | `IntroductionHandler` | 固定文案 + capability 列表 |
| 能力清单 | `re.search(r'(支持\|能办\|有什么).*(业务\|功能\|服务)', msg)` | `CapabilityListHandler` | 从 catalog 动态生成 |
| 健康检查 | `msg in {"ping", "测试", "在吗"}` | `HealthCheckHandler` | 固定 OK 文案 |
| 已确认动作回复 | `last_pending_action != null and msg in {"确认", "好", "ok"}` | `ConfirmationDispatcher` | 触发 confirm flow |

> **设计准则**：能用 100 行规则覆盖的高频确定性输入，绝不浪费一次模型调用。
> 但要严格约束规则的命中条件——**宁可漏，不可错**。漏了让 L2 接管即可。

### 4.3 L2：意图路由（intent_router）到

唯一一次"全局意图识别"模型调用，已经在我们之前的重构里完成：

- 模型只能从 `StandardIntent` 枚举里选一个
- 输出 `scene + standard_intent + filled_slots + missing_slots + confidence`
- resolver 按 `standard_intent` 精确派生 capability

L2 之后，**capability 已确定**，进入 L3 的快速通道或 ReAct。

### 4.4 L3-fast：单工具快速通道

满足以下全部条件的 capability 走 fast_path，**不进 ReAct 多轮**：

```python
def is_fast_path_eligible(capability: CapabilityPolicy) -> bool:
    return (
        capability.risk_level == "read_only"
        and len(capability.allowed_tools) == 1
        and not capability.confirmation_required
        and not capability.required_slots  # 或 slots 已全部填充
    )
```

适用 capability：
- `knowledge.policy.read`（RAG 单工具）
- `application.status.read`（状态查询单工具）

流程：

```text
route → 直接构造 ToolCall → 执行 → 拼答案 → 返回
                                  ↓
                          失败/空集 → 升级到 ReAct loop
```

### 4.5 L3-react：受控 ReAct 多轮

适用 capability：
- `credit.limit.read`（数据查询，工具可能未注册）
- `product.terms.read`（同上）
- `authorization.link.create`（写动作，需确认）
- `application.draft.create`（多工具串联）

进入第 5 章详细描述。

---

## 5. 受控 ReAct 主循环设计

### 5.1 单轮结构

```text
┌──── ReAct Step N ──────────────────────────────────────┐
│                                                          │
│  [Observation]                                           │
│   ├─ route 决策（仅 N=1 时填充）                         │
│   ├─ 已执行工具 + 结果摘要（N>1 时）                     │
│   └─ tool_unavailable / tool_failed / empty_result      │
│                                                          │
│       ↓                                                 │
│  [Reasoning]  ← 模型主判（结构化 JSON 输出）              │
│   action ∈ {                                             │
│     "call_tool":  { tool_name, arguments, why }         │
│     "answer":     { final_answer, evidence_used }       │
│     "ask_user":   { question }                           │
│     "handoff":    { reason }                             │
│   }                                                      │
│   thought（自由文本，仅诊断不参与决策）                   │
│                                                          │
│       ↓                                                 │
│  [Guardrails]  ← 规则强制                                │
│   ├─ tool_name ∉ allowed_tools           → 拒绝，回喂   │
│   ├─ arguments 不符 spec.input_schema    → 拒绝，回喂   │
│   ├─ risk_level != "read_only" 未确认    → 转 pending   │
│   ├─ risk_level 高                        → 强制 handoff │
│   ├─ KNOWLEDGE_QA 答案缺 sources         → 拒绝，回喂   │
│   └─ 工具实际未注册                       → 构造 obs    │
│                                                          │
│       ↓                                                 │
│  [Action]  ← 规则实操                                    │
│   ├─ call_tool: 执行 → 收 result → 进 N+1               │
│   ├─ answer: 输出 → 终止                                 │
│   ├─ ask_user: 输出 → 终止                               │
│   └─ handoff: 输出 → 终止                                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 5.2 模型 Reasoning 的输出契约

新增 `protocol.ReactStepDecision`：

```python
class ReactAction(BaseModel):
    """模型每轮必须输出的 action。"""
    type: Literal["call_tool", "answer", "ask_user", "handoff"]
    # call_tool 专用
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    # answer 专用
    final_answer: str | None = None
    evidence_used: list[str] = Field(default_factory=list)  # source_id
    # ask_user / handoff 专用
    message: str | None = None

class ReactStepDecision(BaseModel):
    """模型每轮的 reasoning 输出。"""
    model_config = ConfigDict(frozen=True, extra="forbid")

    thought: str = Field(max_length=500, description="思考过程，仅诊断")
    action: ReactAction
    confidence: float = Field(ge=0.0, le=1.0)
```

提示词强约束：
- JSON Schema 注入到 prompt
- 列出当前 capability 的 `allowed_tools` 和每个工具的 `description + input_schema + 是否已执行过`
- 列出当前 Observation
- 明确 4 种 action 的选择规则

### 5.3 Observation 协议（喂给模型下一轮）

新增 `protocol.ReactObservation`：

```python
class ToolExecutionObservation(BaseModel):
    """单次工具执行的观察结果，供下一轮模型推理。"""
    kind: Literal["success", "failed", "unavailable", "empty", "blocked"]
    tool_name: str
    arguments: dict[str, Any]
    # success
    summary: str | None = None              # 工具结果摘要（不超过 N tokens）
    business_status: str | None = None
    sources_count: int = 0
    # failed / unavailable / blocked
    reason: str | None = None
    # 对模型的提示
    hint: str | None = None
    # 同 capability 内还可用的工具
    alternative_tools: list[str] = Field(default_factory=list)


class ReactObservation(BaseModel):
    """喂给模型的下一轮上下文。"""
    step: int
    user_message: str
    route: RouteDecision                    # 第 1 轮路由决策
    capability: CapabilityPolicy            # 当前能力
    available_tools: list[ToolSpecBrief]    # 当前 capability 允许的工具
    executed_tools: list[ToolExecutionObservation]  # 历史执行
    last_observation: ToolExecutionObservation | None  # 最近一次观察
    session_summary: str | None = None
```

### 5.4 工具未注册的处理（用户问的核心场景）

**绝不**返回 None 让规则吞掉。**永远**构造一个 `unavailable` observation 喂回模型：

```python
def _resolve_tool_call(
    self, action: ReactAction, capability: CapabilityPolicy
) -> tuple[ToolCall | None, ToolExecutionObservation | None]:
    # action.tool_name 不在白名单
    if action.tool_name not in capability.allowed_tools:
        return None, ToolExecutionObservation(
            kind="blocked",
            tool_name=action.tool_name,
            arguments=action.arguments,
            reason=f"{action.tool_name} 不在当前能力 {capability.capability_id} 的白名单内。",
            alternative_tools=capability.allowed_tools,
            hint="请从 alternative_tools 里重新选择，或改用 answer/ask_user。",
        )

    # 在白名单但工具尚未在 ToolRegistry 注册（如 query_product_terms 占位）
    spec = self._tool_registry.spec_for_name(action.tool_name)
    if spec is None:
        return None, ToolExecutionObservation(
            kind="unavailable",
            tool_name=action.tool_name,
            arguments=action.arguments,
            reason=f"{action.tool_name} 工具尚未在工具注册表接入。",
            alternative_tools=[t for t in capability.allowed_tools if t != action.tool_name],
            hint=(
                "本能力依赖的具体后端未接入。可考虑：\n"
                "1. 改用 alternative_tools 中的其他工具；\n"
                "2. 改用 answer，明确告知用户该数值查询暂未上线；\n"
                "3. 改用 ask_user 让用户改述为政策类问题（可走 RAG 知识检索）。"
            ),
        )

    # 输入参数校验失败
    try:
        spec.input_schema.validate(action.arguments)
    except ValidationError as exc:
        return None, ToolExecutionObservation(
            kind="blocked",
            tool_name=action.tool_name,
            arguments=action.arguments,
            reason=f"参数不符合 schema：{exc}",
            hint="请按 input_schema 重新构造 arguments。",
        )

    return ToolCall(tool_name=action.tool_name, arguments=action.arguments), None
```

模型在下一轮看到 `last_observation.kind == "unavailable"`，可以选择：

1. `action.type = "call_tool"` 改用 `alternative_tools` 中的工具
2. `action.type = "answer"` 直接告知用户
3. `action.type = "ask_user"` 追问

**不会出现幻觉编造**，因为 prompt 明确"answer 必须基于 executed_tools 里成功的 observation 或 evidence"。

### 5.5 工具失败的处理

| `tool_result.status` | `business_status` | observation.kind | 模型典型应对 |
|---|---|---|---|
| `success` | `OK` / `CREDIT_AMOUNT_FOUND` 等 | `success` | answer |
| `success` | `NOT_FOUND` / `EMPTY` | `empty` | answer（说明未找到）或 ask_user |
| `success` | `PARTIAL_DATA` | `success` + summary 标注 | answer（带不完整提示） |
| `error` | timeout / network | `failed` | call_tool（重试 1 次）或 handoff |
| `error` | permission denied | `blocked` | ask_user（请求授权）或 handoff |
| `error` | 后端 5xx | `failed` | handoff |

**重要**：runtime 不主动重试工具。重试由模型在下一轮基于 observation 决定。但通过 `executed_tools` 的同 (tool_name, arguments) 出现次数限制：

```python
def _detect_repeat_call(action, executed_tools):
    same_calls = [
        e for e in executed_tools
        if e.tool_name == action.tool_name
        and _arguments_equivalent(e.arguments, action.arguments)
    ]
    if len(same_calls) >= 2:
        # 已经调过 2 次相同参数还要再调，强制阻断
        return ToolExecutionObservation(
            kind="blocked",
            ...,
            reason="同 (tool_name, arguments) 已经执行过 2 次。",
            hint="请基于已有结果给出 answer，或改用 ask_user / handoff。",
        )
```

参考 claude-code，重复检测**只兜上限**（执行 2 次相同的就阻断），不主动去重——给模型在小范围内重试的余地。

### 5.6 工具结果如何流转到下一轮

借鉴 claude-code 的 messages 累积模型，但我们的 messages 不是 Anthropic API 的 content blocks（我们用的是 OpenAI 兼容协议），改成结构化 ReactObservation 注入：

```python
def _build_react_messages(observation: ReactObservation) -> list[dict]:
    """每轮重新构造一次完整 messages，不依赖累积。"""
    system = _build_react_system_prompt(observation.capability, observation.available_tools)
    user = _format_observation_as_user_message(observation)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _format_observation_as_user_message(obs: ReactObservation) -> str:
    parts = [
        f"# 用户问题\n{obs.user_message}\n",
        f"# 当前能力\n{obs.capability.capability_id}（{obs.capability.description}）\n",
        f"# 可用工具\n{_render_tool_list(obs.available_tools)}\n",
    ]
    if obs.executed_tools:
        parts.append("# 已执行工具与结果\n")
        for i, e in enumerate(obs.executed_tools, 1):
            parts.append(f"{i}. [{e.kind}] {e.tool_name}({e.arguments}) → {e.summary or e.reason}\n")
    if obs.last_observation:
        parts.append(f"\n# 你上一步的执行观察\n{_render_observation(obs.last_observation)}\n")
    parts.append(_render_decision_instructions(obs.step, obs.capability))
    return "\n".join(parts)
```

**为什么不累积 messages**？  
我们的 reasoning 是结构化 JSON，不是自由文本对话。每轮重新拼一份完整 observation，**让模型上下文始终是 deterministic 的**，方便回放、调试、压测。代价是每轮 prompt 略大，但我们的 step 上限只有 4，可控。

---

## 6. 结束规则（Terminal Decision）

```python
class ReactTerminal(BaseModel):
    reason: Literal[
        "completed",            # 模型 action=answer 通过证据校验
        "answered_no_tool",     # capability 不需要工具直接答（smalltalk）
        "ask_user",             # 模型 action=ask_user，或 max_steps 兜底
        "wait_confirmation",    # 风险动作进入 pending_action
        "handoff",              # 模型 action=handoff，或风险阻断
        "tool_blocked",         # 工具被风险策略硬阻断
        "max_steps",            # 达到 max_steps 仍无 answer/ask_user/handoff
        "runtime_error",        # 模型 JSON 解析失败超限、内部异常
    ]
    final_answer: str
    sources: list[SourceDocument] = []
    pending_action: PendingAction | None = None
```

**结束规则简化为单一核心命题**：

> 每一轮模型 reasoning 都必须给出 4 种 action 之一，**只有 `call_tool` 能让循环进入下一轮**，另外三种都是终止态。`max_steps` 是兜底，触发后**强制降级为 `ask_user`**——绝不允许 max_steps 后让模型自由生成"完整回答"。

**关键差异 vs 当前 RuntimeStepPolicy**：

| 当前 | 新设计 |
|---|---|
| 11 处规则枚举决定下一步 | 4 种 action 由模型选 |
| `route.confidence < 0.7` 规则强制 ASK_USER | 模型自己看 confidence，自己选 ask_user |
| `result.terminal == True` 规则强制 GENERATE_FINAL_ANSWER | 模型看 observation 自己选 answer |
| `allowed_next_tools 全已执行` 规则强制 STOP | 模型看 executed_tools 自己收敛 |
| 工具未注册降级为 ANSWER_WITHOUT_TOOL（幻觉） | 工具未注册作为 observation 喂回模型 |
| max_steps 后给 LLM 自由生成回答 | max_steps 强制降级为 ask_user |

---

## 7. 工具元数据契约扩展

在现有 `ToolSpec` 基础上补足 fail-closed 字段：

```python
class ToolSpec(BaseModel):
    # 已有字段
    name: str
    description: str
    category: str
    input_slots: list[Slot]

    # 新增字段（默认值都是 fail-closed）
    is_read_only: bool = False                  # 默认假定有副作用
    is_idempotent: bool = False                 # 默认假定不幂等
    is_concurrency_safe: bool = False           # 默认假定不可并发
    risk_level: Literal["read_only", "link_create", "state_create", "state_mutate"] = "state_mutate"
    cost_class: Literal["cheap", "medium", "expensive"] = "medium"  # 用于 token_budget
    max_duration_ms: int = 5000                 # 单次执行硬超时

    # observation 摘要器
    result_summarizer: Callable[[ToolResult], str] | None = None
```

注册新工具时必须显式声明：

```python
# 不再允许：
ToolRegistry.register(my_tool)  # 默认 is_read_only=False

# 必须显式：
ToolRegistry.register(
    my_tool,
    spec=ToolSpec(
        name="query_product_terms",
        is_read_only=True,
        is_idempotent=True,
        is_concurrency_safe=True,
        risk_level="read_only",
        cost_class="cheap",
        max_duration_ms=2000,
    ),
)
```

---

## 8. 与现有代码的映射

| 当前文件 | 改动方向 |
|---|---|
| `runtime_policy.py` | 改名 `react_step_engine.py`，删除 11 处规则枚举，保留 4 种 action 的 guardrails |
| `runtime.py` `ControlledAgentLoop.run` | 主循环改为 ReAct：build_observation → model.reason → guardrails → execute → next_observation |
| `protocol.py` | 新增 `ReactAction` / `ReactStepDecision` / `ReactObservation` / `ToolExecutionObservation` / `ReactTerminal` |
| `router/rules.py` | 拆为 `rules_guard.py`（L0）+ `scene_direct.py`（L1） |
| `router/service.py` | 增加 `pre_route_guard()` 和 `scene_direct_dispatch()` 短路 |
| `tools/registry.py` | `execute()` 工具未注册不抛异常，返回 `ToolExecutionObservation(kind="unavailable")` |
| `tools/base.py` | `ToolSpec` 补 `is_read_only` / `is_idempotent` / `cost_class` 等字段 |
| `capabilities/base.py` | `CapabilityPolicy` 增加 `fast_path_eligible: bool` |

---

## 9. 实施路线图

按风险从低到高、依赖从底到顶。

### Phase 1：协议先行（半天）

- 新增 `ReactAction` / `ReactStepDecision` / `ReactObservation` / `ToolExecutionObservation`
- 旧 `RuntimeStepDecision` 保留并标记 deprecated，给现有测试过渡用
- `ToolSpec` 补字段，所有现有工具显式声明 `is_read_only=True`

### Phase 2：工具注册表 fail-closed 化（半天）

- `ToolRegistry.execute()` 工具未注册返回结构化 observation
- 输入 schema 校验失败返回结构化 observation
- 单元测试覆盖：未注册 / 参数错 / 工具抛异常 / 工具超时 4 个场景

### Phase 3：L0 / L1 入口短路（1 天）

- 实现 `RulesGuard` 和 `SceneDirectDispatcher`
- 在 `IntentRouter.route()` 之前接入
- 对照 capability catalog 列出能短路的输入清单
- 编写 8~12 条直达入口的测试用例

### Phase 4：L3-fast 单工具快速通道（半天）

- 在 `runtime.py` 增加 `_run_fast_path()`
- 适用 `knowledge.policy.read` / `application.status.read`
- 失败 / 空集自动升级到 ReAct loop

### Phase 5：ReactStepEngine（核心，2 天）

- 实现 `react_step_engine.py`：guardrails + tool 执行
- 实现 `react_prompt.py`：observation → prompt 构造
- 实现 `react_runtime.py`：主循环
- `runtime.py` `ControlledAgentLoop` 切换到新引擎
- max_steps 兜底降级为 ask_user

### Phase 6：联调验证（1 天）

- 跑 `scripts/diag_intent_router.py`：路由仍 9/9 正确
- 跑 `scripts/smoke_chat_endpoint.py`：扩展到 12 条 query
  - 包括"小微税贷利率"（工具未注册场景，应 fallback 到 RAG 或 ask_user，不能幻觉）
  - 包括"生成授权链接"（state_create 场景，应进入 pending_action）
  - 包括"信易贷有什么准入条件"（KNOWLEDGE_QA 直答）
- 写 `test/contract/test_react_loop.py` 覆盖 8 种 terminal reason

---

## 10. 决策表速查

### 10.1 入口分流决策

| 输入特征 | 路由 | 模型调用次数 |
|---|---|---|
| 空消息 / 纯标点 | L0 reject | 0 |
| `你是谁` / `能做什么` | L1 IntroductionHandler | 0 |
| `pending_action 存在 + 用户回"确认"` | L1 ConfirmationDispatcher | 0 |
| 一般业务问句（policy / data / auth） | L2 IntentRouter → L3 | 1 路由 + N reasoning |
| KNOWLEDGE_QA + 单工具 + 只读 | L3-fast | 1 路由 + 1 答案生成 |
| 其他 capability | L3-react | 1 路由 + N×reasoning |

### 10.2 单轮 ReAct 决策

| 模型输出 action | 满足 guardrails | 不满足 guardrails |
|---|---|---|
| `call_tool` + 白名单工具 + 参数合法 + 风险只读 | 执行 → 下一轮 | （不可能） |
| `call_tool` + 工具未注册 | （阻断） | obs.kind=unavailable → 下一轮 |
| `call_tool` + 不在白名单 | （阻断） | obs.kind=blocked → 下一轮 |
| `call_tool` + risk!=read_only | （阻断） | 转 pending_action → 终止 wait_confirmation |
| `call_tool` + 重复 2 次 | （阻断） | obs.kind=blocked → 下一轮（提示收敛） |
| `answer` + KNOWLEDGE_QA + 有 sources | 直接终止 completed | obs.kind=blocked（缺证据）→ 下一轮 |
| `answer` + DATA_QUERY + 至少 1 次 success tool | 直接终止 completed | obs.kind=blocked（缺数据）→ 下一轮 |
| `ask_user` | 直接终止 ask_user | - |
| `handoff` | 直接终止 handoff | - |

### 10.3 终止 reason 与 stop_reason 映射

| ReactTerminal.reason | ChatResponse.stop_reason | 落库的 turn_summary 字段 |
|---|---|---|
| `completed` | `completed` | 含 sources / business_status |
| `answered_no_tool` | `answered_without_tool` | - |
| `ask_user` | `missing_slots` | missing_slots / clarify_question |
| `wait_confirmation` | `waiting_confirmation` | pending_action |
| `handoff` | `handoff` | handoff_reason |
| `tool_blocked` | `tool_blocked` | block_reason |
| `max_steps` | `max_steps`（实际答案是降级 ask_user） | step_count |
| `runtime_error` | `runtime_error` | error_class / error_message |

---

## 11. 安全网（Safety Nets）

借鉴 claude-code 的"系统重试不计 turnCount"原则，我们也设计 4 道安全网，**任何一道触发都不消耗 max_steps**：

| 安全网 | 触发条件 | 处理 | 上限 |
|---|---|---|---|
| **JSON 解析重试** | 模型输出非合法 JSON 或不通过 schema | 用更严格的 system prompt 重新调一次 | 1 次/step |
| **schema 字段补齐** | 模型遗漏 `confidence` 等可选字段 | 用默认值兜底，不重试 | - |
| **空 observation 容错** | guardrails 拒绝模型 action 后 observation 仍空 | 注入"请重新选择 action"提示 | 1 次/step |
| **超时熔断** | 单 step 总耗时 > 30s | 终止并返回 runtime_error | - |

**`max_steps = 4` 的语义**：4 次"模型 reasoning + 一次 tool execution"或终止 action 的循环。系统级重试、JSON 修复、prompt 修补都不算。

---

## 12. 与 claude-code 设计的取舍对照

| 维度 | claude-code | 我们 | 理由 |
|---|---|---|---|
| messages 累积 | 累积式（messages = [...prev, ...new]） | 每轮重新构造 observation | 我们结构化、可压测、可回放，且 step 数量小 |
| 工具去重 | 完全交给模型 | 同 (name, args) 第 3 次硬阻断 | 业务工具调用有金钱/合规成本，必须兜底 |
| 模型 action 集合 | 隐式（call_tool 或 final_answer） | 显式 4 种 enum | 必须强约束 ask_user / handoff 路径 |
| 幻觉防护 | 弱（开放域） | 强（KNOWLEDGE_QA 必须 sources，max_steps 强制 ask_user） | 金融 / 合规要求 |
| 入口分层 | 单层 query loop + slash command 短路 | 4 层（L0~L3） | 我们的高确定输入占比远大于 claude-code |
| 工具并发 | 编排层 batch | 暂不实现，预留 `is_concurrency_safe` 字段 | 业务工具数量少、串行简单 |
| 上下文压缩 | 复杂的 collapse / reactive_compact / snip | 不实现 | 我们的 turn 短，session 摘要已经够 |

---

## 附录 A：参考源码索引

| 主题 | claude-code 文件 | 行号 |
|---|---|---|
| 主循环 | `query.ts` | 241-1729 |
| Terminal reason 定义 | `query/transitions.ts` | - |
| 工具执行 | `services/tools/toolExecution.ts` | 337-490 |
| 工具编排（并发） | `services/tools/toolOrchestration.ts` | 19-150 |
| 流式工具执行 | `services/tools/StreamingToolExecutor.ts` | - |
| Tool 接口 | `Tool.ts` | 380-525 / 706-769 |
| Tool fail-closed 默认 | `Tool.ts` | 757-769 |
| 输入分流 | `utils/processUserInput/processUserInput.ts` | 466-588 |
| Slash command 处理 | `utils/processUserInput/processSlashCommand.tsx` | 309-920 |
| Stop hook | `query/stopHooks.ts` | - |
| Token budget | `query/tokenBudget.ts` | - |
| Max output tokens 续写 | `query.ts` | 1185-1256 |

## 附录 B：信易贷场景测试矩阵

实施完成后，必须 100% 通过：

| 输入 | 期望路由 | 期望终止 | 期望 sources | 期望 stop_reason |
|---|---|---|---|---|
| `（空字符串）` | L0 reject | ask_user | - | missing_slots |
| `你是谁` | L1 IntroductionHandler | answered_no_tool | - | answered_without_tool |
| `信易贷的准入条件是什么` | L3-fast knowledge.policy.read | completed | ≥3 | completed |
| `高风险标签都包括哪些情况` | L3-fast knowledge.policy.read | completed | ≥3 | completed |
| `查一下 ABC 公司的授信额度` | L3-react credit.limit.read | completed | 0 | completed |
| `小微税贷的利率是多少` | L3-react product.terms.read → unavailable → 模型 fallback 到 answer | ask_user 或 answer 明告未上线 | - | answered_without_tool 或 missing_slots |
| `帮我给 ABC 公司生成授权链接` | L3-react authorization.link.create → 转 pending | wait_confirmation | - | waiting_confirmation |
| `XXX 公司有哪些贷款申请` | L3-react application.status.read | completed 或 ask_user | - | completed / missing_slots |
| `ABCDEFGHIJK` (乱码) | L0 reject 或 L2 confidence<0.7 | ask_user | - | missing_slots |
| `谢谢` | L1 SmalltalkHandler | answered_no_tool | - | answered_without_tool |

## 附录 C：术语表

- **capability**：业务能力，对应 `CapabilityPolicy`，由 catalog 静态定义。每个 capability 唯一对应一个 `standard_intent`。
- **scene**：业务场景维度，如 `KNOWLEDGE_QA` / `DATA_QUERY` / `AUTHORIZATION`。
- **standard_intent**：模型可输出的意图枚举，由 catalog 派生。
- **action**：模型每轮 reasoning 输出的动作类型，∈ {call_tool, answer, ask_user, handoff}。
- **observation**：上一轮工具执行结果的结构化封装，喂给模型下一轮。
- **guardrails**：规则强制层，对模型输出 action 的合法性校验，不通过则构造 blocked observation 回喂。
- **fast_path**：单工具只读 capability 的快速通道，跳过 ReAct 多轮。
- **fail-closed**：默认禁止/不安全，必须显式声明才放行。

---

## 修订记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-05-09 | 初版：claude-code 精读 + 信易贷 ReAct 设计 |
