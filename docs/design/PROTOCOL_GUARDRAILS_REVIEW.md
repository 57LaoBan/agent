# Agent 协议设计与多重护栏机制

> **面试视角**：本文档从系统架构角度阐述如何通过"协议先行 + 多重护栏"设计模式，构建一个可控、可观测、可扩展的 Agent 系统。适用于讨论 LLM Agent 工程化、系统设计、风险控制等话题。

---

## 一、核心设计理念

### 1.1 问题背景

在构建生产级 Agent 系统时，我们面临三个核心挑战：

1. **不确定性**：LLM 输出格式、内容、行为都存在不确定性
2. **安全性**：Agent 可能执行高风险操作（状态变更、资金操作等）
3. **可控性**：需要在保持 Agent 灵活性的同时，确保系统行为可预测

### 1.2 设计原则

我们采用"协议先行 + 多重护栏"的设计模式，核心原则：

**原则 1：模型只负责语义理解，不负责执行决策**
- 模型输出是"建议"而非"指令"
- 所有执行权限由后端策略控制

**原则 2：结构化先于执行**
- 模型输出必须先通过 Pydantic 校验转为结构化对象
- 不符合协议的输出在进入业务逻辑前被拦截

**原则 3：多层防御，纵深防护**
- 路由层、ReAct 层、工具层各有独立校验
- 单点失效不会导致系统失控

**原则 4：异常不穿透**
- 工具执行失败转为结构化 ToolResult
- 协议错误转为 invalid_hint 让模型修正
- 主循环始终保持可控状态

**原则 5：证据驱动回答**
- 知识问答必须基于检索证据
- 状态变更必须基于用户确认快照

---

## 二、系统架构与数据流

### 2.1 整体架构

```text
┌─────────────────────────────────────────────────────────────┐
│                    ControlledAgentLoop                      │
│  (统一入口，编排路由、ReAct、工具执行、响应构建)              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │         ReactRuntime.run()            │
        │  (协调路由、记忆、工具执行、ReAct)     │
        └───────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  路由协议层   │   │  ReAct 协议层 │   │  工具执行层   │
│              │   │              │   │              │
│ 护栏 1-3     │   │ 护栏 4-5     │   │ 护栏 6-7     │
└──────────────┘   └──────────────┘   └──────────────┘
```

### 2.2 核心数据流

```text
用户请求
  ↓
[路由协议层]
  ModelIntentRouter.route()
    → 模型输出 JSON
    → Pydantic 校验 → ModelRouteOutput
    → 护栏 1: 非 JSON / Schema 错误 → 修复或降级
  
  RoutePolicy.apply()
    → 后端能力策略映射
    → 护栏 2: 低置信度 / 缺少能力 → 追问
    → 护栏 3: 缺少必填槽位 → 追问
    → 输出: RouteDecision (含工具白名单、风险等级)
  ↓
[ReAct 协议层]
  ReactStepEngine.step()
    → 模型输出 JSON
    → Pydantic 校验 → ReactStepDecision
    → 护栏 4: 动作类型校验 (只允许 4 种)
    → 护栏 5: 证据校验 (知识问答必须有 sources)
  
  _handle_call_tool()
    → 护栏 6: 工具白名单校验
    → 护栏 7: 重复调用检测
    → 护栏 8: 高风险工具 → pending_action
  ↓
[工具执行层]
  ToolRegistry.execute()
    → 护栏 9: 工具注册检查
    → 护栏 10: 输入 Schema 校验
    → 护栏 11: pending_action 快照校验
    → 执行工具
    → 护栏 12: 输出 Schema 校验
    → 失败 → ToolResult (不抛异常)
  ↓
结构化响应 (answer + trace + evidence + decision)
```

---

## 三、核心模块设计

### 3.1 路由协议层

#### 模块职责
- 将用户自然语言转为标准意图 (standard_intent)
- 映射到后端能力策略 (CapabilityPolicy)
- 确定工具白名单、必填槽位、风险等级

#### 关键组件

**ModelIntentRouter** (`src/xinyidai_agent/router/model_router.py`)

```python
def route(self, request: AgentRequest) -> RouteDecision:
    # 1. 调用 LLM 进行意图分类
    raw = model.complete(...)
    
    # 2. 解析 JSON (护栏 1)
    payload = _parse_json(raw)
    if not payload:
        # 尝试修复一次
        payload = _fix_json(raw)
    
    # 3. 结构化校验 (护栏 1)
    model_output = ModelRouteOutput.model_validate(payload)
    
    # 4. 转为路由决策
    return _to_route_decision(model_output)
```

**设计考量**：
- **为什么要先结构化？** 防止模型输出的自由文本直接进入业务逻辑，导致解析错误或注入攻击
- **为什么要修复一次？** 平衡鲁棒性和成本，给模型一次纠错机会
- **为什么失败返回 unknown_route？** 降级策略，确保系统不会因路由失败而崩溃

**RoutePolicy** (`src/xinyidai_agent/router/policy.py`)

```python
def apply(self, model_output: ModelRouteOutput) -> RouteDecision:
    # 1. 置信度检查 (护栏 2)
    if model_output.confidence < threshold:
        return _ask_user_to_clarify()
    
    # 2. 能力策略解析 (护栏 2)
    capability = CapabilityResolver.resolve(model_output.route)
    if not capability:
        return _ask_user_to_clarify()
    
    # 3. 槽位检查 (护栏 3)
    missing_slots = _check_required_slots(capability, request)
    if missing_slots:
        return _ask_user_for_slots(missing_slots)
    
    # 4. 生成最终决策
    return RouteDecision(
        capability_id=capability.id,
        allowed_tools=capability.allowed_tools,  # 后端控制
        risk_level=capability.risk_level,        # 后端控制
        confirmation_required=RiskPolicy.requires_confirmation(...),
    )
```

**设计考量**：
- **为什么模型不能输出 allowed_tools？** 权限控制必须由后端策略决定，不能信任模型输出
- **为什么要分离 ModelRouteOutput 和 RouteDecision？** 前者是模型理解，后者是执行决策，分离关注点
- **为什么缺槽位直接追问？** 避免进入 ReAct 循环浪费 token，快速收集必要信息

#### 模块交互
```text
ModelIntentRouter → ModelRouteOutput (模型理解)
       ↓
RoutePolicy → RouteDecision (执行决策)
       ↓
ReactRuntime (消费路由决策)
```

---

### 3.2 ReAct 协议层

#### 模块职责
- 编排多步推理和工具调用
- 校验每步动作的合法性
- 处理动作失败和异常情况

#### 关键组件

**ReactStepEngine** (`src/xinyidai_agent/runtime/react_engine.py`)

```python
def step(self, observation: ReactObservation) -> ReactStepOutcome:
    # 1. 推理下一步动作
    decision = _reason(observation)
    
    # 2. 结构化校验 (护栏 4)
    action = ReactStepDecision.model_validate(decision)
    
    # 3. 动作分发
    if action.type == "call_tool":
        return _handle_call_tool(action, observation)
    elif action.type == "answer":
        return _validate_answer(action, observation)
    elif action.type == "ask_user":
        return _validate_ask_user(action)
    elif action.type == "handoff":
        return _validate_handoff(action)
```

**ReactStepDecision 协议** (`src/xinyidai_agent/protocol.py`)

```python
class ReactAction(BaseModel):
    type: Literal["call_tool", "answer", "ask_user", "handoff"]
    # 模型不允许输出 "terminal"
    
    class Config:
        extra = "forbid"  # 拒绝模型输出额外字段
```

**设计考量**：
- **为什么只允许 4 种动作？** 限制模型行为空间，确保每种动作都有对应的校验逻辑
- **为什么模型不能输出 terminal？** terminal 是后端校验通过后的内部状态，不能由模型决定
- **为什么用 extra="forbid"？** 防止模型输出未定义字段绕过校验

**_handle_call_tool** (护栏 6-8)

```python
def _handle_call_tool(action, observation):
    # 护栏 6: 工具白名单检查
    if action.tool_name not in observation.capability.allowed_tools:
        return invalid_action("工具不在白名单")
    
    # 护栏 7: 重复调用检测
    if _is_duplicate_call(action, history):
        return invalid_action("重复调用同一工具")
    
    # 护栏 8: 高风险工具转确认
    if _requires_confirmation(action.tool_name):
        return pending_action_required(action)
    
    # 执行工具
    return ToolRegistry.execute_observed(action, observation)
```

**_validate_answer** (护栏 5)

```python
def _validate_answer(action, observation):
    # 护栏 5: 证据校验
    if observation.capability.requires_evidence:
        if not _has_success_observation_with_sources(history):
            return invalid_action("必须先检索获取证据")
        if not action.evidence_used:
            return invalid_action("必须声明使用的证据")
    
    return terminal(action.final_answer)
```

**设计考量**：
- **为什么要证据校验？** 防止模型编造答案，确保知识问答基于真实数据源
- **为什么高风险工具不直接执行？** 状态变更操作必须经过用户确认，不能由模型单方面决定
- **为什么返回 invalid_action 而不是抛异常？** 让模型有机会修正错误，而不是直接终止流程

#### 模块交互
```text
ReactRuntime._run_react_steps()
       ↓
ReactObservationBuilder.build()  (构造上下文)
       ↓
ReactStepEngine.step()  (推理 + 校验)
       ↓
ToolRegistry.execute()  (执行工具)
       ↓
ReactObservation (下一轮输入)
```

---

### 3.3 工具执行层

#### 模块职责
- 统一管理所有工具的注册和执行
- 校验工具输入输出的合法性
- 处理工具执行失败，转为结构化结果

#### 关键组件

**ToolRegistry** (`src/xinyidai_agent/tools/registry.py`)

```python
def execute(self, tool_call: ToolCall, context: ExecutionContext) -> ToolResult:
    # 护栏 9: 工具注册检查
    tool = self._tools.get(tool_call.name)
    if not tool:
        return ToolResult(status="TOOL_NOT_FOUND")
    
    # 护栏 10: 输入 Schema 校验
    try:
        validated_input = tool.spec.validate_input(tool_call.arguments)
    except ValidationError as e:
        return ToolResult(status="TOOL_SCHEMA_ERROR", error=str(e))
    
    # 护栏 11: pending_action 快照校验
    if tool.spec.requires_confirmation:
        if not _validate_pending_action_snapshot(context):
            return ToolResult(status="TOOL_BLOCKED", reason="快照校验失败")
    
    # 执行工具
    try:
        output = tool.execute(validated_input, context)
    except ToolTimeout:
        return ToolResult(status="TOOL_TIMEOUT")
    except Exception as e:
        return ToolResult(status="TOOL_EXECUTION_ERROR", error=str(e))
    
    # 护栏 12: 输出 Schema 校验
    try:
        validated_output = tool.spec.validate_output(output)
    except ValidationError as e:
        return ToolResult(status="TOOL_SCHEMA_ERROR", error=str(e))
    
    return ToolResult(status="SUCCESS", data=validated_output)
```

**设计考量**：
- **为什么所有工具必须经过 ToolRegistry？** 统一入口便于实施权限控制、日志记录、监控告警
- **为什么失败返回 ToolResult 而不是抛异常？** 异常会穿透主循环，ToolResult 可以转为 observation 让模型处理
- **为什么要输入输出都校验？** 输入校验防止注入攻击，输出校验确保工具返回符合预期格式

**PendingActionValidator** (`src/xinyidai_agent/action_validator.py`)

```python
def validate_pending_action_snapshot(pending_action, current_context):
    # 1. 会话校验
    if pending_action.session_id != current_context.session_id:
        return False
    
    # 2. 过期校验
    if pending_action.expired_at < now():
        return False
    
    # 3. 槽位快照校验
    current_slots = _extract_slots(current_context)
    if pending_action.slot_snapshot != current_slots:
        return False  # 关键参数已变化
    
    # 4. 前置条件哈希校验
    current_hash = _compute_precondition_hash(current_context)
    if pending_action.precondition_hash != current_hash:
        return False  # 业务状态已变化
    
    return True
```

**设计考量**：
- **为什么需要快照校验？** 防止 TOCTOU (Time-of-Check-Time-of-Use) 攻击，确认时的状态和执行时的状态一致
- **为什么要哈希前置条件？** 业务状态可能在确认和执行之间发生变化（如额度已用完），必须重新校验
- **为什么要过期时间？** 防止旧的确认被恶意重放

#### 模块交互
```text
ReactStepEngine._handle_call_tool()
       ↓
ToolRegistry.execute()
       ↓
PendingActionValidator.validate() (高风险工具)
       ↓
Tool.execute() (实际业务逻辑)
       ↓
ToolResult → ReactObservation
```

---

## 四、关键设计决策与权衡

### 4.1 为什么不用 Function Calling？

**决策**：自定义 JSON 协议 + Pydantic 校验

**权衡**：
- ✅ 优势：完全控制协议格式，可以添加自定义字段（如 confidence、evidence_used）
- ✅ 优势：可以在不同 LLM 之间保持一致的协议
- ✅ 优势：可以实现协议修复逻辑
- ❌ 劣势：需要维护 prompt 和解析逻辑
- ❌ 劣势：模型可能不遵守协议

**适用场景**：需要精细控制协议、跨模型兼容、自定义字段的场景

### 4.2 为什么要分离路由和 ReAct？

**决策**：路由只做意图分类，ReAct 负责工具编排

**权衡**：
- ✅ 优势：简单请求不进入 ReAct，节省 token 和延迟
- ✅ 优势：路由模型可以用更小更快的模型
- ✅ 优势：路由失败不影响 ReAct 的稳定性
- ❌ 劣势：增加了系统复杂度
- ❌ 劣势：路由和 ReAct 的边界需要仔细设计

**适用场景**：有大量简单请求、需要优化成本和延迟的场景

### 4.3 为什么要 pending_action 而不是直接确认？

**决策**：高风险操作先生成快照，确认后再校验快照

**权衡**：
- ✅ 优势：防止 TOCTOU 攻击
- ✅ 优势：可以在确认页面展示完整的操作详情
- ✅ 优势：可以实现批量确认、延迟确认等高级功能
- ❌ 劣势：需要维护 pending_action 的存储和过期逻辑
- ❌ 劣势：增加了一次额外的校验开销

**适用场景**：金融、医疗等高风险领域，状态变更操作必须严格控制

### 4.4 为什么要证据校验？

**决策**：知识问答必须基于检索到的 sources

**权衡**：
- ✅ 优势：防止模型编造答案（幻觉）
- ✅ 优势：可以追溯答案来源，提高可信度
- ✅ 优势：可以实现引用、溯源等功能
- ❌ 劣势：模型可能检索到无关内容也强行回答
- ❌ 劣势：增加了一次检索的延迟和成本

**适用场景**：政策咨询、法律咨询等需要准确性和可追溯性的场景

---

## 五、可扩展性设计

### 5.1 新增工具

**扩展点**：ToolRegistry + ToolSpec

```python
# 1. 定义工具 Spec
class NewToolSpec(ToolSpec):
    name = "new_tool"
    category = "data_query"
    input_schema = {...}
    output_schema = {...}
    requires_confirmation = False

# 2. 实现工具逻辑
class NewTool(BaseTool):
    def execute(self, input, context):
        # 业务逻辑
        return output

# 3. 注册工具
ToolRegistry.register(NewTool())
```

**无需修改**：
- 路由逻辑（通过能力策略配置）
- ReAct 逻辑（自动支持新工具）
- 校验逻辑（基于 ToolSpec 自动校验）

### 5.2 新增能力

**扩展点**：CapabilityPolicy

```python
# 在配置文件中添加新能力
{
  "capability_id": "new_capability",
  "standard_intent": "new_intent",
  "allowed_tools": ["tool_a", "tool_b"],
  "required_slots": ["slot_1"],
  "risk_level": "medium",
  "requires_evidence": false,
  "max_react_steps": 5
}
```

**无需修改**：
- 路由代码（通过 CapabilityResolver 自动解析）
- ReAct 代码（根据能力策略自动调整行为）

### 5.3 新增护栏

**扩展点**：Validator 链

```python
# 在 ToolRegistry.execute() 中添加新的校验器
def execute(self, tool_call, context):
    # 现有校验
    ...
    
    # 新增自定义校验
    if not CustomValidator.validate(tool_call, context):
        return ToolResult(status="CUSTOM_VALIDATION_FAILED")
    
    # 执行工具
    ...
```

**设计模式**：责任链模式，每个校验器独立，易于添加和移除

### 5.4 新增动作类型

**扩展点**：ReactAction + ReactStepEngine

```python
# 1. 扩展协议
class ReactAction(BaseModel):
    type: Literal["call_tool", "answer", "ask_user", "handoff", "new_action"]

# 2. 添加处理逻辑
def step(self, observation):
    ...
    elif action.type == "new_action":
        return _handle_new_action(action)
```

**注意**：新增动作类型需要同步更新 prompt、校验逻辑、测试用例

---

## 六、生产实践经验

### 6.1 监控指标

**路由层**：
- 路由成功率、失败率、降级率
- 各意图的分布和置信度分布
- JSON 解析失败率、Schema 校验失败率

**ReAct 层**：
- 平均步数、最大步数触发率
- 各动作类型的分布
- invalid_action 的原因分布

**工具层**：
- 各工具的调用量、成功率、延迟
- pending_action 的确认率、取消率、过期率
- 工具失败的原因分布

### 6.2 常见问题与解决

**问题 1：模型不遵守协议**
- 现象：输出非 JSON、缺少必填字段、输出未定义字段
- 解决：强化 prompt、添加 few-shot 示例、实现协议修复逻辑

**问题 2：ReAct 循环不收敛**
- 现象：达到 max_steps 仍未结束
- 解决：优化 observation 构造、添加循环检测、调整 max_steps

**问题 3：工具调用失败率高**
- 现象：参数错误、超时、业务异常
- 解决：优化工具 Schema、添加参数提示、实现重试逻辑

**问题 4：pending_action 过期率高**
- 现象：用户确认前状态已变化
- 解决：缩短过期时间、优化确认流程、添加状态锁定

### 6.3 性能优化

**优化 1：路由缓存**
- 相同请求的路由结果可以缓存（考虑会话状态）
- 减少路由模型调用次数

**优化 2：工具并行执行**
- 独立工具可以并行调用
- 需要修改 ReAct 协议支持批量 call_tool

**优化 3：Prompt 压缩**
- observation 中的历史信息可以摘要
- 减少 ReAct 每步的 token 消耗

**优化 4：模型选择**
- 路由用小模型（如 GPT-3.5）
- ReAct 用大模型（如 GPT-4）
- 根据能力复杂度动态选择

---

## 七、面试讲述框架

### 7.1 开场（1-2 分钟）

"我们的 Agent 系统采用'协议先行 + 多重护栏'的设计模式。核心理念是**模型只负责语义理解，不负责执行决策**。所有执行权限由后端策略控制，模型输出必须先通过 Pydantic 校验转为结构化对象，不符合协议的输出在进入业务逻辑前被拦截。"

### 7.2 架构层（2-3 分钟）

"系统分为三层：**路由协议层**负责意图分类和能力映射，**ReAct 协议层**负责多步推理和动作校验，**工具执行层**负责统一的工具管理和执行。每层都有独立的护栏，形成纵深防御。"

"数据流是这样的：用户请求先经过路由层，模型输出 JSON 经过 Pydantic 校验后，由后端能力策略派生工具白名单和风险等级。然后进入 ReAct 层，模型每步只能输出 4 种动作，每种动作都有对应的校验逻辑。最后到工具层，统一经过 ToolRegistry 执行，输入输出都要 Schema 校验，失败转为结构化 ToolResult 而不是抛异常。"

### 7.3 细节层（3-5 分钟，根据面试官兴趣展开）

**如果问路由**：
"路由分为两步：ModelIntentRouter 负责语义理解，输出标准意图和置信度；RoutePolicy 负责执行决策，根据后端能力策略派生工具白名单、必填槽位、风险等级。这样分离的好处是模型不能自创工具权限，所有权限都来自配置。"

**如果问 ReAct**：
"ReAct 的关键是模型不能输出 terminal，只能输出 call_tool、answer、ask_user、handoff 四种动作。terminal 是后端校验通过后的内部状态。比如 answer 动作，如果能力要求 requires_evidence，后端会校验是否先调用了检索工具并拿到了 sources，没有的话返回 invalid_action 让模型修正。"

**如果问工具**：
"所有工具必须经过 ToolRegistry，这是统一入口。高风险工具不会直接执行，而是先生成 pending_action，保存槽位快照和前置条件哈希。用户确认后，再次校验快照，防止 TOCTOU 攻击。工具执行失败不会抛异常，而是返回结构化 ToolResult，转为 observation 让模型处理。"

### 7.4 权衡层（2-3 分钟）

"这个设计的核心权衡是**灵活性 vs 可控性**。我们选择了可控性优先，通过协议和护栏限制模型行为空间，代价是增加了系统复杂度。但在金融、医疗等高风险领域，这个权衡是值得的。"

"另一个权衡是**性能 vs 安全性**。每层都有校验会增加延迟，但我们通过分层设计，简单请求可以在路由层直接返回，不进入 ReAct，平衡了性能和安全性。"

### 7.5 扩展层（1-2 分钟）

"系统的扩展性主要体现在三个方面：新增工具只需定义 ToolSpec 和实现逻辑，无需修改路由和 ReAct；新增能力只需配置 CapabilityPolicy，无需修改代码；新增护栏可以通过责任链模式添加到 Validator 链中。"

### 7.6 收尾（1 分钟）

"总结一下，这个系统的核心是**协议先行**，模型输出必须符合协议；**多重护栏**，每层都有独立校验；**异常不穿透**，失败转为结构化结果；**证据驱动**，知识问答基于检索，状态变更基于确认。这些设计让我们在保持 Agent 灵活性的同时，确保了系统的可控性和安全性。"

---

## 八、附录：核心代码路径

```text
路由协议层：
  src/xinyidai_agent/router/model_router.py    # ModelIntentRouter
  src/xinyidai_agent/router/policy.py          # RoutePolicy
  src/xinyidai_agent/protocol.py               # ModelRouteOutput

ReAct 协议层：
  src/xinyidai_agent/runtime/react_engine.py   # ReactStepEngine
  src/xinyidai_agent/runtime/react_loop.py     # ReactRuntime
  src/xinyidai_agent/protocol.py               # ReactStepDecision

工具执行层：
  src/xinyidai_agent/tools/registry.py         # ToolRegistry
  src/xinyidai_agent/pending_actions.py        # PendingActionBuilder
  src/xinyidai_agent/action_validator.py       # PendingActionValidator

统一入口：
  src/xinyidai_agent/runtime/__init__.py       # ControlledAgentLoop
```
