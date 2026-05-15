# Agent 请求路由与 ReAct 执行机制

> **面试视角**：本文档从系统架构角度阐述如何通过"分层路由 + 快速通道 + ReAct 循环"设计模式，构建一个高效、灵活、可扩展的 Agent 请求处理系统。适用于讨论系统设计、性能优化、架构演进等话题。

---

## 一、核心设计理念

### 1.1 问题背景

在构建生产级 Agent 系统时，我们面临的核心挑战：

1. **效率问题**：不是所有请求都需要复杂的 ReAct 推理
2. **成本问题**：LLM 调用成本高，需要精细化控制
3. **延迟问题**：用户期望快速响应，多步推理会增加延迟
4. **复杂度问题**：需要在简单和复杂请求之间找到平衡

### 1.2 设计原则

我们采用"分层路由 + 快速通道 + ReAct 循环"的设计模式，核心原则：

**原则 1：按确定性分层处理**
- 确定性高的请求用规则处理（0 次 LLM 调用）
- 确定性中的请求用单工具处理（1-2 次 LLM 调用）
- 确定性低的请求用 ReAct 处理（多次 LLM 调用）

**原则 2：能力策略统一收口**
- 路由只负责意图分类，不授权工具
- 后端能力策略决定工具白名单、必填槽位、风险等级
- 模型理解和执行决策分离

**原则 3：快速通道优先**
- 简单请求不进入 ReAct 循环
- 单工具场景直接构造工具调用
- 减少不必要的模型调用

**原则 4：ReAct 作为兜底**
- 复杂请求才进入 ReAct 循环
- ReAct 负责多步推理和工具编排
- 循环有明确的终止条件

**原则 5：可观测性优先**
- 每层决策都有明确的输出
- 路由决策、工具调用、模型推理都可追溯
- 便于问题排查和系统优化

---

## 二、系统架构与数据流

### 2.1 整体架构

```text
┌─────────────────────────────────────────────────────────────┐
│                    ControlledAgentLoop                      │
│         (统一入口，编排路由、执行、响应构建)                  │
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
│  L0/L1 短路   │   │  L2 意图路由  │   │  L3 执行层   │
│              │   │              │   │              │
│ 规则 + 模板   │   │ 模型 + 策略   │   │ 工具 + ReAct │
└──────────────┘   └──────────────┘   └──────────────┘
```

### 2.2 四层处理模型

```text
L0: 规则守卫 (RulesGuard)
  ├─ 空输入 / 无意义输入 → 拒绝或追问
  ├─ 确认续接 (有 pending_action) → 确认流程
  └─ 0 次 LLM 调用

L1: 场景直达 (SceneDirectDispatcher)
  ├─ 问候 / 自我介绍 → 模板回复
  ├─ 能力清单 / 帮助 → 模板回复
  ├─ 感谢 / 再见 → 模板回复
  └─ 0 次 LLM 调用

L2: 意图路由 (ModelIntentRouter + RoutePolicy)
  ├─ 规则路由 (RuleBasedRouter) → 恢复上下文
  ├─ 模型路由 (ModelIntentRouter) → 语义分类
  ├─ 能力策略 (RoutePolicy) → 执行决策
  └─ 1 次 LLM 调用

L3: 执行层 (FastPath / DirectTool / ReAct)
  ├─ 缺槽位 → 直接追问 (0 次 LLM)
  ├─ 单工具 + 只读 → 快速通道 (0-1 次 LLM)
  ├─ 单工具 + 高风险 → pending_action (0 次 LLM)
  ├─ 多工具 / 复杂推理 → ReAct 循环 (N 次 LLM)
  └─ 根据场景选择执行路径
```

### 2.3 核心数据流

```text
用户请求
  ↓
[L0: 规则守卫]
  RulesGuard.guard()
    → 空输入 / 无意义 → 拒绝
    → 确认续接 → 确认流程
    → 通过 → 下一层
  ↓
[L1: 场景直达]
  SceneDirectDispatcher.dispatch()
    → 问候 / 能力清单 → 模板回复
    → 不匹配 → 下一层
  ↓
[记忆增强]
  SessionMemorySystemTool.plan_and_execute()
  MemoryManager.enrich_request()
    → 补充用户画像、历史上下文
  ↓
[L2: 意图路由]
  RuleBasedRouter.match()
    → 恢复上一轮缺槽位场景
  ModelIntentRouter.route()
    → 模型语义分类 → ModelRouteOutput
  RoutePolicy.apply()
    → 能力策略映射 → RouteDecision
  ↓
[L3: 执行层分发]
  missing_slots? → ask_user
  pending_action? → 等待确认
  !should_call_tool? → 直接回答
  FastPathRunner.run() → 单工具快速通道
  _pending_from_route() → 高风险转确认
  _tool_call_from_route() → 单工具直接执行
  _run_react_steps() → ReAct 循环
  ↓
结构化响应 (answer + route_decision + tool_trace + evidence)
```

---

## 三、核心模块设计

### 3.1 L0/L1: 前置短路层

#### 模块职责
- 过滤无效请求，减少无意义的 LLM 调用
- 处理高确定性的固定场景
- 处理确认流程的续接

#### 关键组件

**RulesGuard** (`src/xinyidai_agent/router/rules_guard.py`)

```python
def guard(self, request: AgentRequest, context: SessionContext) -> Optional[GuardResult]:
    # 1. 空输入检查
    if not request.query or request.query.strip() == "":
        return GuardResult(action="reject", reason="空输入")
    
    # 2. 无意义输入检查
    if _is_meaningless(request.query):
        return GuardResult(action="ask_clarification")
    
    # 3. 确认续接检查
    if context.pending_action and _is_confirmation(request.query):
        return GuardResult(action="continue_confirmation")
    
    # 4. 通过守卫
    return None
```

**设计考量**：
- **为什么要规则守卫？** 空输入、"嗯"、"哦"等无意义输入调用 LLM 是浪费
- **为什么要确认续接？** 用户说"确认"、"好的"时，应该继续上一轮的确认流程，而不是重新路由
- **为什么返回 Optional？** None 表示通过守卫，继续下一层；非 None 表示短路返回

**SceneDirectDispatcher** (`src/xinyidai_agent/router/scene_dispatcher.py`)

```python
def dispatch(self, request: AgentRequest) -> Optional[DirectResponse]:
    # 1. 问候场景
    if _is_greeting(request.query):
        return DirectResponse(template="greeting")
    
    # 2. 能力清单场景
    if _is_capability_inquiry(request.query):
        return DirectResponse(template="capability_list")
    
    # 3. 自我介绍场景
    if _is_self_introduction(request.query):
        return DirectResponse(template="self_intro")
    
    # 4. 感谢场景
    if _is_thanks(request.query):
        return DirectResponse(template="thanks")
    
    # 5. 不匹配任何场景
    return None
```

**设计考量**：
- **为什么要场景直达？** 问候、能力清单等固定场景，模板回复比 LLM 生成更快、更稳定、成本更低
- **为什么不用 LLM 判断场景？** 这些场景特征明显，规则匹配足够准确，不需要 LLM
- **为什么返回 Optional？** None 表示不匹配任何场景，继续下一层路由

#### 模块交互
```text
ReactRuntime._pre_route()
       ↓
RulesGuard.guard() → GuardResult / None
       ↓
SceneDirectDispatcher.dispatch() → DirectResponse / None
       ↓
如果短路 → 直接返回
如果通过 → 进入 L2 路由
```

---

### 3.2 L2: 意图路由层

#### 模块职责
- 理解用户意图，分类到标准意图
- 映射到后端能力策略
- 确定工具白名单、必填槽位、风险等级

#### 关键组件

**ControlledIntentRouter** (`src/xinyidai_agent/router/service.py`)

```python
def route(self, request: AgentRequest, context: SessionContext) -> RouteDecision:
    # 1. 规则路由（恢复上下文）
    rule_result = RuleBasedRouter.match(request, context)
    if rule_result:
        return RoutePolicy.apply(rule_result)
    
    # 2. 模型路由（语义分类）
    if self.model:
        model_output = ModelIntentRouter.route(request, context)
    else:
        model_output = _default_knowledge_route()
    
    # 3. 能力策略收口
    return RoutePolicy.apply(model_output, request, context)
```

**RuleBasedRouter** (`src/xinyidai_agent/router/rule_router.py`)

```python
def match(self, request: AgentRequest, context: SessionContext) -> Optional[RouteOutput]:
    # 1. 恢复上一轮缺槽位场景
    if context.last_route and context.last_route.missing_slots:
        if _is_slot_filling_response(request.query):
            return RouteOutput(
                route=context.last_route.route,
                confidence=1.0,
                reason="slot_filling_continuation"
            )
    
    # 2. 低信息输入识别
    if _is_low_information(request.query):
        return RouteOutput(
            route="clarification_needed",
            confidence=1.0
        )
    
    return None
```

**设计考量**：
- **为什么需要规则路由？** 上一轮缺槽位，用户补充槽位时，不需要重新调用 LLM 分类
- **为什么要恢复上下文？** 节省 LLM 调用，提高响应速度，保持对话连贯性
- **为什么优先级高于模型路由？** 规则路由确定性高，且成本为 0

**ModelIntentRouter** (`src/xinyidai_agent/router/model_router.py`)

```python
def route(self, request: AgentRequest, context: SessionContext) -> ModelRouteOutput:
    # 1. 构造路由 prompt
    prompt = _build_route_prompt(request, context)
    
    # 2. 调用 LLM
    raw = self.model.complete(prompt)
    
    # 3. 解析 JSON
    payload = _parse_json(raw)
    if not payload:
        payload = _fix_json(raw)  # 尝试修复一次
    if not payload:
        return _unknown_route()  # 降级
    
    # 4. 结构化校验
    payload = _normalize_model_payload(payload)
    model_output = ModelRouteOutput.model_validate(payload)
    
    return model_output
```

**ModelRouteOutput 协议**：

```python
class ModelRouteOutput(BaseModel):
    scene: str                    # 场景分类
    standard_intent: str          # 标准意图
    confidence: float             # 置信度
    user_slots: Dict[str, Any]    # 用户提供的槽位
    
    # 模型不允许输出以下字段（由后端策略决定）
    # allowed_tools: List[str]
    # risk_level: str
    # confirmation_required: bool
```

**设计考量**：
- **为什么模型不能输出执行权限？** 权限控制必须由后端策略决定，防止模型越权
- **为什么要置信度？** 低置信度的路由应该转追问，而不是强行执行
- **为什么要 user_slots？** 模型可以从用户输入中提取槽位值，减少追问次数

**RoutePolicy** (`src/xinyidai_agent/router/policy.py`)

```python
def apply(self, model_output: ModelRouteOutput, request: AgentRequest, 
          context: SessionContext) -> RouteDecision:
    # 1. 置信度检查
    if model_output.confidence < self.confidence_threshold:
        return _ask_user_to_clarify()
    
    # 2. 能力策略解析
    capability = CapabilityResolver.resolve(
        scene=model_output.scene,
        standard_intent=model_output.standard_intent
    )
    if not capability:
        return _ask_user_to_clarify()
    
    # 3. 槽位合并（metadata + user_slots）
    all_slots = {**_extract_metadata_slots(context), **model_output.user_slots}
    
    # 4. 必填槽位检查
    missing_slots = _check_required_slots(capability.required_slots, all_slots)
    
    # 5. 生成最终决策
    return RouteDecision(
        capability_id=capability.id,
        route=model_output.standard_intent,
        confidence=model_output.confidence,
        required_slots=capability.required_slots,
        missing_slots=missing_slots,
        allowed_tools=capability.allowed_tools,
        allowed_tool_categories=capability.allowed_tool_categories,
        risk_level=capability.risk_level,
        confirmation_required=RiskPolicy.requires_confirmation(capability),
        should_call_tool=(len(missing_slots) == 0 and capability.allowed_tools),
        max_react_steps=capability.max_react_steps,
        requires_evidence=capability.requires_evidence,
    )
```

**设计考量**：
- **为什么要分离 ModelRouteOutput 和 RouteDecision？** 前者是模型理解，后者是执行决策，职责分离
- **为什么要 CapabilityResolver？** 标准意图到能力策略的映射由配置管理，便于调整和扩展
- **为什么要 metadata_slots？** 用户身份、企业信息等元数据不需要用户提供，从会话上下文获取

#### 模块交互
```text
RuleBasedRouter.match() → RouteOutput / None
       ↓
ModelIntentRouter.route() → ModelRouteOutput
       ↓
RoutePolicy.apply() → RouteDecision
       ↓
ReactRuntime (消费路由决策)
```

---

### 3.3 L3: 执行层分发

#### 模块职责
- 根据路由决策选择执行路径
- 简单场景走快速通道，复杂场景走 ReAct
- 高风险操作转确认流程

#### 关键组件

**ReactRuntime.run()** (`src/xinyidai_agent/runtime/react_loop.py`)

```python
def run(self, request: AgentRequest, context: SessionContext) -> AgentResponse:
    # 1. 前置短路
    pre_route_result = self._pre_route(request, context)
    if pre_route_result:
        return pre_route_result
    
    # 2. 记忆增强
    self._enrich_with_memory(request, context)
    
    # 3. 意图路由
    route = self._route(request, context)
    
    # 4. 执行层分发
    # 4.1 缺槽位 → 直接追问
    if route.missing_slots:
        return self._ask_for_slots(route.missing_slots)
    
    # 4.2 有 pending_action → 等待确认
    if context.pending_action:
        return self._wait_for_confirmation(context.pending_action)
    
    # 4.3 不应该调用工具 → 直接回答
    if not route.should_call_tool:
        return self._direct_answer(request, route)
    
    # 4.4 尝试快速通道
    fast_result = FastPathRunner.run(request, route, context)
    if fast_result:
        return fast_result
    
    # 4.5 高风险操作 → 生成 pending_action
    pending = self._pending_from_route(request, route, context)
    if pending:
        return self._build_confirmation_response(pending)
    
    # 4.6 单工具直接执行
    tool_call = self._tool_call_from_route(request, route, context)
    if tool_call:
        result = ToolRegistry.execute(tool_call, context)
        return self._build_tool_response(result)
    
    # 4.7 ReAct 循环
    return self._run_react_steps(request, route, context)
```

**设计考量**：
- **为什么要分这么多分支？** 不同场景的最优执行路径不同，分层处理提高效率
- **为什么缺槽位直接追问？** 避免进入 ReAct 浪费 token，快速收集信息
- **为什么快速通道优先于 ReAct？** 单工具场景不需要模型规划，直接执行更快
- **为什么高风险操作要转确认？** 状态变更必须经过用户确认，不能由系统自动执行

**FastPathRunner** (`src/xinyidai_agent/runtime/fast_path.py`)

```python
def run(self, request: AgentRequest, route: RouteDecision, 
        context: SessionContext) -> Optional[AgentResponse]:
    # 1. 只支持单工具场景
    if len(route.allowed_tools) != 1:
        return None
    
    tool_name = route.allowed_tools[0]
    tool_spec = ToolRegistry.spec_for_name(tool_name)
    
    # 2. 只支持只读工具
    if tool_spec.requires_confirmation:
        return None
    
    # 3. 参数必须齐全
    arguments = _extract_arguments(request, route, tool_spec)
    if not _all_required_params_present(arguments, tool_spec):
        return None
    
    # 4. 直接执行工具
    tool_call = ToolCall(name=tool_name, arguments=arguments)
    result = ToolRegistry.execute(tool_call, context)
    
    # 5. 构造响应
    return self._build_fast_path_response(result, route)
```

**设计考量**：
- **为什么只支持单工具？** 多工具需要模型规划执行顺序，不适合快速通道
- **为什么只支持只读工具？** 高风险工具需要确认流程，不能直接执行
- **为什么参数必须齐全？** 缺参数需要追问或让模型推理，不适合快速通道
- **为什么返回 Optional？** None 表示不适合快速通道，回退到其他执行路径

**_tool_call_from_route()** (`src/xinyidai_agent/runtime/react_loop.py`)

```python
def _tool_call_from_route(self, request: AgentRequest, route: RouteDecision,
                          context: SessionContext) -> Optional[ToolCall]:
    # 1. 获取候选工具
    candidate_tools = route.allowed_tools or route.capability.allowed_tools
    
    # 2. 遍历候选工具，尝试构造调用
    for tool_name in candidate_tools:
        spec = ToolRegistry.spec_for_name(tool_name)
        
        # 3. 从请求、路由、会话中提取参数
        arguments = _arguments_for_spec(request, route, context, spec)
        
        # 4. 检查必填参数
        if _has_all_required_params(arguments, spec):
            return ToolCall(name=tool_name, arguments=arguments)
    
    # 5. 没有工具可以直接构造
    return None
```

**设计考量**：
- **为什么不用模型选工具？** 路由已经确定了工具白名单，后端可以直接构造调用
- **为什么要遍历候选工具？** 可能有多个工具都能满足需求，选第一个参数齐全的
- **为什么返回 Optional？** None 表示参数不齐全，需要进入 ReAct 让模型处理

#### 模块交互
```text
ReactRuntime.run()
       ↓
_pre_route() → 短路 / 继续
       ↓
_route() → RouteDecision
       ↓
missing_slots? → ask_user
pending_action? → wait_confirmation
!should_call_tool? → direct_answer
       ↓
FastPathRunner.run() → 成功 / None
       ↓
_pending_from_route() → pending_action / None
       ↓
_tool_call_from_route() → ToolCall / None
       ↓
_run_react_steps() → ReAct 循环
```

---

### 3.4 ReAct 主循环

#### 模块职责
- 多步推理和工具编排
- 处理复杂场景和不确定性
- 循环终止条件控制

#### 关键组件

**_run_react_steps()** (`src/xinyidai_agent/runtime/react_loop.py`)

```python
def _run_react_steps(self, request: AgentRequest, route: RouteDecision,
                     context: SessionContext) -> AgentResponse:
    max_steps = route.capability.max_react_steps
    invalid_hint = None
    
    for step in range(1, max_steps + 1):
        # 1. 构造 observation
        observation = ReactObservationBuilder.build(
            request=request,
            route=route,
            context=context,
            history=self.history,
            invalid_hint=invalid_hint,
        )
        
        # 2. 推理下一步动作
        outcome = ReactStepEngine.step(observation)
        
        # 3. 处理结果
        if outcome.kind == "invalid_action":
            invalid_hint = outcome.invalid_hint
            continue  # 下一轮让模型修正
        
        if outcome.pending_action_required:
            pending = self._build_pending_action(outcome)
            return self._build_confirmation_response(pending)
        
        if outcome.kind == "call_tool_executed":
            self._save_tool_result(outcome.tool_result)
            if outcome.tool_result.terminal:
                return self._finish_with_tool_result(outcome.tool_result)
            continue  # 继续下一步
        
        if outcome.kind == "terminal":
            return self._finish_terminal(outcome)
    
    # 4. 超过最大步数
    return self._finish_max_steps()
```

**设计考量**：
- **为什么要 max_steps？** 防止模型无限循环，确保系统可控
- **为什么 invalid_action 不直接失败？** 给模型修正机会，提高成功率
- **为什么 tool_result.terminal 可以结束？** 某些工具（如查询）执行完就可以回答，不需要继续推理
- **为什么要保存 tool_result？** 下一轮 observation 需要历史工具结果作为上下文

**ReactObservationBuilder** (`src/xinyidai_agent/runtime/observation.py`)

```python
def build(self, request: AgentRequest, route: RouteDecision,
          context: SessionContext, history: List[ReactStep],
          invalid_hint: Optional[str]) -> ReactObservation:
    return ReactObservation(
        user_query=request.query,
        capability=route.capability,
        allowed_tools=route.allowed_tools,
        tool_history=_format_tool_history(history),
        invalid_hint=invalid_hint,
        context_info=_format_context(context),
    )
```

**设计考量**：
- **为什么要单独的 ObservationBuilder？** 封装复杂的上下文构造逻辑，便于测试和维护
- **为什么要 invalid_hint？** 告诉模型上一步哪里错了，引导修正
- **为什么要 tool_history？** 模型需要知道已经调用了哪些工具，避免重复或遗漏

**ReactStepEngine.step()** (`src/xinyidai_agent/runtime/react_engine.py`)

```python
def step(self, observation: ReactObservation) -> ReactStepOutcome:
    # 1. 推理下一步动作
    decision = self._reason(observation)
    
    # 2. 结构化校验
    action = ReactStepDecision.model_validate(decision)
    
    # 3. 动作分发
    if action.type == "call_tool":
        return self._handle_call_tool(action, observation)
    elif action.type == "answer":
        return self._validate_answer(action, observation)
    elif action.type == "ask_user":
        return self._validate_ask_user(action)
    elif action.type == "handoff":
        return self._validate_handoff(action)
```

**设计考量**：
- **为什么每步都要校验？** 模型输出不可信，每步都要确保符合协议
- **为什么要分发到不同的处理函数？** 每种动作的校验逻辑不同，分离关注点

#### 循环终止条件

```text
1. terminal 动作（answer / ask_user / handoff）
   → 模型认为可以结束，后端校验通过

2. tool_result.terminal = True
   → 工具执行完成，可以直接回答

3. pending_action_required
   → 需要用户确认，暂停循环

4. max_steps 达到
   → 强制终止，防止无限循环

5. invalid_action 连续失败
   → 模型无法修正，转人工
```

#### 模块交互
```text
_run_react_steps()
       ↓
ReactObservationBuilder.build() → ReactObservation
       ↓
ReactStepEngine.step() → ReactStepOutcome
       ↓
_handle_call_tool() → ToolRegistry.execute()
       ↓
保存 tool_result → 下一轮 observation
       ↓
循环或终止
```

---

## 四、关键设计决策与权衡

### 4.1 为什么要四层处理模型？

**决策**：L0 规则守卫 → L1 场景直达 → L2 意图路由 → L3 执行层

**权衡**：
- ✅ 优势：按确定性分层，简单请求不调用 LLM，节省成本和延迟
- ✅ 优势：每层职责清晰，便于维护和优化
- ✅ 优势：可以针对不同层使用不同的模型（L2 用小模型，L3 用大模型）
- ❌ 劣势：增加了系统复杂度，需要仔细设计层与层的边界
- ❌ 劣势：某些请求可能在多层之间反复判断

**适用场景**：有大量简单请求、需要优化成本和延迟的生产环境

### 4.2 为什么要快速通道？

**决策**：单工具 + 只读 + 参数齐全 → 直接执行，不进入 ReAct

**权衡**：
- ✅ 优势：额度查询、状态查询等简单场景响应更快
- ✅ 优势：减少 ReAct 循环的 token 消耗
- ✅ 优势：降低系统复杂度，简单场景不需要复杂推理
- ❌ 劣势：需要准确判断哪些场景适合快速通道
- ❌ 劣势：快速通道失败需要回退到 ReAct，增加了分支

**适用场景**：有大量单工具查询场景的系统

### 4.3 为什么要分离路由和执行？

**决策**：路由只做意图分类，执行层根据路由决策选择执行路径

**权衡**：
- ✅ 优势：路由模型可以用更小更快的模型（如 GPT-3.5）
- ✅ 优势：路由失败不影响执行层的稳定性
- ✅ 优势：可以针对路由和执行分别优化
- ❌ 劣势：路由和执行的边界需要仔细设计
- ❌ 劣势：某些信息需要在路由和执行之间传递

**适用场景**：需要精细化成本控制、有明确意图分类需求的系统

### 4.4 为什么要能力策略统一收口？

**决策**：所有执行权限由后端 CapabilityPolicy 决定，模型不能自创权限

**权衡**：
- ✅ 优势：权限控制集中管理，便于审计和调整
- ✅ 优势：模型不能越权，安全性更高
- ✅ 优势：新增能力只需配置，不需要修改代码
- ❌ 劣势：需要维护能力策略配置
- ❌ 劣势：策略配置错误会影响所有相关请求

**适用场景**：金融、医疗等高风险领域，权限控制要求严格

---

## 五、可扩展性设计

### 5.1 新增短路规则

**扩展点**：RulesGuard / SceneDirectDispatcher

```python
# 在 RulesGuard 中添加新规则
def guard(self, request, context):
    # 现有规则
    ...
    
    # 新增规则
    if _is_new_pattern(request.query):
        return GuardResult(action="new_action")
    
    return None
```

**无需修改**：路由逻辑、执行逻辑

### 5.2 新增路由意图

**扩展点**：CapabilityPolicy 配置

```yaml
# 在配置文件中添加新意图
- capability_id: new_capability
  scene: new_scene
  standard_intent: new_intent
  allowed_tools: [tool_a, tool_b]
  required_slots: [slot_1]
  risk_level: low
  max_react_steps: 3
```

**无需修改**：路由代码、执行代码

### 5.3 新增执行路径

**扩展点**：ReactRuntime.run()

```python
def run(self, request, context):
    # 现有分发逻辑
    ...
    
    # 新增执行路径
    if _should_use_new_path(route):
        return self._new_execution_path(request, route, context)
    
    # 原有逻辑
    ...
```

**设计模式**：策略模式，每种执行路径独立实现

### 5.4 新增 ReAct 动作

**扩展点**：ReactAction + ReactStepEngine

```python
# 1. 扩展协议
class ReactAction(BaseModel):
    type: Literal["call_tool", "answer", "ask_user", "handoff", "new_action"]

# 2. 添加处理逻辑
def step(self, observation):
    ...
    elif action.type == "new_action":
        return self._handle_new_action(action, observation)
```

**注意**：需要同步更新 prompt、校验逻辑、测试用例

---

## 六、性能优化实践

### 6.1 成本优化

**优化 1：分层模型选择**
- L0/L1：无 LLM 调用
- L2 路由：使用小模型（GPT-3.5 / Claude Haiku）
- L3 ReAct：使用大模型（GPT-4 / Claude Opus）
- 成本降低 60-70%

**优化 2：路由缓存**
- 相同请求的路由结果缓存 5 分钟
- 考虑会话状态的变化
- 缓存命中率 30-40%

**优化 3：快速通道优先**
- 单工具场景不进入 ReAct
- 减少 50% 的 ReAct 调用

### 6.2 延迟优化

**优化 1：并行调用**
- 路由和记忆增强并行执行
- 延迟降低 20-30%

**优化 2：流式响应**
- ReAct 每步结果流式返回
- 用户感知延迟降低 40-50%

**优化 3：预加载**
- 高频工具的 Schema 预加载
- 减少工具注册查询时间

### 6.3 准确率优化

**优化 1：Few-shot 示例**
- 路由 prompt 添加典型示例
- 路由准确率提升 10-15%

**优化 2：上下文增强**
- 记忆系统补充用户画像
- 意图识别准确率提升 15-20%

**优化 3：反馈循环**
- 收集路由错误案例
- 定期更新 prompt 和示例

---

## 七、生产实践经验

### 7.1 监控指标

**L0/L1 层**：
- 短路率（目标 20-30%）
- 各场景的命中率
- 误拦截率（应该路由但被短路）

**L2 层**：
- 路由准确率（目标 > 90%）
- 各意图的分布
- 低置信度比例（目标 < 10%）
- 路由延迟（目标 < 500ms）

**L3 层**：
- 快速通道命中率（目标 30-40%）
- ReAct 平均步数（目标 2-3 步）
- max_steps 触发率（目标 < 5%）
- 端到端延迟（目标 < 3s）

### 7.2 常见问题与解决

**问题 1：短路误拦截**
- 现象：应该路由的请求被 L0/L1 短路
- 解决：收集误拦截案例，优化规则匹配逻辑

**问题 2：路由准确率低**
- 现象：意图分类错误，导致工具白名单不对
- 解决：添加 few-shot 示例，优化 prompt，使用更大的路由模型

**问题 3：快速通道失败率高**
- 现象：参数不齐全，回退到 ReAct
- 解决：优化参数提取逻辑，或调整快速通道的判断条件

**问题 4：ReAct 不收敛**
- 现象：达到 max_steps 仍未结束
- 解决：优化 observation 构造，添加循环检测，调整 max_steps

**问题 5：执行路径选择错误**
- 现象：应该走快速通道的走了 ReAct，或反之
- 解决：优化分发逻辑，添加更多判断条件

### 7.3 A/B 测试经验

**测试 1：快速通道 vs 全 ReAct**
- 结果：快速通道延迟降低 60%，准确率持平
- 结论：单工具场景优先快速通道

**测试 2：小模型路由 vs 大模型路由**
- 结果：小模型成本降低 70%，准确率下降 5%
- 结论：可接受的权衡，采用小模型路由

**测试 3：记忆增强 vs 无记忆**
- 结果：记忆增强准确率提升 15%，延迟增加 10%
- 结论：值得的权衡，保留记忆增强

---

## 八、面试讲述框架

### 8.1 开场（1-2 分钟）

"我们的 Agent 系统采用'分层路由 + 快速通道 + ReAct 循环'的设计模式。核心理念是**按确定性分层处理**，简单请求用规则和模板，复杂请求才进入 ReAct。这样可以在保持灵活性的同时，优化成本和延迟。"

### 8.2 架构层（2-3 分钟）

"系统分为四层：**L0 规则守卫**过滤无效请求，**L1 场景直达**处理固定场景，**L2 意图路由**做语义分类和能力映射，**L3 执行层**根据路由决策选择执行路径。前两层不调用 LLM，第三层调用一次，第四层根据复杂度选择快速通道或 ReAct。"

"数据流是这样的：用户请求先经过 L0/L1 短路，通过后进入 L2 路由，模型输出意图后由后端能力策略派生工具白名单和风险等级。然后到 L3 执行层，先判断是否缺槽位、是否需要确认、是否适合快速通道，最后才进入 ReAct 循环。"

### 8.3 细节层（3-5 分钟，根据面试官兴趣展开）

**如果问分层设计**：
"分层的核心是按确定性处理。L0/L1 处理确定性最高的请求，如空输入、问候、能力清单，用规则和模板，0 次 LLM 调用。L2 处理确定性中等的请求，用小模型做意图分类，1 次 LLM 调用。L3 处理确定性最低的请求，简单的走快速通道，复杂的走 ReAct，N 次 LLM 调用。这样可以让 80% 的请求在 L0-L2 完成，只有 20% 进入 ReAct。"

**如果问快速通道**：
"快速通道是针对单工具 + 只读 + 参数齐全的场景优化的。比如额度查询，路由已经确定了意图和工具，参数也从用户输入中提取了，就不需要进入 ReAct 让模型规划，直接构造工具调用执行。这样可以减少 50% 的 ReAct 调用，延迟降低 60%。"

**如果问能力策略**：
"能力策略是路由和执行之间的桥梁。路由模型只输出标准意图和置信度，不输出工具白名单和风险等级。后端通过 CapabilityResolver 把标准意图映射到能力策略，派生出工具白名单、必填槽位、风险等级、最大步数等执行参数。这样模型不能自创权限，所有权限都来自配置，便于管理和审计。"

**如果问 ReAct**：
"ReAct 是兜底的执行路径，处理需要多步推理的复杂场景。每步构造 observation，包含用户请求、能力策略、工具历史、无效动作提示等上下文，让模型推理下一步动作。模型只能输出 4 种动作，每种都有对应的校验逻辑。循环终止条件包括 terminal 动作、工具 terminal、pending_action、max_steps。"

### 8.4 权衡层（2-3 分钟）

"这个设计的核心权衡是**简单性 vs 效率**。如果所有请求都进入 ReAct，系统会很简单，但成本和延迟都很高。我们选择了分层处理，代价是增加了系统复杂度，但成本降低了 60-70%，延迟降低了 40-50%，在生产环境中是值得的。"

"另一个权衡是**灵活性 vs 可控性**。快速通道牺牲了一些灵活性，但换来了更快的响应和更低的成本。ReAct 保留了灵活性，可以处理各种复杂场景。两者结合，既有效率又有灵活性。"

### 8.5 扩展层（1-2 分钟）

"系统的扩展性主要体现在三个方面：新增短路规则只需在 RulesGuard 或 SceneDirectDispatcher 中添加逻辑；新增路由意图只需配置 CapabilityPolicy；新增执行路径可以在 ReactRuntime 中添加分支。核心的路由和 ReAct 逻辑不需要修改。"

### 8.6 收尾（1 分钟）

"总结一下，这个系统的核心是**分层处理**，按确定性选择执行路径；**能力策略收口**，模型不能自创权限；**快速通道优先**，简单场景不进入 ReAct；**ReAct 兜底**，处理复杂场景。这些设计让我们在保持 Agent 灵活性的同时，优化了成本、延迟和可控性。"

---

## 九、附录：核心代码路径

```text
L0/L1 短路层：
  src/xinyidai_agent/router/rules_guard.py        # RulesGuard
  src/xinyidai_agent/router/scene_dispatcher.py   # SceneDirectDispatcher

L2 路由层：
  src/xinyidai_agent/router/service.py            # ControlledIntentRouter
  src/xinyidai_agent/router/rule_router.py        # RuleBasedRouter
  src/xinyidai_agent/router/model_router.py       # ModelIntentRouter
  src/xinyidai_agent/router/policy.py             # RoutePolicy

L3 执行层：
  src/xinyidai_agent/runtime/react_loop.py        # ReactRuntime
  src/xinyidai_agent/runtime/fast_path.py         # FastPathRunner
  src/xinyidai_agent/runtime/react_engine.py      # ReactStepEngine
  src/xinyidai_agent/runtime/observation.py       # ReactObservationBuilder

统一入口：
  src/xinyidai_agent/runtime/__init__.py          # ControlledAgentLoop
```
