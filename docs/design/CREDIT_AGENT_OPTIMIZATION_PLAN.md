# 信贷 Agent 优化实施方案

> 交付给 Codex 实现使用。本文总结当前信贷 Agent 的优化方向，并给出 runtime 与短期记忆模块的升级方案。

**目标：** 在保持“安全、可控、可审计优先”的前提下，将当前单轮受控业务 Agent 升级为支持受控多步推进的企业级信贷经办 Agent。

**核心原则：** 模型只能理解、判断、建议、生成话术和候选动作；最终业务决策、工具授权、风险确认、状态变更必须由 Runtime / Policy / Capability / ToolRegistry 裁决。

**当前架构基线：**

- Runtime：`src/xinyidai_agent/runtime.py` 中的 `ControlledAgentLoop`
- 路由：`src/xinyidai_agent/router/`
- 能力目录：`src/xinyidai_agent/capabilities/`
- 工具注册：`src/xinyidai_agent/tools/registry.py`
- 会话记忆：`src/xinyidai_agent/memory/session.py`
- 系统记忆工具：`src/xinyidai_agent/system_tools/session_memory.py`
- Skill/SOP：`src/xinyidai_agent/skills/` 与 `src/xinyidai_agent/skills/documents/*.md`
- API/SSE：`src/xinyidai_agent/api.py`
- 前端事件类型：`web/src/types.ts`
- 合同测试：`test/contract/`

---

## 1. 优化方向摘要

当前信贷 Agent 的设计方向是正确的：

```text
用户自然语言
  -> 结构化意图识别
  -> RoutePolicy 收敛
  -> Capability 解析业务能力
  -> Slot/Session Memory 补齐上下文
  -> ToolRegistry 限制工具边界
  -> Risk/Confirmation 控制高风险动作
  -> Skill/SOP 约束模型话术和流程
  -> Event Trace 支撑审计
```

后续优化不应走开放式通用 ReAct，也不应让模型自由决定连续调用工具。推荐升级为：

```text
Controlled Multi-Step Runtime
= 受控多步业务状态机
+ LLM 局部判断/建议
+ Runtime/Policy 最终裁决
+ ToolRegistry 强执行边界
+ Session State 短期记忆
+ PendingAction 人类确认
+ Event Trace 可审计
```

也就是说：

- 不做通用 Agent 式自由规划。
- 不引入前台多 Agent 编排。
- 不让模型拥有工具授权权。
- 允许一个用户 turn 内执行多个“安全、受控、可验证”的 step。
- 状态变化、高风险动作、跨 capability 推进必须停下来等待用户确认。

---

## 2. Runtime 升级方案

### 2.1 当前问题

当前 `ControlledAgentLoop.run()` 更接近“一轮用户输入 -> 一轮 runtime -> 返回结果”的单步链路。

这对简单问答、额度查询、缺槽追问是足够的，但对真实信贷经办流程会有几个限制：

1. 工具返回后无法在同一 turn 内继续进行受控判断。
2. 查询型工具链不能自然串联，例如产品搜索 + 政策 RAG + 推荐解释。
3. 创建申请后无法由 runtime 明确判断下一阶段，例如授权阶段。
4. 多轮流程 stage 不够显式，容易只靠 slots 判断当前进度。
5. pending action 的确认与执行应更强绑定具体 action snapshot。

### 2.2 目标形态

将 `ControlledAgentLoop` 升级为“外层用户回合 + 内层受控 step loop”。

伪流程：

```python
for step_index in range(max_steps):
    step_context = build_step_context(request, session_state, route, last_tool_result)

    decision = runtime_policy.decide_next_step(step_context)

    if decision.type == "ASK_USER":
        return ask_user(decision.question)

    if decision.type == "FINAL_ANSWER":
        return final_answer(decision.answer)

    if decision.type == "PROPOSE_PENDING_ACTION":
        pending_action = build_pending_action(decision, session_state)
        save_pending_action(pending_action)
        return confirmation_required(pending_action)

    if decision.type == "EXECUTE_TOOL":
        checked = policy.validate_tool_action(decision.tool_call, route, session_state)
        if not checked.allowed:
            return blocked_or_ask_user(checked.reason)

        tool_result = tool_registry.execute(request, route, checked.tool_call)
        session_state = state_reducer.apply(session_state, tool_result)
        continue

    if decision.type == "HANDOFF":
        return handoff_to_human(decision.reason)

return safe_stop_due_to_max_steps()
```

### 2.3 Step 类型建议

新增一个结构化 step decision 类型，建议放在 `src/xinyidai_agent/protocol.py`。

建议枚举：

```python
RuntimeStepType = Literal[
    "ROUTE",
    "ASK_USER",
    "ANSWER_WITHOUT_TOOL",
    "PROPOSE_TOOL",
    "EXECUTE_TOOL",
    "PROPOSE_PENDING_ACTION",
    "WAIT_CONFIRMATION",
    "GENERATE_FINAL_ANSWER",
    "HANDOFF",
    "STOP",
]
```

建议新增数据模型：

```python
class RuntimeStepDecision(BaseModel):
    step_type: RuntimeStepType
    reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_call: ToolCall | None = None
    pending_action: PendingAction | None = None
    answer: str | None = None
    question: str | None = None
    missing_slots: list[str] = Field(default_factory=list)
    handoff_reason: str | None = None
```

注意：模型可以生成候选 `RuntimeStepDecision`，但 Runtime 必须再校验。

### 2.4 允许自动连续执行的动作

只允许低风险、只读、同一业务能力内的动作自动连续执行。

可以自动继续：

- `read_only` 风险等级。
- RAG / 政策查询 / 产品查询 / 授信额度查询 / 申请状态查询。
- 工具结果只用于回答用户，不改变业务状态。
- 同一个 capability 内的 evidence collection。

必须中断等待用户的情况：

- 缺少必要槽位。
- 低置信度。
- 工具失败或证据不足。
- 需要创建、修改、提交、授权等状态变化。
- `confirmation_required = True`。
- 从 read-only capability 跳转到 state_create / state_update / final_submit。
- 用户表达模糊、反悔、换企业、换产品。
- 超过 `max_steps`。

### 2.5 风险等级建议

保留并强化当前风险分层：

```text
read_only      只读查询，可在受控 step loop 内自动执行
link_create    生成授权链接等，需要确认
state_create   创建申请草稿等，需要确认
state_update   修改业务状态，需要确认或转人工
final_submit   最终提交/签约/授权确认，强确认，建议转人工或二次确认
```

规则：

- `read_only` 可以自动执行。
- `link_create`、`state_create` 必须 pending action。
- `state_update`、`final_submit` 默认不要自动化，至少二次确认，必要时 handoff。

### 2.6 PendingAction 升级

当前 pending action 应升级为 action snapshot，避免 TOCTOU 风险。

建议在 `src/xinyidai_agent/protocol.py` 扩展 `PendingAction`：

```python
class PendingAction(BaseModel):
    action_id: str
    tool_call: ToolCall
    title: str
    summary: str
    risk_level: RiskLevel
    confirm_label: str = "确认"
    cancel_label: str = "取消"
    details: list[dict[str, str]] = Field(default_factory=list)

    # 新增建议字段
    session_id: str | None = None
    scene: RouteScene | None = None
    capability_id: str | None = None
    stage: str | None = None
    slot_snapshot: dict[str, Any] = Field(default_factory=dict)
    precondition_hash: str | None = None
    expires_at: str | None = None
    created_at: str | None = None
```

确认执行前必须校验：

- action_id 存在且未过期。
- session_id 匹配。
- 当前 stage 与 snapshot 匹配。
- 关键 slots 未发生冲突。
- capability_id 仍然允许该工具。
- risk_level 仍然需要且已经获得确认。

### 2.7 Flow Stage 升级

贷款申请不是只有 scene 和 slots，还需要明确 stage。

建议在 session state 中加入：

```python
active_flow: str | None
current_stage: str | None
completed_stages: list[str]
stage_history: list[dict[str, Any]]
```

贷款申请建议 stage：

```text
LOAN_APPLY.collect_company
LOAN_APPLY.select_product
LOAN_APPLY.confirm_application
LOAN_APPLY.application_draft_created
LOAN_APPLY.await_authorization
LOAN_APPLY.authorization_link_created
LOAN_APPLY.finished
```

授权流程建议 stage：

```text
AUTHORIZATION.collect_company
AUTHORIZATION.confirm_link_create
AUTHORIZATION.link_created
AUTHORIZATION.finished
```

额度查询建议 stage：

```text
DATA_QUERY.collect_company
DATA_QUERY.querying
DATA_QUERY.answered
```

### 2.8 State Reducer

建议新增文件：

`src/xinyidai_agent/runtime_state.py`

职责：

- 根据 route/tool_result/pending_action 更新 session_state。
- 将工具结果转换为结构化短期记忆。
- 推进 flow stage。
- 记录 stage_history。
- 处理用户换企业/换产品/取消确认。

建议接口：

```python
class RuntimeStateReducer:
    def apply_route(self, state: SessionState, route: RouteDecision) -> SessionState: ...
    def apply_tool_result(self, state: SessionState, route: RouteDecision, result: ToolResult) -> SessionState: ...
    def apply_pending_action(self, state: SessionState, action: PendingAction) -> SessionState: ...
    def apply_confirmation(self, state: SessionState, action_id: str, confirmed: bool) -> SessionState: ...
```

---

## 3. 短期记忆模块升级方案

### 3.1 记忆边界

该信贷 Agent 不需要主动长期记忆，不应像通用 Agent 一样维护用户画像或跨会话学习客户隐私。

需要的是：

1. 当前会话 transcript 落库。
2. 当前会话结构化 state。
3. 当前上下文窗口的短期摘要。
4. 可恢复、可审计、可追踪。

不要做：

- 主动长期记忆。
- 跨客户记忆。
- 模型自动保存用户偏好。
- 未授权地沉淀个人/企业敏感信息到长期 memory。

### 3.2 两类短期记忆

#### A. 原始聊天历史 Transcript

用于审计、恢复、回放、离线评测。

建议记录：

- session_id
- turn_id
- sequence
- role
- content
- event_type
- route_decision
- tool_call
- tool_result
- pending_action
- confirmation_event
- final_answer
- model_name
- skill_version
- capability_id
- created_at

#### B. 结构化 Session State

用于 runtime 决策，不要每轮让模型从长历史里自己回忆。

建议字段：

```python
class SessionState(BaseModel):
    session_id: str
    active_scene: str | None = None
    active_capability_id: str | None = None
    active_flow: str | None = None
    current_stage: str | None = None

    confirmed_slots: dict[str, Any] = Field(default_factory=dict)
    pending_slots: dict[str, Any] = Field(default_factory=dict)
    awaiting_slots: list[str] = Field(default_factory=list)

    selected_company_name: str | None = None
    selected_product_name: str | None = None
    last_credit_amount: dict[str, Any] | None = None
    last_application_id: str | None = None
    authorization_status: str | None = None

    pending_action: PendingAction | None = None
    confirmation_status: Literal["none", "waiting", "confirmed", "cancelled"] = "none"

    short_summary: str = ""
    recent_turns: list[dict[str, Any]] = Field(default_factory=list)
    completed_stages: list[str] = Field(default_factory=list)
    stage_history: list[dict[str, Any]] = Field(default_factory=list)
    turn_count: int = 0
    updated_at: str
```

### 3.3 MemoryManager 升级

当前 `src/xinyidai_agent/memory/session.py` 应承担以下职责：

- load(session_id)
- save(session_state)
- update(session_state, request, route, result)
- enrich_request(request, session_state)
- append_turn(...)
- get_recent_turns(session_id, limit)
- summarize_if_needed(session_state)

建议拆分：

```text
memory/session.py          SessionState 模型 + MemoryManager
memory/store.py            落库适配，可先 JSON/SQLite，后续可换数据库
memory/summarizer.py       短期摘要生成，可选
memory/reducer.py          如果不放 runtime_state.py，也可放这里
```

### 3.4 窗口短期记忆策略

每轮 prompt 不应塞完整历史，而应塞：

```text
1. 当前 user_message
2. 结构化 session_state 摘要
3. 最近 N 轮对话
4. 当前 scene 对应 skill
5. route/capability/tool specs
6. 必要时 short_summary
```

建议：

- 最近 6~10 轮保留原文。
- 更早历史落库，不默认进 prompt。
- 当 recent_turns 超出窗口时，用模型或规则生成 short_summary。
- short_summary 只描述业务状态，不保存无关闲聊。

short_summary 示例：

```text
客户正在为杭州示例科技有限公司办理小微税贷申请。已确认企业名称：杭州示例科技有限公司；已选择产品：小微税贷；申请草稿尚未创建。当前等待用户确认是否创建申请草稿。
```

### 3.5 Session 压缩/摘要规则

借鉴 Hermes 的短期记忆思路，但不需要通用压缩。

建议触发条件：

- recent_turns 超过 10 轮。
- prompt 估算超过模型上下文的 40%~50%。
- 工具结果过长。
- 用户从一个 scene 切换到另一个 scene。

压缩策略：

- 保留最近 N 轮原文。
- 保留 pending_action 原文和 snapshot。
- 保留所有 confirmed_slots。
- 保留 last tool result 的关键业务字段。
- 将更早对话写入 transcript store。
- 生成/更新 short_summary。

禁止压缩丢失：

- 企业名称。
- 产品名称。
- 金额。
- 申请编号。
- 授权链接状态。
- pending_action。
- 用户确认/取消记录。
- 工具返回的业务状态。

---

## 4. 具体实施任务

### Task 1: 增加 RuntimeStepDecision 协议模型

**目标：** 给受控多步 runtime 提供结构化 step 决策对象。

**文件：**

- 修改：`src/xinyidai_agent/protocol.py`
- 测试：`test/contract/test_runtime_step_contract.py`

**实现要点：**

- 新增 `RuntimeStepType`。
- 新增 `RuntimeStepDecision`。
- 保持现有 `ChatResponse` 兼容。

**验收：**

- Pydantic 模型可以序列化/反序列化。
- 非法 step_type 被拒绝。
- confidence 范围校验生效。

---

### Task 2: 扩展 PendingAction 为 action snapshot

**目标：** 让确认动作绑定具体业务状态，避免确认错配。

**文件：**

- 修改：`src/xinyidai_agent/protocol.py`
- 修改：`web/src/types.ts`
- 测试：`test/contract/test_pending_action_contract.py`

**实现要点：**

- 添加 `session_id`、`scene`、`capability_id`、`stage`、`slot_snapshot`、`precondition_hash`、`expires_at`、`created_at`。
- 前端类型同步。
- 保持旧字段兼容。

**验收：**

- pending action JSON 中包含 snapshot。
- 确认时可校验 action_id 与 snapshot。

---

### Task 3: 升级 SessionState 字段

**目标：** 显式记录 flow/stage 和短期业务上下文。

**文件：**

- 修改：`src/xinyidai_agent/memory/session.py`
- 修改：`web/src/types.ts`
- 测试：`test/contract/test_session_memory_contract.py`

**新增字段：**

- `active_flow`
- `current_stage`
- `completed_stages`
- `stage_history`
- `last_application_id`
- `authorization_status`
- `recent_turns`

**验收：**

- 旧 session state 加载时有默认值。
- `model_dump()` 输出前端可消费。
- session 更新不会丢失 confirmed_slots。

---

### Task 4: 新增 RuntimeStateReducer

**目标：** 把 session state 更新逻辑从 runtime 主流程中拆出，集中处理业务状态变化。

**文件：**

- 创建：`src/xinyidai_agent/runtime_state.py`
- 测试：`test/contract/test_runtime_state_reducer.py`

**接口建议：**

```python
class RuntimeStateReducer:
    def apply_route(self, state, route): ...
    def apply_tool_result(self, state, route, result): ...
    def apply_pending_action(self, state, action): ...
    def apply_confirmation(self, state, action_id, confirmed): ...
```

**验收案例：**

- DATA_QUERY 成功后写入 `last_credit_amount`。
- create_application 成功后写入 `last_application_id`，stage 进入 `application_draft_created`。
- pending authorization action 后 `confirmation_status = waiting`。
- 用户取消后 pending_action 清空。

---

### Task 5: 增加 Controlled Multi-Step Runtime Loop

**目标：** 将当前单步 runtime 改造成每个用户 turn 内最多执行 N 个受控 step。

**文件：**

- 修改：`src/xinyidai_agent/runtime.py`
- 测试：`test/contract/test_controlled_multistep_runtime.py`

**实现要点：**

- 保留 `ControlledAgentLoop.answer()` 和 `run()` 对外接口。
- 新增内部 `_run_steps(...)` 或 `_next_step(...)`。
- 默认 `max_steps = 3`，可通过构造参数传入。
- 每个 step 都 emit `state_changed` 或新增 `runtime_step` 诊断事件。
- 遇到 ask_user / pending_action / final_answer / handoff / max_steps 立即结束当前 turn。

**硬规则：**

- 只有 `read_only` 可自动执行。
- `confirmation_required` 必须返回 pending_action。
- 不允许模型绕过 route.allowed_tools。
- 不允许跨 capability 自动状态变更。

---

### Task 6: 工具执行前增加统一 Action Validator

**目标：** 所有工具调用前都经过同一套 policy 校验。

**文件：**

- 创建或修改：`src/xinyidai_agent/router/policy.py`
- 修改：`src/xinyidai_agent/tools/registry.py`
- 测试：`test/contract/test_action_validator.py`

**校验内容：**

- tool_name 在 `route.allowed_tools` 中。
- tool_category 在 `route.allowed_tool_categories` 中。
- required_slots 完整。
- slot 类型正确。
- risk_level 与 capability 一致。
- confirmation_required 时未确认不得执行。
- pending_action snapshot 与当前 session_state 一致。

---

### Task 7: 短期记忆 recent_turns 与 short_summary

**目标：** 支持客服型 Agent 的会话连续性，不引入长期记忆。

**文件：**

- 修改：`src/xinyidai_agent/memory/session.py`
- 可选创建：`src/xinyidai_agent/memory/summarizer.py`
- 测试：`test/contract/test_short_term_memory.py`

**实现要点：**

- 每轮结束 append recent_turn。
- recent_turns 超过阈值后，将旧 turns 合入 short_summary。
- pending_action、confirmed_slots、last_application_id 等关键业务状态不得被摘要覆盖或丢失。
- `enrich_request()` 注入结构化 state，而不是完整历史。

---

### Task 8: 前端事件与诊断面板同步

**目标：** 前端可以展示多 step runtime 的执行轨迹。

**文件：**

- 修改：`web/src/types.ts`
- 修改：`web/src/App.tsx`
- 修改：`web/src/agentClient.ts`

**建议新增事件：**

```text
runtime_step_started
runtime_step_decision
runtime_step_finished
pending_action_validated
handoff_required
```

如果不想扩展事件类型，也可继续复用：

- `state_changed`
- `route_decision`
- `tool_call_proposed`
- `confirmation_required`
- `tool_started`
- `tool_result`

但 payload 中应带 `step_index`。

---

### Task 9: 增加业务回放评测集

**目标：** 用真实客服场景验证路由、槽位、工具、确认、摘要是否稳定。

**文件：**

- 创建：`test/fixtures/business_dialogues/*.json`
- 创建：`test/contract/test_business_dialogue_replay.py`

**覆盖场景：**

- 额度查询。
- 缺企业名追问。
- 缺产品名追问。
- 查询产品后推荐。
- 创建申请前确认。
- 确认后创建申请。
- 创建申请后提示授权。
- 用户中途换企业。
- 用户取消 pending action。
- 低置信度转追问。
- 工具失败后安全返回。
- RAG 证据不足。

---

## 5. 推荐最终 Runtime 流程

```text
收到用户消息
  ↓
load session_state
  ↓
session memory tool / reducer 提取槽位、确认状态、短期上下文
  ↓
进入 controlled step loop，最多 max_steps
  ↓
[Step 1] route / resolve capability / apply policy
  ↓
缺槽？-> ask_user -> end turn
  ↓
低置信度？-> ask_user 或 handoff -> end turn
  ↓
需要高风险动作？-> pending_action -> end turn
  ↓
只读工具？-> validate -> execute -> update state -> continue
  ↓
工具结果足够？-> generate final answer -> end turn
  ↓
工具失败/证据不足？-> safe answer / handoff -> end turn
  ↓
超过 max_steps？-> safe stop -> end turn
  ↓
save transcript + session_state + events
```

---

## 6. Codex 实现注意事项

1. 不要把 runtime 改成开放式 ReAct。
2. 不要让模型直接决定最终工具执行权。
3. 不要让模型输出的 `allowed_tools`、`risk_level`、`confirmation_required` 成为可信来源；这些必须由 capability/policy 派生。
4. 不要引入跨会话长期记忆。
5. 不要把完整历史无限塞入 prompt。
6. 所有状态变化动作必须有 pending_action snapshot。
7. 所有工具调用必须经过 ToolRegistry 和 Action Validator。
8. 每个新增行为必须加 contract test。
9. 前端 SSE 事件要保持向后兼容。
10. 优先保证安全、可控、可审计，而不是自动化程度。

---

## 7. 建议测试命令

项目当前测试不是 Hermes 自身项目，优先按本项目现有方式运行。

建议先查看 `pyproject.toml` 后执行：

```bash
cd /mnt/d/Code/agent
python3 -m pytest test/contract -q
```

如项目使用虚拟环境，则先激活对应 venv。

---

## 8. 最终验收标准

实现完成后，应满足：

1. 用户同一 turn 内可以完成多个只读受控步骤。
2. 状态变化工具不会被自动执行，必须 pending_action。
3. pending_action 确认绑定 action snapshot。
4. session state 能记录 active_flow/current_stage。
5. 用户换企业/换产品时不会错误沿用旧 pending action。
6. 工具结果后 runtime 可以继续判断下一阶段，但必须受 policy 限制。
7. recent_turns 与 short_summary 能保证当前会话短期记忆。
8. transcript 与 event trace 足够审计。
9. 所有关键路径有 contract tests。
10. 前端能展示 route、step、tool、confirmation、final answer 的完整轨迹。

---

## 9. 一句话总结

这个信贷 Agent 的最佳升级方向不是通用 ReAct，也不是多 Agent 编排，而是“受控多步业务 Runtime”：模型负责理解和建议，Runtime/Policy/Capability 负责裁决，ToolRegistry 负责执行边界，SessionState 负责短期记忆，Event Trace 负责审计。
