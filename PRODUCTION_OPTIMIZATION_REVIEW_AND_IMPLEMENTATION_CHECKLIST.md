# 信贷 Agent 生产级严格审核与优化实现清单

> 给 Codex/实现者使用。本文不是宽松建议，而是生产验收清单。任何未满足 P0/P1 项的实现，都不能视为完成 `CREDIT_AGENT_OPTIMIZATION_PLAN.md`。

**目标：** 将当前信贷 Agent 从“已有优化骨架”严格升级为可生产落地的 controlled workflow agent。

**核心原则：**

```text
模型只负责理解、判断、建议、提取槽位、生成话术、提出候选动作。
Runtime / Policy / Capability / ToolRegistry / StateReducer 才是业务执行与状态变更的裁决者。
RAG 只提供 evidence，不做业务决策。
所有高风险动作必须绑定 pending_action snapshot，确认后仍需二次校验。
所有事件、状态、工具调用、确认动作必须可审计、可回放。
```

---

## 0. 当前审核结论

当前实现不能按生产要求验收。

它已经完成了一些骨架：

- 新增 `RuntimeStepDecision` / `RuntimeStepType`。
- 扩展了 `PendingAction` snapshot 字段。
- 新增 `RuntimeStateReducer`。
- 在 `runtime.py` 中加入有限 step loop 和 runtime step 事件。
- `RoutePolicy` 初步增加了 `validate_tool_action()`。
- `SessionStateSnapshot` 增加了 flow/stage/recent_turns 等字段。
- 当前 contract tests 能通过。

但它没有严格完成生产目标：

- 没有完整 pending action 确认执行链路。
- 没有确认前 snapshot 强校验。
- 没有真正的 `last_tool_result` 驱动 step policy。
- 没有稳定的 read-only 多步自动推进能力。
- pending action 文案和行为硬编码为“创建贷款申请”。
- ToolRegistry 没有内聚统一 action validator。
- short-term memory 只是窗口截断，没有 transcript store 与摘要合并。
- 测试覆盖没有打到关键安全路径。
- 普通 git diff 被 CRLF 行尾污染，review 噪声极大。

---

## 1. P0 阻断项：必须先修，否则不能继续扩功能

### P0-1. 清理行尾污染，建立可审查 diff

**问题：**

普通 `git diff --stat` 显示 77 个文件、上万行增删，但 `--ignore-space-at-eol` 后真实逻辑变更只有约 10 个文件。说明大量文件被 CRLF/LF 行尾重写污染。

**风险：**

- code review 不可读。
- Codex 很容易借噪声掩盖偷懒实现。
- 后续 blame / diff / merge 都会被污染。

**实现要求：**

1. 新增或修正 `.gitattributes`：

```text
*.py text eol=lf
*.md text eol=lf
*.txt text eol=lf
*.json text eol=lf
*.toml text eol=lf
*.yaml text eol=lf
*.yml text eol=lf
*.ts text eol=lf
*.tsx text eol=lf
*.css text eol=lf
*.html text eol=lf
*.mjs text eol=lf
*.ps1 text eol=crlf
*.bat text eol=crlf
```

2. 只保留真实逻辑变更。
3. 提交前必须确认：

```bash
git diff --ignore-space-at-eol --stat
```

真实变更应聚焦在：

```text
src/xinyidai_agent/protocol.py
src/xinyidai_agent/runtime.py
src/xinyidai_agent/runtime_state.py
src/xinyidai_agent/router/policy.py
src/xinyidai_agent/tools/registry.py
src/xinyidai_agent/memory/session.py
src/xinyidai_agent/system_tools/session_memory.py
web/src/types.ts
web/src/App.tsx
test/contract/*.py
```

**验收：**

- `git diff --stat` 与 `git diff --ignore-space-at-eol --stat` 不应有数量级差异。
- 不允许无意义重写 README、docs、web package-lock、全量 CSS 等文件。
- `git diff --check` 不应出现大规模 trailing whitespace。

---

### P0-2. 实现 PendingActionValidator，禁止“只看 action_id”

**问题：**

当前 `RuntimeStateReducer.apply_confirmation()` 只校验 action_id。
当前 `RoutePolicy.validate_tool_action()` 只校验确认状态、pending_action 存在和 tool_name 一致。

这不满足生产要求，存在 TOCTOU 风险。

**必须新增文件：**

```text
src/xinyidai_agent/policies/action_validator.py
```

或如果不想新建目录，也可创建：

```text
src/xinyidai_agent/router/action_validator.py
```

**必须实现：**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from xinyidai_agent.protocol import PendingAction, RouteDecision, SessionStateSnapshot, ToolCall
from xinyidai_agent.runtime_state import build_pending_action_hash
from xinyidai_agent.tools.base import ToolSpec


@dataclass(frozen=True)
class ActionValidationResult:
    allowed: bool
    reason: str
    code: str
    tool_call: ToolCall | None = None
    details: dict[str, Any] = field(default_factory=dict)


class PendingActionValidator:
    def validate_pending_action_snapshot(
        self,
        *,
        state: SessionStateSnapshot,
        route: RouteDecision,
        action: PendingAction,
        tool_call: ToolCall,
        spec: ToolSpec,
        now: datetime | None = None,
    ) -> ActionValidationResult:
        """确认后的工具执行前校验 pending action snapshot 是否仍与当前状态一致。"""
        ...
```

**必须校验：**

```text
1. state.pending_action 存在。
2. action.action_id 与 state.pending_action.action_id 一致。
3. action.session_id 与 state.session_id 一致。
4. action.scene 与 route.scene 一致。
5. action.capability_id 与 route.capability_id 一致。
6. action.stage 与 state.current_stage 一致，或符合允许的确认阶段迁移规则。
7. action.tool_call.tool_name 与 tool_call.tool_name 一致。
8. action.tool_call.arguments 与 tool_call.arguments 一致，不能确认 A 参数后执行 B 参数。
9. action.risk_level 与 spec.risk_level 一致。
10. spec.risk_level != read_only 时必须已确认。
11. action.expires_at 未过期。
12. slot_snapshot 与当前 confirmed_slots/route.filled_slots/tool_call.arguments 无冲突。
13. precondition_hash 重新计算后必须一致。
```

**禁止：**

- 禁止只校验 action_id。
- 禁止只校验 tool_name。
- 禁止用户确认后重新让模型生成新的 tool_call 再执行。
- 禁止确认期间用户切换企业/产品后继续执行旧 action。

**测试文件：**

```text
test/contract/test_pending_action_validator.py
```

**必须覆盖：**

```text
- valid snapshot -> allowed
- wrong action_id -> blocked
- wrong session_id -> blocked
- expired action -> blocked
- changed company_name -> blocked
- changed product_name -> blocked
- changed tool arguments -> blocked
- changed capability_id -> blocked
- changed risk_level -> blocked
- missing pending_action -> blocked
- confirmed action A cannot execute tool B
```

---

### P0-3. 增加明确的确认 API / 确认入口

**问题：**

当前系统能返回 `confirmation_required`，但没有生产级确认入口。没有确认入口，就不存在完整“pending -> confirm -> validate -> execute”的闭环。

**必须实现二选一：**

#### 方案 A：单独确认接口

```text
POST /chat/confirm
```

请求体：

```python
class ConfirmActionRequest(BaseModel):
    session_id: str
    action_id: str
    confirmed: bool
    user_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

返回：

```python
ChatResponse
```

#### 方案 B：复用 /chat，但必须结构化识别 confirmation

`ChatRequest` 增加：

```python
confirmation: ConfirmationInput | None = None
```

不推荐只靠自然语言“确认/取消”判断，因为金融场景必须绑定 action_id。

**确认执行流程必须是：**

```text
load session_state
  -> find pending_action by action_id
  -> if cancelled: clear pending_action, save event, final answer
  -> if confirmed:
       validate pending_action snapshot
       validate route/capability/tool spec
       execute original pending_action.tool_call
       apply tool result reducer
       save transcript + events + session_state
       generate final answer
```

**验收：**

- 用户确认后执行的是 pending_action 里保存的原始 tool_call，而不是重新让模型规划。
- 用户取消后 pending_action 清空，不执行工具。
- 用户切换企业/产品后旧 pending_action 自动失效。
- 确认接口所有路径都有事件 trace。

---

### P0-4. 重构 runtime 为真正的 Controlled Step Policy

**问题：**

当前 runtime 虽然有 `for step in range(...)`，但本质仍然是：

```python
route = router.route(request)
for step:
    tool_call = _plan_tool(request, route, session_state)
```

它没有明确使用 `last_tool_result` 决定下一步。非 terminal 工具结果可能导致重复调用同一个工具。

**必须新增：**

```text
src/xinyidai_agent/runtime_policy.py
```

**必须模型：**

```python
class RuntimeStepContext(BaseModel):
    step_index: int
    request: ChatRequest
    session_state: SessionStateSnapshot
    route: RouteDecision
    last_tool_result: ToolResult | None = None
    last_sources: list[SourceDocument] = Field(default_factory=list)
    pending_action: PendingAction | None = None


class RuntimeStepPolicy:
    def decide_next_step(self, context: RuntimeStepContext) -> RuntimeStepDecision:
        ...
```

**step policy 必须硬编码安全规则，不得交给模型自由决定：**

```text
1. route.missing_slots 非空 -> ASK_USER。
2. route.confidence 低于阈值 -> ASK_USER 或 HANDOFF。
3. pending_action waiting -> WAIT_CONFIRMATION。
4. last_tool_result failed -> GENERATE_FINAL_ANSWER 或 HANDOFF。
5. last_tool_result evidence insufficient -> GENERATE_FINAL_ANSWER，不允许编造。
6. last_tool_result terminal -> GENERATE_FINAL_ANSWER。
7. read_only 且同 capability 且未超过 max_steps -> 允许 PROPOSE_TOOL / EXECUTE_TOOL。
8. risk_level != read_only -> PROPOSE_PENDING_ACTION。
9. 跨 capability 状态变化 -> HANDOFF 或 ASK_USER。
10. step_index 超限 -> STOP。
```

**runtime.py 必须从“内联大循环”改成：**

```python
for step_index in range(1, self._max_steps + 1):
    step_context = self._build_step_context(...)
    decision = self._step_policy.decide_next_step(step_context)
    yield runtime_step_decision

    if decision.step_type == "ASK_USER": ...
    if decision.step_type == "PROPOSE_PENDING_ACTION": ...
    if decision.step_type == "EXECUTE_TOOL": ...
    if decision.step_type == "GENERATE_FINAL_ANSWER": ...
    if decision.step_type == "HANDOFF": ...
    if decision.step_type == "STOP": ...
```

**禁止：**

- 禁止每个 step 重新调用同一个 `_plan_tool()` 但不使用 last_tool_result。
- 禁止非 terminal 工具无限重复。
- 禁止 read_only 工具结果自动跨到 state_create。

**测试文件：**

```text
test/contract/test_controlled_step_policy.py
test/contract/test_controlled_multistep_runtime.py
```

**必须覆盖：**

```text
- read_only tool terminal=True -> 执行一次后 final answer
- read_only tool terminal=False 且 next_step 允许 -> 继续下一只读工具
- 非 terminal 但没有 allowed_next_tools -> safe stop/final answer
- 同一工具重复执行被阻断
- max_steps 后 STOP
- state_create 在 step policy 中转 pending_action，不执行
- link_create 在 step policy 中转 pending_action，不执行
- final_submit 默认 handoff 或 strong confirmation，不自动执行
```

---

### P0-5. pending action 生成必须泛化，不能硬编码贷款申请

**问题：**

当前 `_build_pending_action()` 对所有确认类动作都生成“确认创建贷款申请”。

**必须修改：**

```text
src/xinyidai_agent/runtime.py
```

或拆出：

```text
src/xinyidai_agent/pending_actions.py
```

**建议实现：**

```python
class PendingActionBuilder:
    def build(
        self,
        *,
        request: ChatRequest,
        route: RouteDecision,
        state: SessionStateSnapshot,
        tool_call: ToolCall,
        spec: ToolSpec,
    ) -> PendingAction:
        ...
```

**文案和风险必须按 tool/spec/capability 派生：**

```text
create_application:
  title: 确认创建贷款申请草稿
  risk: state_create
  summary: 将为 {company_name} 创建 {product_name} 申请草稿，尚不会最终提交。

create_authorization_link:
  title: 确认生成企业授权链接
  risk: link_create
  summary: 将为 {company_name} 生成授权链接，客户点击后进入授权流程。

update_application:
  title: 确认修改申请信息
  risk: state_update
  summary: 将修改申请 {application_id} 的业务信息。

final_submit:
  title: 确认最终提交
  risk: final_submit
  summary: 该操作可能产生正式业务提交，建议二次确认或转人工。
```

**测试：**

```text
test/contract/test_pending_action_builder.py
```

**必须覆盖：**

```text
- create_application 文案正确
- create_authorization_link 文案正确
- state_update 文案正确
- final_submit 强风险提示正确
- unknown state-changing tool 使用安全兜底文案，不误导用户
```

---

### P0-6. ToolRegistry 必须接入统一 ActionValidator

**问题：**

现在 runtime 外面调用 `RoutePolicy.validate_tool_action()`，但 ToolRegistry 里没有完整 action validator。未来任何绕过 runtime 的调用都可能出问题。

**要求：**

`ToolRegistry.execute()` 必须支持传入：

```python
session_state: SessionStateSnapshot | None = None
pending_action_validated: bool = False
```

或更好：

```python
validation_context: ToolValidationContext
```

并在内部统一做：

```text
tool exists
route allowed tools/categories
capability risk consistency
slot input validation
pending action validation if required
execute
output validation
```

**建议新增：**

```python
class ToolValidationContext(BaseModel):
    route: RouteDecision
    session_state: SessionStateSnapshot
    confirmed_action: PendingAction | None = None
    require_pending_action_validation: bool = True
```

**验收：**

- 任何 `risk_level != read_only` 的工具，如果未带已验证 pending_action，不得执行。
- 即使 runtime 忘了校验，registry 也应 fail closed。
- registry 返回 blocked ToolResult，且带清晰 code/reason。

---

## 2. P1 生产必需项：完成后才能进入联调/演示

### P1-1. Session Memory 必须区分 transcript store 和 runtime state

**问题：**

当前 `recent_turns` 只是窗口数组，超出后直接截断，没有 transcript store，没有可回放审计链。

**必须新增：**

```text
src/xinyidai_agent/memory/store.py
```

**建议接口：**

```python
class TranscriptStore(Protocol):
    def append_event(self, session_id: str, turn_id: str, sequence: int, event: AgentEvent) -> None: ...
    def append_turn_summary(self, session_id: str, turn_id: str, summary: dict[str, Any]) -> None: ...
    def list_events(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]: ...
```

第一版可用 JSONL 文件或 SQLite。

**推荐路径：**

```text
.data/transcripts/{session_id}.jsonl
```

或：

```text
.data/xinyidai_agent.sqlite
```

**必须记录：**

```text
session_id
turn_id
sequence
event_type
visibility
payload
route_decision
tool_call
tool_result
pending_action
confirmation_event
final_answer
model_name
skill_version
capability_id
created_at
```

**验收：**

- 每次 `/chat` 和 `/chat/stream` 都能落完整 event trace。
- 可以按 session_id 回放。
- transcript 不进入默认 prompt，只用于审计/恢复/评测。

---

### P1-2. recent_turns 超窗后必须合入 short_summary

**问题：**

当前 `_append_recent_turn()` 是直接截断。

**必须实现：**

```text
src/xinyidai_agent/memory/summarizer.py
```

第一版可以规则摘要，不一定调用模型。

**规则：**

```text
- 保留最近 6~10 轮原文。
- 旧 turns 合入 short_summary。
- short_summary 只记录业务事实，不记录闲聊。
- pending_action、confirmed_slots、last_application_id、authorization_status 不得丢。
```

**禁止：**

- 禁止摘要覆盖结构化字段。
- 禁止摘要生成新的未确认业务事实。
- 禁止保存跨客户长期记忆。

**测试：**

```text
test/contract/test_short_term_memory_summary.py
```

---

### P1-3. 用户切换企业/产品必须使旧 pending_action 失效

**当前部分实现：**

`SessionMemorySystemTool` 在切换企业时会清理一些状态。

**不足：**

这只是 memory tool 语义识别路径，不足以作为生产硬约束。

**必须在 reducer/policy 层硬编码：**

```text
如果 confirmed_slots.company_name 改变：
  clear last_credit_amount
  clear last_application_id
  clear authorization_status
  clear pending_action
  confirmation_status = none
  append audit event: pending_action_invalidated

如果 product_name 改变：
  clear pending_action for product-bound actions
  clear selected_product_name related pending slots
```

**测试：**

```text
test/contract/test_pending_action_invalidation.py
```

**必须覆盖：**

```text
- 用户换企业后旧 create_application pending action 不可确认
- 用户换产品后旧 create_application pending action 不可确认
- 用户换企业后旧额度结果不可继续引用
```

---

### P1-4. 风险等级策略必须集中定义，不能散落 if 判断

**必须新增或集中：**

```text
src/xinyidai_agent/policies/risk_policy.py
```

**规则：**

```python
RISK_POLICY = {
    "read_only": {
        "auto_execute": True,
        "requires_confirmation": False,
        "requires_handoff": False,
    },
    "link_create": {
        "auto_execute": False,
        "requires_confirmation": True,
        "requires_handoff": False,
    },
    "state_create": {
        "auto_execute": False,
        "requires_confirmation": True,
        "requires_handoff": False,
    },
    "state_update": {
        "auto_execute": False,
        "requires_confirmation": True,
        "requires_handoff": True,
    },
    "final_submit": {
        "auto_execute": False,
        "requires_confirmation": True,
        "requires_handoff": True,
    },
}
```

**验收：**

- 代码里不应散落多处 `risk_level != "read_only"` 作为唯一判断。
- 所有风险判断统一走 risk policy。
- final_submit 默认不应自动执行。

---

### P1-5. RAG evidence 边界必须硬化

**背景：**

当前 RAG 仍是 `EmptyRetriever` 或 mock，不能假装真实知识库已经接入。

**必须保证：**

```text
- no sources -> 不能输出确定性政策/准入/产品结论
- sources 为空时 business_status = PARTIAL_DATA / EVIDENCE_NOT_FOUND
- answer generator 必须明确“未检索到依据”
- retrieval_trace 必须记录 retriever 类型和 index_version/mock 标识
```

**建议新增：**

```text
src/xinyidai_agent/evidence_policy.py
```

**规则：**

```python
class EvidencePolicy:
    def assess(route, sources, retrieval_trace) -> EvidenceDecision:
        ...
```

**测试：**

```text
test/contract/test_evidence_policy.py
```

**必须覆盖：**

```text
- KNOWLEDGE_QA 无 sources -> 不编造
- PRODUCT/RULE 查询无 sources -> 不给准入结论
- sources 有效 -> answer 必须包含 citation/source title
```

---

## 3. P2 质量与工程化要求

### P2-1. 事件 trace 必须完整且稳定

每个 turn 必须至少有：

```text
turn_started
session_loaded
route_started
route_decision
runtime_step_started
runtime_step_decision
可能的 tool_call_proposed
可能的 pending_action_validated
可能的 confirmation_required
可能的 tool_started
可能的 tool_result
可能的 state_changed/session_updated
final_answer
turn_finished
```

**要求：**

- 所有 event payload 必须可 JSON 序列化。
- event_type 必须在协议 Literal 中声明。
- 前端类型必须同步。
- diagnostic 和 user visibility 不能混淆。

---

### P2-2. Contract tests 必须从“字段存在”升级为“业务不变量”

当前测试太浅，必须新增不变量测试。

**必须新增测试文件：**

```text
test/contract/test_pending_action_validator.py
test/contract/test_confirmation_flow.py
test/contract/test_controlled_step_policy.py
test/contract/test_pending_action_invalidation.py
test/contract/test_evidence_policy.py
test/contract/test_transcript_store.py
test/contract/test_business_dialogue_replay.py
```

**测试原则：**

- 每个生产安全规则必须有失败用例。
- 不能只测 happy path。
- 每个 blocked path 都要验证 reason/code。
- 每个风险等级都要有测试。

---

### P2-3. 增加业务回放 fixture

**新增目录：**

```text
test/fixtures/business_dialogues/
```

**至少新增：**

```text
credit_amount_missing_company.json
credit_amount_success.json
loan_apply_missing_company.json
loan_apply_missing_product.json
loan_apply_pending_create.json
loan_apply_confirm_create_success.json
loan_apply_switch_company_invalidates_pending.json
authorization_link_pending.json
knowledge_qa_no_evidence.json
unknown_low_confidence.json
```

每个 fixture 包含：

```json
{
  "case_id": "loan_apply_pending_create",
  "turns": [
    {
      "user": "我要申请小微税贷，企业是杭州示例科技有限公司",
      "expect": {
        "stop_reason": "waiting_confirmation",
        "pending_action.risk_level": "state_create",
        "tool_not_executed": "create_application"
      }
    }
  ]
}
```

---

### P2-4. Codex 防偷懒验收规则

给 Codex 的实现要求必须写死：

```text
1. 不允许只让测试通过但绕过生产路径。
2. 不允许只增加字段不接入 runtime。
3. 不允许只写 happy path 测试。
4. 不允许 mock 掉核心 validator。
5. 不允许把安全规则写在 prompt/Skill 文档里就算完成，必须有 policy 代码硬约束。
6. 不允许用自然语言“确认”替代 action_id confirmation。
7. 不允许重新规划已确认的 tool_call。
8. 不允许让模型输出的 allowed_tools/risk_level/confirmation_required 成为可信来源。
9. 不允许 pending_action 只校验 action_id。
10. 不允许行尾污染式大 diff。
```

---

## 4. 推荐实施顺序

### Phase 0：仓库卫生

1. 修 `.gitattributes`。
2. 清理 CRLF/LF 噪声。
3. 删除或决定是否提交测试副作用文件：`.venv/`、`uv.lock`。
4. 确认真实 diff 可读。

验收：

```bash
git diff --stat
git diff --ignore-space-at-eol --stat
git diff --check
```

---

### Phase 1：Pending Action 安全闭环

1. 写 `test_pending_action_validator.py` 失败用例。
2. 实现 `PendingActionValidator`。
3. 实现 `PendingActionBuilder`。
4. 修改 `RoutePolicy` / `ToolRegistry` 接入 validator。
5. 新增 `/chat/confirm` 或结构化 confirmation 输入。
6. 跑测试。

验收：

```bash
UV_PROJECT_ENVIRONMENT=/tmp/xinyidai-agent-venv uv run python -m unittest \
  test.contract.test_pending_action_validator \
  test.contract.test_pending_action_contract \
  test.contract.test_confirmation_flow
```

---

### Phase 2：真正 Controlled Step Runtime

1. 新增 `runtime_policy.py`。
2. 新增 `RuntimeStepContext`。
3. 把 runtime.py 的 step loop 改为 `RuntimeStepPolicy.decide_next_step()` 驱动。
4. 加 `last_tool_result` / `last_sources` / `allowed_next_tools`。
5. 禁止重复执行同一工具。
6. 完成 max_steps safe stop。

验收：

```bash
UV_PROJECT_ENVIRONMENT=/tmp/xinyidai-agent-venv uv run python -m unittest \
  test.contract.test_controlled_step_policy \
  test.contract.test_controlled_multistep_runtime
```

---

### Phase 3：短期记忆与审计

1. 新增 transcript store。
2. 每个 event append 到 transcript。
3. recent_turns 超窗后合并到 short_summary。
4. 用户切换企业/产品时由 reducer 硬失效 pending action。
5. 增加 replay tests。

验收：

```bash
UV_PROJECT_ENVIRONMENT=/tmp/xinyidai-agent-venv uv run python -m unittest \
  test.contract.test_short_term_memory \
  test.contract.test_transcript_store \
  test.contract.test_pending_action_invalidation \
  test.contract.test_business_dialogue_replay
```

---

### Phase 4：Evidence / RAG 生产边界

1. 新增 `EvidencePolicy`。
2. 无 evidence 不允许确定性政策/准入回答。
3. retrieval_trace 明确 retriever 类型、index_version/mock 标识。
4. answer 必须基于 sources/citations。

验收：

```bash
UV_PROJECT_ENVIRONMENT=/tmp/xinyidai-agent-venv uv run python -m unittest \
  test.contract.test_evidence_policy
```

---

### Phase 5：全量回归

```bash
UV_PROJECT_ENVIRONMENT=/tmp/xinyidai-agent-venv uv run python -m unittest discover -s test -p "test_*.py"
```

预期：

```text
OK
```

同时检查：

```bash
git diff --check
git diff --ignore-space-at-eol --stat
git status --short
```

---

## 5. 最终生产验收标准

必须全部满足：

1. 同一用户 turn 内可以完成多个只读受控步骤。
2. 非 read_only 工具绝不自动执行。
3. state_create/link_create 必须返回 pending_action。
4. state_update/final_submit 默认强确认或 handoff。
5. pending_action 绑定完整 snapshot。
6. 确认执行前重新校验 snapshot。
7. 确认后执行的是原始 pending_action.tool_call，不是新规划的 tool_call。
8. 用户换企业/换产品后旧 pending_action 失效。
9. 用户取消后 pending_action 清空且不执行工具。
10. 工具调用必须经过 capability/policy/tool registry 三重边界。
11. 模型输出的 allowed_tools/risk_level/confirmation_required 不可信。
12. RAG 无证据时不编造政策或准入结论。
13. event trace 可完整重建本轮业务执行过程。
14. transcript 可落库、查询、回放。
15. session state 只保存当前会话短期业务状态，不做长期画像。
16. recent_turns 超窗后合入 short_summary，不丢关键业务事实。
17. 所有 P0/P1 场景有 contract tests。
18. 全量测试通过。
19. git diff 干净、可审查，没有行尾污染。
20. 前端能展示 route、step、tool、confirmation、final answer、diagnostic trace。

---

## 6. 给 Codex 的执行提示词建议

可以直接把下面这段给 Codex：

```text
你不是做功能 demo，而是做生产级安全补全。严格阅读：
1. CREDIT_AGENT_OPTIMIZATION_PLAN.md
2. PRODUCTION_OPTIMIZATION_REVIEW_AND_IMPLEMENTATION_CHECKLIST.md

按 Phase 0 -> Phase 5 顺序执行。
每一 phase 必须先写失败测试，再实现，再跑指定测试。
禁止跳过 P0。
禁止只写 happy path。
禁止只加字段不接 runtime。
禁止让模型输出成为权限来源。
禁止 pending_action 只校验 action_id。
禁止确认后重新规划 tool_call。
禁止行尾污染式大 diff。

每完成一个 phase，输出：
- 修改文件列表
- 新增测试列表
- 测试命令和结果
- 对应验收项逐条打勾
- 未完成项明确说明，不能说“基本完成”
```

---

## 7. 一句话结论

当前代码是“优化方案的第一版骨架”，不是生产实现。下一步不要继续堆功能，必须围绕 pending action 安全闭环、真正 controlled step policy、统一 action validator、短期记忆审计落库、RAG evidence 边界这五个方向补齐。否则测试即使通过，也不能上线。
