# 架构草案

## 主流程

```text
ChatRequest
  -> Runtime 受控循环
  -> ControlledIntentRouter 规则优先、模型补充、策略收口
  -> RouteDecision 识别场景、允许工具分类和允许工具
  -> ToolCall 提出候选动作
  -> ToolRegistry 查找并执行工具
  -> PendingAction 拦截需确认动作
  -> ToolResult 回填工具结果
  -> LLM Gateway 基于工具结果生成回答
  -> AgentEvent 流式输出过程
  -> ChatResponse 聚合答案、来源、工具 trace 和诊断信息
```

## 模块边界

- `protocol` 只定义稳定协议模型。
- `config` 只负责配置读取，不创建业务对象。
- `llm` 只负责模型调用，不知道信易贷业务。
- `capabilities` 定义后端允许的业务能力池，能力负责派生标准意图、工具池、槽位、风险等级和确认策略。
- `router` 负责业务意图识别，采用规则优先、模型结构化识别、策略校验三层设计。
- `rag` 只负责检索结果和 trace，不生成最终回答。
- `tools` 负责把 RAG、mock 数据接口、申请动作等能力包装成 Agent 可调用工具，并维护工具分类、风险等级和工具描述。
- `runtime` 只负责把一次问答流程串起来，不直接依赖 retriever 或具体业务接口。
- `api` 只负责 FastAPI 路由，不承载业务编排。

## 生产级响应协议

对外响应保留 `answer`、`route_decision`、`tool_trace`、`events` 等现有字段，同时吸收旧版 Agent 中已经验证过的统一封装思想：

- `protocol_version`：协议版本，便于前端和回放兼容。
- `business_status`：本轮业务状态，例如 `RAG_RESULT_READY`、`CREDIT_AMOUNT_FOUND`、`AUTH_REQUIRED`。
- `ToolResultEnvelope`：工具结果统一外壳，包含 `success`、`status`、`code`、`message`、`data`、`next_step`、`actions`、`audit`。
- `EvidenceState`：回答收敛依赖的证据状态，说明证据是否可用、已有信号和缺失项。
- `ModelDecision`：本轮模型/规则决策摘要，说明是否收敛、下一步动作、置信度、追问信息和来源。
- `AgentAction`：面向前端的动作描述，支持打开链接、小程序、业务动作、联系客服，并携带风险、确认和审计信息。
- `PerformanceSummary`：本轮耗时摘要，供检测窗口和后续性能分析使用。

## 业务能力与工具权限

工具权限不再由模型直接决定，而是由后端能力策略派生。模型只提供 `scene`、`raw_intent`、槽位和置信度等语义信号，`RoutePolicy` 通过 `CapabilityResolver` 在预定义能力池中选择 `capability_id`，再生成标准 `intent`、`allowed_tools`、`risk_level` 和 `confirmation_required`。

```text
模型候选路由
  -> CapabilityResolver 选择后端能力
  -> CapabilityPolicy 派生工具池和风险策略
  -> ToolRegistry 执行前做工具名、分类、输入槽位和输出槽位校验
```

当前能力池示例：

- `knowledge.policy.read`：政策、产品、准入规则知识问答，允许 `rag_search`。
- `credit.limit.read`：授信额度只读查询，允许 `query_credit_amount`。
- `authorization.link.create`：生成企业授权链接，风险等级 `link_create`，需要确认。
- `application.draft.create`：创建贷款申请草稿，风险等级 `state_create`，需要确认。
- `application.status.read`：申请状态只读查询。
- `smalltalk.respond`：闲聊和助手能力说明，不开放业务工具。
- `unknown.clarify`：无法确认意图时追问。

## 工具分类

工具注册表按分类维护工具，能力策略先决定本轮允许的分类和具体工具，模型不能直接扩大工具池。

- `knowledge`：政策、产品、准入规则等知识检索，例如 `rag_search`。
- `data_query`：只读数值查询，例如 `query_credit_amount`。
- `application`：申请创建、申请状态变更等会产生业务状态的动作。
- `authorization`：授权链接、授权状态相关动作。
- `status`：申请进度、审批状态等状态查询。
- `utility`：辅助类工具。

注册表执行工具前会检查工具名和工具分类，当前路由未允许的工具会返回 `blocked`，避免模型越权调用。

工具还会声明输入槽位和输出槽位，并为每个槽位声明类型，例如 `query_credit_amount` 必须接收字符串类型的 `company_name`、`query`，并返回字符串类型的 `company_name`、`credit_amount`、`data_time`。注册表执行工具前校验输入槽位的字段名和类型，执行工具后校验输出槽位的字段名和类型，缺失或类型不匹配会被拦截或标记为失败，避免基于错误业务数据生成回答。

## 意图识别

意图识别不是让模型直接决定执行动作，而是先产出候选路由，再由系统策略裁决。

```text
用户输入
  -> RuleBasedRouter 处理申请、额度等强先验表达
  -> ModelIntentRouter 处理规则未覆盖的自由表达，要求模型只返回 JSON 语义信号
  -> RoutePolicy 检查置信度、解析 capability、校验必填槽位和风险等级
  -> RouteDecision 交给 runtime 继续规划工具或追问
```

当前策略约束：

- 低于置信度阈值的路由会转为 `UNKNOWN` 并进入追问。
- 必填槽位缺失时不执行工具，先进入 `CLARIFYING`。
- 模型原始意图保留在 `raw_intent`，标准意图由 capability 派生到 `intent`。
- 模型输出的 `allowed_tools`、`allowed_tool_categories`、`risk_level` 会被忽略，工具池由 capability 派生。
- 创建申请、授权链接、最终提交等状态类动作需要通过 `PendingAction` 做执行前确认。

## 事件流

后端同时保留两种接口：

- `POST /chat`：返回聚合后的 `ChatResponse`，便于测试和兼容普通调用。
- `POST /chat/stream`：返回 SSE 事件流，前端按 `visibility` 分流用户内容和诊断内容。

聊天区只展示 `visibility=user` 的事件，例如 `assistant_delta`、`final_answer`、`confirmation_required`；检测窗口展示 `visibility=diagnostic` 的事件，例如 `route_decision`、`tool_call_proposed`、`tool_result`、`state_changed`。

## 非目标

- 第一阶段不做多 Agent。
- 第一阶段不做自动 planner。
- 第一阶段不做模型自驱动无限工具循环。
