# 架构草案

## 主流程

```text
ChatRequest
  -> Runtime 受控循环
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
- `rag` 只负责检索结果和 trace，不生成最终回答。
- `tools` 负责把 RAG、mock 数据接口、申请动作等能力包装成 Agent 可调用工具，并维护工具分类、风险等级和工具描述。
- `runtime` 只负责把一次问答流程串起来，不直接依赖 retriever 或具体业务接口。
- `api` 只负责 FastAPI 路由，不承载业务编排。

## 工具分类

工具注册表按分类暴露工具，路由结果先决定本轮允许的分类，再由模型或规则在该分类下选择具体工具。

- `knowledge`：政策、产品、准入规则等知识检索，例如 `rag_search`。
- `data_query`：只读数值查询，例如 `query_credit_amount`。
- `application`：申请创建、申请状态变更等会产生业务状态的动作。
- `authorization`：授权链接、授权状态相关动作。
- `status`：申请进度、审批状态等状态查询。
- `utility`：辅助类工具。

注册表执行工具前会检查工具名和工具分类，当前路由未允许的工具会返回 `blocked`，避免模型越权调用。

## 事件流

后端同时保留两种接口：

- `POST /chat`：返回聚合后的 `ChatResponse`，便于测试和兼容普通调用。
- `POST /chat/stream`：返回 SSE 事件流，前端按 `visibility` 分流用户内容和诊断内容。

聊天区只展示 `visibility=user` 的事件，例如 `assistant_delta`、`final_answer`、`confirmation_required`；检测窗口展示 `visibility=diagnostic` 的事件，例如 `route_decision`、`tool_call_proposed`、`tool_result`、`state_changed`。

## 非目标

- 第一阶段不做多 Agent。
- 第一阶段不做自动 planner。
- 第一阶段不做模型自驱动无限工具循环。
