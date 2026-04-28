# 当前进度

## 已完成

- 初始化独立 Agent 目录：`D:\companyCode\agent`。
- 建立测试分层：`smoke`、`contract`、`e2e`。
- 跑通阿里云百炼 `qwen-plus` OpenAI 兼容接口冒烟测试。
- 建立第一版单 Agent 受控 loop 骨架。
- 将请求、响应、trace 结构统一收敛到 `protocol.py`，避免和契约测试目录混名。
- 加入 FastAPI HTTP 入口，接口层不承载业务编排。
- 新增 `web/` 前端工作台，左侧聊天区只展示用户可见内容，右侧检测窗口展示意图、工具、状态和原始事件。

## 当前设计取舍

- 主路径采用单 Agent 问答，不引入复杂 planner 或多 agent 协作。
- RAG、模型调用、响应契约、诊断 trace 分模块维护。
- 右侧检测窗口后续直接消费结构化 trace，而不是解析聊天文本。
- 前端按 `visibility` 分流同一条 Agent 事件流，避免把诊断过程污染到聊天框。

## 下一步

- 接入真实 RAG 检索接口。
- 定义前端检测窗口需要的 `retrieval_trace`、`citation_trace`、`rule_trace`。
- 增加政策问答、产品推荐、准入规则判断三个端到端场景。
- 将前端 mock 事件流替换为后端真实 SSE 事件流。
