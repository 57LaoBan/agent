# 架构草案

## 主流程

```text
ChatRequest
  -> Runtime 受控 loop
  -> Retriever 检索证据
  -> Prompt Builder 组织上下文
  -> LLM Gateway 调用模型
  -> ChatResponse 返回答案、来源和诊断信息
```

## 模块边界

- `protocol` 只定义稳定协议模型。
- `config` 只负责配置读取，不创建业务对象。
- `llm` 只负责模型调用，不知道信易贷业务。
- `rag` 只负责检索结果和 trace，不生成最终回答。
- `runtime` 只负责把一次问答流程串起来。
- `api` 只负责 FastAPI 路由，不承载业务编排。

## 非目标

- 第一阶段不做多 Agent。
- 第一阶段不做自动 planner。
- 第一阶段不做模型自驱动无限工具循环。
