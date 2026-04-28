# AGENTS.md

## 当前环境

- Windows 11
- PowerShell
- Conda 管理 Python 环境

## 协作偏好

- 注释、文档、commit message 默认使用中文。
- 代码结构先行，业务实现分阶段推进。
- 优先保持单 Agent、RAG-first、可诊断的受控 loop。

## 结构约定

- `src/xinyidai_agent/protocol.py`：请求、响应、trace、证据来源等稳定协议模型。
- `src/xinyidai_agent/config.py`：环境变量和本地配置读取。
- `src/xinyidai_agent/llm.py`：大模型网关，先支持 OpenAI 兼容接口。
- `src/xinyidai_agent/rag.py`：检索接口和 trace 结构，不直接耦合模型。
- `src/xinyidai_agent/runtime.py`：单 Agent 受控 loop。
- `src/xinyidai_agent/api.py`：FastAPI HTTP 入口，只承载接口层。
- `test/smoke/`：冒烟测试，比如百炼 API 连通性。
- `test/contract/`：契约测试，保证前后端字段稳定。
- `test/e2e/`：端到端测试，覆盖完整业务场景。

## 边界

- 不把复杂 planner、多 agent、文件系统代理作为第一阶段底座。
- 不把本地 `.env`、生成报告、缓存目录提交到仓库。
- 新功能先明确契约，再写实现，再补测试。
