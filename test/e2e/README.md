# 端到端测试

这个包用于验证用户从提问到获得答案、证据、工具调用和诊断信息的完整流程。

## 覆盖目标

- 前端聊天框能否把用户消息发送到后端 Agent。
- 后端 Agent 能否真实调用模型生成回答。
- Agent 在未接入专门意图识别模型前，能否通过当前规则正确判断场景。
- 数值查询场景能否只暴露 `data_query` 分类下的工具，并调用 `query_credit_amount` 返回 mock 数值。
- 申请场景能否停在执行前确认，而不是直接创建业务状态。

## 样本数据

- `fixtures/e2e_cases.json`：后端和前端端到端样本。

## 运行方式

先确认 `config/.env` 中存在 `LLM_API_KEY` 或 `DASHSCOPE_API_KEY`，端到端测试默认要求真实模型调用。

后端端到端：

```powershell
$env:PYTHONPATH="src"
python .\test\e2e\run_backend_e2e.py --start-server
```

前端浏览器端到端：

```powershell
cd .\web
npx playwright install chromium
cd ..
node .\test\e2e\run_frontend_e2e.mjs --start-backend --start-frontend
```

全部运行：

```powershell
.\test\e2e\run_all_e2e.ps1
```

只运行单个样本：

```powershell
python .\test\e2e\run_backend_e2e.py --start-server --case credit_amount_tool
node .\test\e2e\run_frontend_e2e.mjs --start-backend --start-frontend --case frontend_credit_amount_live_sse
```
