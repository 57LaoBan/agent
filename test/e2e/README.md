# 端到端测试

这个包用于验证用户从提问到获得答案、证据、工具调用和诊断信息的完整流程。

## 覆盖目标

- 前端聊天框能否把用户消息发送到后端 Agent。
- 后端 Agent 能否真实调用模型生成回答。
- Agent 能否区分模型识别、低置信度追问和槽位缺失追问。
- 数值查询场景能否只暴露 `data_query` 分类下的工具，并调用 `query_credit_amount` 返回 mock 数值。
- 申请场景能否停在执行前确认，而不是直接创建业务状态。

## 样本数据

- `fixtures/e2e_cases.json`：后端和前端端到端样本。
- `fixtures/mock_tool_data.json`：端到端用例对应的 mock 数据口径。
- `manual_frontend_cases.md`：人工打开前端页面时逐条输入的验收清单。

## 运行方式

先确认 `config/.env` 中存在 `LLM_API_KEY` 或 `DASHSCOPE_API_KEY`，端到端测试默认要求真实模型调用。

后端端到端：

```powershell
$env:PYTHONPATH="src"
python .\test\e2e\run_backend_e2e.py --start-server
```

受控路由矩阵端到端：

```powershell
$env:PYTHONPATH="src"
python .\test\e2e\run_controlled_route_e2e.py
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
python .\test\e2e\run_backend_e2e.py --start-server --case model_credit_amount_high_confidence
node .\test\e2e\run_frontend_e2e.mjs --start-backend --start-frontend --case frontend_rule_credit_amount_live_sse
```

## 用例分层

- `backend_cases` 调真实 HTTP API 和真实模型，适合验证前后联调的真实效果。
- `controlled_route_cases` 使用确定性模型桩，适合稳定验证模型场景判断、低置信度和槽位缺失等边界。
- `frontend_cases` 用 Playwright 或人工页面输入验证聊天区和检测窗口是否符合预期。
