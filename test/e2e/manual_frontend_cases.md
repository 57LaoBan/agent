# 前后端联调人工验收用例

这些用例用于你直接启动前端页面后，在聊天框里逐条输入验证。

## 启动方式

```powershell
cd D:\companyCode\agent
$env:PYTHONPATH="src"
python -m uvicorn xinyidai_agent.api:create_app --factory --host 127.0.0.1 --port 8000
```

另开一个 PowerShell：

```powershell
cd D:\companyCode\agent\web
$env:VITE_AGENT_API_BASE_URL="http://127.0.0.1:8000"
$env:VITE_USE_MOCK_AGENT="false"
npm run dev -- --host 127.0.0.1 --port 5173
```

打开 `http://127.0.0.1:5173`。

## 用例矩阵

| ID | 前端输入 | 期望识别来源 | 期望场景 | 期望工具暴露 | 期望结果 |
| --- | --- | --- | --- | --- | --- |
| FE-001 | `杭州示例科技有限公司能贷多少钱？` | 规则命中，高置信度 | `DATA_QUERY` / `CREDIT_LIMIT_QUERY` | 只暴露 `data_query`，调用 `query_credit_amount` | 聊天回答包含 `50万元`，检测窗口工具结果包含 `credit_amount=50万元`、`data_time=2026-04-28` |
| FE-002 | `我要申请小微税贷` | 规则命中，高风险动作 | `LOAN_APPLY` / `CREATE_APPLICATION` | 暴露 `knowledge`、`application`、`authorization` 分类，但不直接执行创建 | 聊天区出现 `确认创建贷款申请` 卡片，状态为 `AUTH_REQUIRED`，等待用户确认 |
| FE-003 | `信易贷适合哪些企业？` | 模型识别或知识问答回退 | `KNOWLEDGE_QA` / `POLICY_OR_PRODUCT_QA` | 只暴露并调用 `rag_search` | 聊天回答基于检索证据生成，检测窗口工具栏显示 `rag_search` 和 `RAG_RESULT_READY` |
| FE-004 | `判断杭州示例科技有限公司是否符合融资准入，能不能获得平台贷款支持` | 模型识别，非关键词输入 | `DATA_QUERY` | 只暴露 `data_query`，调用 `query_credit_amount` | 聊天回答包含 `50万元`；若模型未稳定识别，该用例就是当前意图识别能力的真实风险点 |
| FE-005 | `那个东西能不能帮我弄一下？` | 模型识别，低置信度 | `UNKNOWN` / `LOW_CONFIDENCE` | 不暴露工具 | 聊天区应追问用户想查询政策、查询额度还是发起贷款申请，不应调用任何工具 |
| FE-006 | `小微税贷这个产品给我走一下创建流程` | 模型识别，高置信度但槽位缺失 | `LOAN_APPLY` | 不执行工具 | 应追问企业名称，不应创建申请 |

## 检查重点

- 聊天区只展示用户应该看到的回答、追问或确认卡片。
- 检测窗口展示路由、允许工具、工具结果、事件流，不混入聊天正文。
- 数值查询必须来自 `query_credit_amount`，不能由模型编造。
- 申请类动作必须先出现确认卡片，不能直接提交或创建业务状态。
- 低置信度和槽位缺失时，应该追问，不应该调用工具。
