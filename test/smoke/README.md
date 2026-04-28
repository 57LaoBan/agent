# 冒烟测试

这个包用于验证 Agent 开发依赖的基础外部服务是否可用。

## 百炼模型 API 连通性

在 `D:\companyCode\agent` 目录下执行：

```powershell
python .\test\smoke\smoke_test.py
```

也可以直接从任意目录执行：

```powershell
python D:\companyCode\agent\test\smoke\smoke_test.py
```

脚本默认读取项目根目录下的 `config/.env`：

```text
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
```

`config/.env` 是本地私密配置，不要提交真实 API Key；`config/.env.example` 是模板。
