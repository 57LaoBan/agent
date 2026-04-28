# Agent 测试目录

测试按类型分包，后续所有 Agent 相关测试都放在这里。

## 目录约定

- `smoke/`：冒烟测试，验证外部依赖、模型 API、基础链路是否可用。
- `contract/`：契约测试，验证请求、响应、工具输出和前端诊断字段是否稳定。
- `e2e/`：端到端测试，验证完整用户场景和多步骤业务流程。

## 当前可运行测试

```powershell
python .\test\smoke\smoke_test.py
```
