# 信易贷聊天 Agent 重构仓库

这是新一版信易贷聊天 Agent 的独立开发目录，目标是把代码结构重新收束成一个可控、可测试、可诊断的单 Agent 问答系统。

## 当前目标

- 以单 Agent 受控 loop 为主线。
- 以 RAG 和业务规则为核心能力。
- 保留完整 trace，供右侧检测窗口展示。
- 避免一开始引入 planner、多 agent、文件系统代理等重型抽象。

## 目录结构

```text
src/xinyidai_agent/   Agent 主包
test/smoke/           冒烟测试
test/contract/        契约测试
test/e2e/             端到端测试
docs/                 设计和进度文档
config/               配置模板
scripts/              本地辅助脚本
```

## 运行百炼冒烟测试

```powershell
python .\test\smoke\smoke_test.py
```

## 运行契约测试

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s test -p "test_*.py"
```

## 第一阶段原则

先把稳定协议跑通，再逐步接入真实 RAG、规则引擎、前端检测窗口字段。
