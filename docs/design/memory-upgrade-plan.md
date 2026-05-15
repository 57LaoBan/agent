# 记忆模块优化方案（codex 直接生成版）

> 目标：在不破坏现有 `MemoryManager` / `SessionStateSnapshot` / `ReactRuntime` 契约的前提下，引入长期记忆（LTM）、跨会话事件记忆（Episodic）、可控压缩与反思流水线。
> 所有任务粒度按文件 + 函数签名给出，codex 可逐文件生成。注释、提交信息使用中文。

---

## 0. 框架选型决策

### 候选框架对比

| 框架 | 核心能力 | 与本项目契合度 | 结论 |
|------|---------|--------------|------|
| **Mem0** (`mem0ai`) | LTM 自动抽取/冲突合并/向量召回；支持 pgvector + HuggingFace 本地 embedder + OpenAI 兼容 LLM | ★★★★ 高度契合 | **采用**：作为 LTM + Episodic 的核心引擎 |
| **LangGraph Store** (`langgraph.store.postgres.PostgresStore`) | KV 式 namespace/key 存储 + 语义搜索 | ★★☆ 中等 | **不采用**：需引入整个 LangGraph 依赖链，且 KV 模型不如 Mem0 的自动抽取 |
| **LangMem** (`langmem`) | LangGraph 生态的 LTM 插件 | ★★☆ 中等 | **不采用**：强绑 LangGraph，本项目是自研 ReAct loop |
| **Zep** | 独立 server，episodic + semantic 双索引 | ★★★ 较好 | **不采用**：需额外部署 Zep server，增加运维复杂度 |
| **Letta/MemGPT** | 完整 agent 框架含记忆文件系统 | ★☆☆ 低 | **不采用**：架构侵入性太强，与自研 runtime 冲突 |
| **tiktoken** | token 计数 | ★★★★★ | **采用**：prompt budget 估算 |

### 最终方案

```
┌─────────────────────────────────────────────────────────────┐
│  STM 层（保持现有自研）                                       │
│  SessionStateSnapshot + MemoryManager + reducer             │
│  + 新增 tiktoken 做 token budget 控制                        │
├─────────────────────────────────────────────────────────────┤
│  LTM + Episodic 层（Mem0 OSS 库）                            │
│  mem0ai + pgvector + huggingface embedder (BGE-M3)          │
│  + 百炼 OpenAI 兼容接口做 LLM 抽取                           │
├─────────────────────────────────────────────────────────────┤
│  压缩层（自研 reducer + 可选 LLM summarizer）                 │
│  结构化折叠（零 LLM 成本）+ 超阈值触发 LLM 摘要              │
├─────────────────────────────────────────────────────────────┤
│  反思层（自研 worker 消费 PG 队列，调 Mem0 写入）             │
│  turn_finished → enqueue → worker → mem0.add()              │
└─────────────────────────────────────────────────────────────┘
```

**为什么 Mem0 而不是全手搓：**
- Mem0 内置了「从对话自动抽取事实 → 冲突检测 → 合并/覆盖 → 向量化存储 → 混合检索」全链路，省掉 ~500 行手搓代码。
- 支持 pgvector 作为向量后端（复用现有 PG 实例），支持 HuggingFace 本地 embedder（复用 BGE-M3），支持 OpenAI 兼容 LLM（百炼 dashscope）。
- 以 Python 库形式嵌入，不需要额外部署服务。
- 自带 `user_id` / `metadata` 过滤，天然支持多租户。

**为什么 STM 和压缩仍然自研：**
- Mem0 不管 session state / slot / 滚动窗口 / 结构化折叠，这些是业务强相关的。
- 现有 `MemoryManager` + `RuntimeStateReducer` 已经稳定运行，没必要换。

### 新增依赖

```toml
# pyproject.toml dependencies 追加
"mem0ai>=0.1.0",
"tiktoken>=0.7.0",
```

---

## 0.5 范围与边界

- **保持单 Agent / RAG-first / 受控 loop 不变**。
- **Mem0 的 LLM 抽取走百炼 OpenAI 兼容接口**，与主 Agent 共用 API key。
- **Mem0 的向量存储复用现有 PG 实例**，但使用独立 collection（`mem0_memories`），与 RAG 文档表物理隔离。
- 所有写入路径**异步可降级**：反思线程失败不阻塞主回合。
- STM 压缩仍走自研 reducer，Mem0 只负责 LTM/Episodic。

## 1. 总体目录结构

```
src/xinyidai_agent/memory/
├── __init__.py                # 重新导出新类型
├── conversation.py            # 现有：ChatSessionRecord 等（不动）
├── postgres.py                # 现有：PG 会话与审计（不动）
├── session.py                 # 重构：注入 Mem0 + 压缩器
├── store.py                   # 现有：JsonlTranscriptStore（不动）
│
├── compression/
│   ├── __init__.py
│   ├── budget.py              # tiktoken token 预算估算
│   ├── summarizer.py          # LLM 摘要器（schema-bound，可选）
│   └── reducer.py             # 结构化折叠 reducer（无 LLM，从 session.py 迁出）
│
├── mem0_adapter/
│   ├── __init__.py
│   ├── config.py              # Mem0 配置工厂（读 env → 生成 mem0 config dict）
│   ├── client.py              # Mem0Client 封装（add / search / delete / get_all）
│   └── retriever.py           # MemoryRetriever：按 user_id + query 召回 LTM
│
├── reflection/
│   ├── __init__.py
│   ├── queue.py               # ReflectionQueue（PG NOTIFY 或内存队列）
│   └── worker.py              # ReflectionWorker（消费 turn → 调 Mem0 add）
│
└── injection/
    ├── __init__.py
    └── composer.py            # MemoryInjector：把 Mem0 召回 + STM 拼成 metadata
```

> 对比之前全手搓版：**砍掉了 `episodic/` 和 `longterm/` 两个子包**（共 ~10 个文件），全部由 Mem0 库内部处理。

## 2. Mem0 配置与适配层

### 2.1 `memory/mem0_adapter/config.py`

```python
"""从环境变量生成 Mem0 OSS 配置字典。"""
import os
from typing import Any


def build_mem0_config() -> dict[str, Any]:
    """读取环境变量，生成 mem0.Memory.from_config 所需的 config dict。

    复用项目已有的百炼 API key 和 PG 实例。
    """
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.getenv("LLM_MODEL", "qwen-plus"),
                "openai_base_url": os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                "api_key": os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY", ""),
                "temperature": 0.1,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": os.getenv("MEM0_EMBEDDER_MODEL", "BAAI/bge-m3"),
                "embedding_dims": int(os.getenv("MEM0_EMBEDDING_DIMS", "1024")),
            },
        },
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "connection_string": _pg_dsn(),
                "collection_name": os.getenv("MEM0_COLLECTION", "mem0_memories"),
                "embedding_model_dims": int(os.getenv("MEM0_EMBEDDING_DIMS", "1024")),
            },
        },
    }


def _pg_dsn() -> str:
    """复用项目已有的 PG DSN 配置。"""
    return (
        os.getenv("MEM0_PG_DSN")
        or os.getenv("SESSION_DB_DSN")
        or os.getenv("DATABASE_URL")
        or "postgresql://xinyidai:xinyidai123@localhost:5432/xinyidai"
    )
```

### 2.2 `memory/mem0_adapter/client.py`

```python
"""Mem0 客户端封装，提供类型安全的 add / search / delete 接口。"""
from __future__ import annotations

from typing import Any

from mem0 import Memory

from xinyidai_agent.memory.mem0_adapter.config import build_mem0_config


class Mem0Client:
    """对 mem0.Memory 的薄封装，统一异常处理和日志。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._memory = Memory.from_config(config or build_mem0_config())

    def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """写入记忆。Mem0 内部自动抽取事实、冲突合并、向量化。"""
        return self._memory.add(
            messages,
            user_id=user_id,
            metadata={**(metadata or {}), "session_id": session_id} if session_id else metadata,
        )

    def search(
        self,
        query: str,
        *,
        user_id: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """按语义检索用户记忆。"""
        return self._memory.search(query, user_id=user_id, limit=limit)

    def get_all(self, *, user_id: str) -> list[dict[str, Any]]:
        """获取用户全部活跃记忆。"""
        return self._memory.get_all(user_id=user_id)

    def delete_all(self, *, user_id: str) -> None:
        """删除用户全部记忆（显式遗忘）。"""
        self._memory.delete_all(user_id=user_id)
```

### 2.3 `memory/mem0_adapter/retriever.py`

```python
"""基于 Mem0 的记忆检索器，供 MemoryInjector 调用。"""
from __future__ import annotations

from typing import Any

from xinyidai_agent.memory.mem0_adapter.client import Mem0Client


class MemoryRetriever:
    """按 user_id + query 召回长期记忆，返回格式化文本列表。"""

    def __init__(self, client: Mem0Client, max_results: int = 8) -> None:
        self._client = client
        self._max_results = max_results

    def retrieve(self, *, user_id: str, query: str) -> list[dict[str, Any]]:
        """检索并返回 Mem0 记忆条目。"""
        return self._client.search(query, user_id=user_id, limit=self._max_results)

    def retrieve_text(self, *, user_id: str, query: str) -> str:
        """检索并拼成可注入 prompt 的纯文本。"""
        results = self.retrieve(user_id=user_id, query=query)
        if not results:
            return ""
        lines = [f"- {item.get('memory', item.get('text', ''))}" for item in results]
        return "用户历史记忆：\n" + "\n".join(lines)
```

## 3. 协议扩展

### `SessionStateSnapshot` 追加字段（兼容默认值）

```python
# protocol.py 中 SessionStateSnapshot 追加：
last_summary_token_count: int = 0
summary_version: int = 0
ltm_user_id: str | None = None  # 当前会话绑定的用户 ID，用于 Mem0 检索
```

> 不新增独立的 `EpisodeRecord` / `FactRecord` model——这些由 Mem0 内部管理，我们只消费其 search 返回的 dict。

## 4. 压缩模块（自研，Mem0 不覆盖此层）

### 4.1 `compression/budget.py`

```python
"""token 预算估算，基于 tiktoken。"""
import tiktoken


_ENC = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """估算文本 token 数。"""
    return len(_ENC.encode(text))


def estimate_state_tokens(state_summary: str, recent_turns: list[dict]) -> int:
    """估算 STM 部分占用的 token。"""
    total = estimate_tokens(state_summary)
    for turn in recent_turns:
        total += estimate_tokens(str(turn))
    return total
```

### 4.2 `compression/reducer.py`（从 session.py 迁出，无 LLM）

```python
"""结构化折叠 reducer，零 LLM 成本。"""

def fold_overflow_turns(turns: list[dict], limit: int) -> tuple[list[dict], str]:
    """保留最近 limit 轮，溢出部分折叠为业务事实摘要。"""
    ...  # 迁移现有 _append_recent_turn / _fold_turns 逻辑
```

### 4.3 `compression/summarizer.py`（可选 LLM 摘要，超阈值触发）

```python
"""受控 LLM 摘要器，仅在 token 超阈值时触发。"""
from pydantic import BaseModel, ConfigDict, Field


class SummaryDelta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    new_summary: str = Field(max_length=600)
    dropped_turn_count: int = 0


class SessionSummarizer:
    def __init__(self, model, max_summary_tokens: int = 350) -> None: ...
    def compress(self, *, previous_summary: str, overflow_turns: list[dict]) -> SummaryDelta: ...
```

## 5. 反思流水线（自研 worker + 调 Mem0 写入）

### 5.1 `reflection/queue.py`

```python
"""反思任务队列，基于 PG 表 + SELECT FOR UPDATE SKIP LOCKED。"""

class ReflectionQueue:
    def __init__(self, dsn: str) -> None: ...
    def enqueue(self, *, session_id: str, turn_id: str, payload: dict) -> str: ...
    def claim_next(self, *, worker_id: str, batch: int = 4) -> list[dict]: ...
    def mark_done(self, job_id: str) -> None: ...
    def mark_failed(self, job_id: str, error: str) -> None: ...
```

SQL 表（追加到 `ensure_schema`）：

```sql
CREATE TABLE IF NOT EXISTS agent_memory_reflection_jobs (
    job_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  TEXT NOT NULL,
    turn_id     TEXT NOT NULL,
    user_id     TEXT,
    payload     JSONB NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','running','done','failed')),
    attempt     INT NOT NULL DEFAULT 0,
    last_error  TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_reflection_status
    ON agent_memory_reflection_jobs(status, created_at);
```

### 5.2 `reflection/worker.py`

```python
"""反思 Worker：消费队列 → 调 Mem0 add → 标记完成。"""
from xinyidai_agent.memory.mem0_adapter.client import Mem0Client
from xinyidai_agent.memory.reflection.queue import ReflectionQueue


class ReflectionWorker:
    def __init__(self, *, queue: ReflectionQueue, mem0_client: Mem0Client) -> None: ...

    def run_once(self, worker_id: str = "default") -> int:
        """处理一批任务，返回处理数量。"""
        jobs = self._queue.claim_next(worker_id=worker_id)
        for job in jobs:
            messages = self._build_messages(job["payload"])
            user_id = job.get("user_id") or job["session_id"]
            try:
                self._mem0.add(messages, user_id=user_id, session_id=job["session_id"])
                self._queue.mark_done(job["job_id"])
            except Exception as exc:
                self._queue.mark_failed(job["job_id"], str(exc))
        return len(jobs)

    def run_forever(self, *, poll_interval_seconds: float = 1.0) -> None:
        """持续轮询处理。"""
        ...

    def _build_messages(self, payload: dict) -> list[dict[str, str]]:
        """把 turn payload 转成 Mem0 期望的 messages 格式。"""
        return [
            {"role": "user", "content": payload.get("user_message", "")},
            {"role": "assistant", "content": payload.get("final_answer", "")},
        ]
```

> 关键点：**Mem0 内部自动完成事实抽取、冲突合并、向量化**，worker 只需要把对话喂给 `mem0.add()`。

## 6. 注入模块

### `injection/composer.py`

```python
"""多层记忆注入器：STM + Mem0 LTM → request.metadata。"""
from xinyidai_agent.memory.compression.budget import estimate_tokens, estimate_state_tokens
from xinyidai_agent.memory.mem0_adapter.retriever import MemoryRetriever
from xinyidai_agent.protocol import ChatRequest, SessionStateSnapshot


class MemoryInjector:
    def __init__(
        self,
        *,
        retriever: MemoryRetriever,
        max_total_tokens: int = 1200,
        max_ltm_results: int = 8,
    ) -> None:
        self._retriever = retriever
        self._max_total_tokens = max_total_tokens
        self._max_ltm_results = max_ltm_results

    def compose_and_attach(
        self,
        request: ChatRequest,
        state: SessionStateSnapshot,
    ) -> ChatRequest:
        """召回 LTM 并注入 request.metadata，不改 user_message。"""
        user_id = state.ltm_user_id or state.session_id
        # 估算 STM 已占 token
        stm_tokens = estimate_state_tokens(state.short_summary, state.recent_turns)
        remaining = self._max_total_tokens - stm_tokens
        if remaining <= 100:
            return request  # 预算不足，跳过 LTM 注入

        # 调 Mem0 检索
        ltm_text = self._retriever.retrieve_text(
            user_id=user_id,
            query=request.user_message,
        )
        if not ltm_text:
            return request

        # 裁剪到 budget
        ltm_tokens = estimate_tokens(ltm_text)
        if ltm_tokens > remaining:
            # 简单截断策略：按行裁剪
            lines = ltm_text.split("\n")
            trimmed = []
            used = 0
            for line in lines:
                line_tokens = estimate_tokens(line)
                if used + line_tokens > remaining:
                    break
                trimmed.append(line)
                used += line_tokens
            ltm_text = "\n".join(trimmed)

        metadata = dict(request.metadata)
        metadata["_ltm_context"] = ltm_text
        return request.model_copy(update={"metadata": metadata})
```

注入位置：`MemoryManager.enrich_request` 末尾调用 `injector.compose_and_attach`。

## 7. MemoryManager 重构

### 新签名

```python
class MemoryManager:
    def __init__(
        self,
        store: SessionStore | None = None,
        transcript_store: TranscriptStore | None = None,
        *,
        max_tool_results: int = 5,
        max_recent_turns: int = 10,
        # --- 新增可选依赖 ---
        summarizer: SessionSummarizer | None = None,
        summarize_threshold_tokens: int = 1500,
        injector: MemoryInjector | None = None,
        reflection_queue: ReflectionQueue | None = None,
    ) -> None: ...
```

### 调用顺序变更

`enrich_request` 改为：

```
1. 注入现有 metadata（保持兼容）
2. 若 injector 存在 → 调 injector.compose_and_attach(request, state)
```

`update` 末尾追加：

```
3. 若估算 token 超 summarize_threshold_tokens → 触发 summarizer.compress
   - 用结果更新 short_summary、summary_version、last_summary_token_count
4. 若 reflection_queue 存在 → enqueue 一个 job
5. 走原有 save 路径
```

> 兼容：`summarizer / injector / reflection_queue` 全部允许为 None；为 None 时降级为现有行为。

### 公共方法新增

```python
def forget(self, *, user_id: str) -> None:
    """显式遗忘：调 Mem0 delete_all。"""
    if self._mem0_client:
        self._mem0_client.delete_all(user_id=user_id)

def export_memory(self, *, user_id: str) -> list[dict]:
    """导出用户全部记忆。"""
    if self._mem0_client:
        return self._mem0_client.get_all(user_id=user_id)
    return []
```

## 8. Runtime 接入点

### `runtime/react_loop.py`

- `ReactRuntime.run` 末尾在 `turn_finished` 之后，把 `(user_message, final_answer, route, tool_trace, state)` 打包成 `payload` 调 `memory.reflection_queue.enqueue`（若存在）。

### `runtime/__init__.py`

`ControlledAgentLoop.__init__` 增加可选依赖：

```python
def __init__(
    self,
    ...,
    memory_injector: MemoryInjector | None = None,
    reflection_queue: ReflectionQueue | None = None,
):
    self._memory = memory_manager or MemoryManager(
        injector=memory_injector,
        reflection_queue=reflection_queue,
    )
```

## 9. API 装配

`api.py` 中：

```python
from xinyidai_agent.memory.mem0_adapter.client import Mem0Client
from xinyidai_agent.memory.mem0_adapter.config import build_mem0_config
from xinyidai_agent.memory.mem0_adapter.retriever import MemoryRetriever
from xinyidai_agent.memory.injection.composer import MemoryInjector
from xinyidai_agent.memory.reflection.queue import ReflectionQueue

# --- 装配 Mem0 ---
mem0_enabled = _to_bool(os.getenv("MEMORY_LTM_ENABLED"), default=True)
mem0_client: Mem0Client | None = None
injector: MemoryInjector | None = None
reflection_queue: ReflectionQueue | None = None

if mem0_enabled and storage.backend == "postgres":
    mem0_client = Mem0Client(build_mem0_config())
    retriever = MemoryRetriever(mem0_client)
    injector = MemoryInjector(retriever=retriever)
    reflection_queue = ReflectionQueue(storage.dsn)

memory_manager = MemoryManager(
    session_store,
    injector=injector,
    reflection_queue=reflection_queue,
    summarizer=SessionSummarizer(model) if mem0_enabled else None,
)
```

`memory` 后端为 in-memory 时全部走 None 降级路径。

## 10. 测试矩阵

按既有目录约定补：

```
test/contract/test_memory_compression.py
    - 触发摘要的 token 阈值
    - SummaryDelta 拒绝越权字段
    - reducer 回退路径
test/contract/test_episodic_store.py
    - upsert / search 时间加权
    - soft_delete 用户隔离
test/contract/test_fact_store.py
    - 冲突写入触发 supersede
    - active 唯一索引
    - 显式遗忘
test/contract/test_memory_injector.py
    - token 预算
    - facts/episodes 排序
    - metadata 注入位置正确
test/contract/test_reflection_queue.py
    - claim_next 并发安全（SKIP LOCKED）
    - 失败重试上限
test/e2e/test_memory_e2e.py
    - 一次会话 → 反思 → 第二次会话能召回上轮事实
```

## 11. 配置项

新增 `config/.env`：

```
MEMORY_LTM_ENABLED=1
MEMORY_EPISODE_ENABLED=1
MEMORY_REFLECTION_ENABLED=1
MEMORY_MAX_RECENT_TURNS=10
MEMORY_SUMMARIZE_THRESHOLD_TOKENS=1500
MEMORY_INJECT_MAX_TOKENS=1200
MEMORY_INJECT_MAX_FACTS=8
MEMORY_INJECT_MAX_EPISODES=4
MEMORY_RECENCY_HALF_LIFE_DAYS=14
MEMORY_REFLECTION_POLL_SECONDS=1.0
MEMORY_REFLECTION_MAX_ATTEMPTS=3
```

`SessionStorageConfig` 增加上述字段读取。

## 12. 演进路线

1. **Phase A（本方案）**：落地 STM 压缩 + Episodic + LTM + Reflection Worker + Memory Injector，复用 PG/pgvector。
2. **Phase B**：把 reflection 抽到独立进程；引入 Redis 热缓存 STM。
3. **Phase C**：增加用户面板（查看 / 编辑 / 删除 LTM），暴露 `GET /api/memory`、`DELETE /api/memory/{fact_id}`。
4. **Phase D**：引入跨用户聚合的语义画像（脱敏后），用于召回辅助。

## 13. codex 任务拆分（建议按顺序生成）

1. 协议扩展：`protocol.py` 追加新 model、`SessionStateSnapshot` 新增字段。
2. SQL 迁移脚本与 `PostgresConversationStore.ensure_schema` 调用。
3. `compression/{budget,reducer,summarizer}.py` 三件套。
4. `episodic/{protocol,store,writer,retriever}.py`。
5. `longterm/{protocol,store,writer,retriever,forgetter}.py`。
6. `reflection/{queue,extractors,worker}.py`。
7. `injection/composer.py`。
8. `MemoryManager` 重构（保持向后兼容）。
9. `ReactRuntime` / `ControlledAgentLoop` 接入。
10. `api.py` 装配。
11. 配置项与 `SessionStorageConfig` 字段。
12. 单元 / 契约 / e2e 测试。

> 每个步骤请独立 commit，commit message 中文格式：`feat(memory): <动作>`，例如 `feat(memory): 新增 PgEpisodeStore 与基础 schema`。
