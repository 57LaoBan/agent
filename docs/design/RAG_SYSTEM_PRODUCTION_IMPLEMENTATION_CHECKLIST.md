# RAG 系统生产级实现清单

> 给 Codex/实现者使用。本文是 RAG 系统从评测体系到生产部署的完整实现清单。

**目标：** 构建生产级 RAG 检索系统，包含评测体系、文档处理、向量检索、重排序、引用生成的完整流程。

**核心原则：**

```text
评测先行：先建立评测体系，再实现检索器，量化优化效果。
模型本地化：使用本地 BGE-M3 和 Reranker，避免外部依赖。
存储统一：PostgreSQL + pgvector 统一存储文本、向量、元数据。
检索可控：混合检索（Dense + Sparse）+ Rerank，可配置权重。
引用溯源：答案必须基于检索文档，带完整引用链路。
性能优化：批处理、缓存、异步并发。
可观测性：完整 trace，记录召回、重排、引用全过程。
```

---

## 0. 本地模型资源

**已准备模型**（位于 `C:\Users\阿猫\.cache\huggingface\hub`）：

1. **BGE-M3** (`models--BAAI--bge-m3`)
   - 用途：文档和查询的向量化（Embedding）
   - 维度：1024
   - 支持：Dense、Sparse、ColBERT 三种检索模式
   - 使用场景：
     - **离线**：批量构建向量数据库（不需要低延迟，优化吞吐量）
     - **在线**：用户查询向量化（必须低延迟 < 50ms，需要缓存）

2. **ms-marco-MiniLM-L-6-v2** (`models--cross-encoder--ms-marco-MiniLM-L-6-v2`)
   - 用途：检索结果重排序（Rerank）
   - 输入：(query, document) 对
   - 输出：相关度分数

**技术选型**：
- Embedding 框架：`sentence-transformers`
- 向量数据库：PostgreSQL + pgvector
- 向量索引：HNSW（高召回率、高性能）

**参考文档**：
- 文档分块：`docs/design/DOCUMENT_CHUNKING_BEST_PRACTICES.md`
- 向量化模块：`docs/design/EMBEDDING_BEST_PRACTICES.md`
- pgvector 存储：`docs/design/PGVECTOR_BEST_PRACTICES.md`
- 检索器：`docs/design/RETRIEVER_BEST_PRACTICES.md`
- 重排序器：`docs/design/RERANKER_BEST_PRACTICES.md`
- **指标体系**：`docs/design/RAG_METRICS_SYSTEM.md`

---

## 1. P0 阻断项：评测体系建设

> **核心原则**：评测先行，先建立评测体系，再实现检索器，量化优化效果。

**完整指标体系**：参见 `docs/design/RAG_METRICS_SYSTEM.md`

### 1.1 指标体系架构

```
业务指标（最重要）
  ├─ Answer Accuracy（答案准确率）> 0.85
  ├─ Task Success Rate（任务成功率）> 0.80
  ├─ Abstention Accuracy（拒答准确率）> 0.90
  └─ Hallucination Rate（幻觉率）< 0.05
    ↓
检索质量指标
  ├─ Recall@5（召回率）> 0.80
  ├─ MRR（平均倒数排名）> 0.70
  ├─ NDCG@5（排序质量）> 0.80
  └─ Precision@5（精确率）> 0.60
    ↓
性能指标
  ├─ Latency p95（延迟）< 1000ms
  ├─ QPS（吞吐量）> 10
  └─ Cache Hit Rate（缓存命中率）> 20%
    ↓
成本指标
  └─ Cost per Query（每次查询成本）< ¥0.01
```

**说明**：
- 业务指标是最终目标，检索指标是手段
- 各子模块有独立的评价指标（见下文）
- RAG 系统指标体系不影响子模块的评价指标

---

### 1.2 评测数据集

**数据集规模**：
- 开发集（Dev Set）：50 个问答对（快速迭代）
- 测试集（Test Set）：200 个问答对（最终评估）
- 无答案测试集：50 个无答案查询（评估拒答能力）

**数据集分层**：
- 按难度：简单 40%、中等 40%、困难 20%
- 按场景：产品问答 30%、准入规则 30%、授权流程 20%、风险标签 10%、其他 10%

**目录结构**：

```text
rag_corpus/
├── knowledge/                    # 知识库文档
│   ├── credit_products_and_flow.md
│   ├── authorization_and_data_scope.md
│   ├── risk_label_dictionary.md
│   └── synthetic_enterprises/
└── evaluation/                   # 评测数据集
    ├── dev_set.json             # 开发集（50 个）
    ├── test_set.json            # 测试集（200 个）
    └── no_answer_set.json       # 无答案测试集（50 个）
```

**数据格式**：

```json
{
  "id": "qa_001",
  "category": "产品问答",
  "difficulty": "easy",
  "question": "小微税贷的利率是多少？",
  "expected_answer": "小微税贷的年化利率为 4.5%",
  "relevant_chunk_ids": ["chunk_001", "chunk_002"],
  "relevant_doc_ids": ["doc_001"],
  "requires_multi_doc": false,
  "requires_reasoning": false,
  "has_answer": true
}
```

---

### 1.3 评测指标实现

**必须创建文件：**

```text
src/xinyidai_agent/evaluation/
├── __init__.py
├── retrieval_metrics.py      # 检索指标
├── answer_metrics.py          # 答案质量指标
└── evaluator.py               # 评测主流程
```

**核心指标**：参见 `docs/design/RAG_METRICS_SYSTEM.md`

---

## 2. P1 生产必需项：文档处理与分块

> **核心原则**：为不同文档类型提供专用分块器，保证语义完整性。

**参考文档**：`docs/design/DOCUMENT_CHUNKING_BEST_PRACTICES.md`

### 2.1 文档类型识别与路由

**必须创建文件：**

```text
src/xinyidai_agent/rag/chunkers/document_router.py
```

**核心功能**：
- 识别文档类型（.md、.pdf、.txt）
- 路由到对应的分块器
- 返回统一的 Chunk 对象

---

### 2.2 Markdown 结构化分块器

**必须创建文件：**

```text
src/xinyidai_agent/rag/chunkers/markdown_chunker.py
```

**核心策略**：标题层级 + 句子边界

**参考实现**：`docs/design/DOCUMENT_CHUNKING_BEST_PRACTICES.md` 第三章

**验收标准**：
- 能按标题层级分割
- 能生成标题路径（如 `/Introduction/Background/`）
- 分块大小 450-550 字符
- 语义完整性 > 90%

---

### 2.3 PDF 布局感知分块器

**必须创建文件：**

```text
src/xinyidai_agent/rag/chunkers/pdf_chunker.py
```

**核心策略**：布局分析 + 表格提取 + 文本分块

**技术选型**：Unstructured.io

**验收标准**：
- 能提取 PDF 文本、标题、表格
- 表格作为独立单元
- 自动过滤页眉、页脚

---

### 2.4 表格专用分块器

**必须创建文件：**

```text
src/xinyidai_agent/rag/chunkers/table_chunker.py
```

**核心策略**：表格序列化，保留行列关系

**验收标准**：
- 能序列化 HTML/Markdown 表格
- 保留列头和行的对应关系
- 输出格式便于 LLM 理解

---

### 2.5 纯文本分块器

**必须创建文件：**

```text
src/xinyidai_agent/rag/chunkers/text_chunker.py
```

**核心策略**：句子边界分块

---

### 2.6 分块器基类

**必须创建文件：**

```text
src/xinyidai_agent/rag/chunkers/base.py
```

```python
@dataclass
class Chunk:
    """文档分块。"""
    
    chunk_id: str
    doc_id: str
    content: str
    start_char: int
    end_char: int
    metadata: dict[str, Any]
```

---

## 3. P1 生产必需项：向量化与存储

### 3.1 向量化模块

**必须创建文件：**

```text
src/xinyidai_agent/rag/embedder.py
```

**使用场景**：

| 场景 | 时机 | 性能要求 | 优化目标 | 配置 |
|------|------|---------|---------|------|
| **离线构建** | 初始化知识库、新增文档 | 吞吐量 | documents/second | batch_size=128, GPU |
| **在线查询** | 用户发起查询 | 延迟 < 50ms | ms/query | 缓存, 量化, CPU |

**完整实现**：参见 `docs/design/EMBEDDING_BEST_PRACTICES.md` 第五章

**子模块评价指标**：

| 指标 | 目标值 | 场景 |
|------|--------|------|
| **在线查询延迟** | < 50ms | 用户查询 |
| **缓存命中率** | > 20% | 用户查询 |
| **离线吞吐量** | > 200 docs/sec | 构建向量库 |
| **错误率** | < 0.1% | 所有场景 |

---

### 3.2 pgvector 存储层

**必须创建文件：**

```text
src/xinyidai_agent/rag/storage.py
```

**核心特性**：
- HNSW 索引（m=16, ef_construction=200）
- 连接池（减少连接开销 25ms → 2ms）
- 批量插入（使用 unnest，提升 20 倍）
- 余弦相似度（15-20% 更快）
- 支持 Dense 和 Sparse 检索

**完整实现**：参见 `docs/design/PGVECTOR_BEST_PRACTICES.md` 第六章

**子模块评价指标**：

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **查询延迟 (p95)** | < 50ms | Dense 检索 |
| **批量插入** | > 1000 docs/sec | 使用 unnest |
| **索引构建时间** | < 10s | 10K 向量 |
| **连接池效率** | 连接开销 < 2ms | vs 25ms（无连接池） |

---

## 4. P1 生产必需项：检索与重排序

### 4.1 检索器（混合检索）

**必须创建文件：**

```text
src/xinyidai_agent/rag/retriever.py
```

**核心策略**：Dense + Sparse + RRF 融合

**完整实现**：参见 `docs/design/RETRIEVER_BEST_PRACTICES.md` 第四章

**子模块评价指标**：

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **Recall@5** | > 0.80 | 前 5 个结果包含答案 |
| **MRR** | > 0.70 | 第一个相关结果的平均排名倒数 |
| **查询延迟** | < 100ms | Dense + Sparse + 融合 |
| **准确率提升** | 20-40% | vs Dense-only |

---

### 4.2 重排序器

**必须创建文件：**

```text
src/xinyidai_agent/rag/reranker.py
```

**核心策略**：Cross-Encoder 重排序

**完整实现**：参见 `docs/design/RERANKER_BEST_PRACTICES.md` 第四章

**子模块评价指标**：

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **MRR 提升** | +0.1-0.2 | vs 无 Reranker |
| **NDCG@5 提升** | +15-25% | 排序质量提升 |
| **延迟** | < 50ms | 20 个候选文档 |
| **准确率提升** | +15-25% | vs 无 Reranker |

---

### 4.3 引用生成器

**必须创建文件：**

```text
src/xinyidai_agent/rag/citation.py
```

**核心功能**：
- 从检索结果生成引用
- 引用包含文档标题、摘要、分数
- 格式化引用为 Prompt 上下文

---

### 4.4 RAG 工具更新

**更新文件：**

```text
src/xinyidai_agent/tools/rag_tool.py
```

**核心流程**：
```
查询 → 向量化 → 混合检索 → 重排序 → 引用生成 → 返回上下文
```

---

## 5. P2 质量与工程化要求

### 5.1 数据入库脚本

**必须创建文件：**

```text
scripts/build_rag_index.py
```

**核心功能**：
- 加载知识库文档
- 文档分块
- 批量向量化
- 批量插入数据库

---

### 5.2 RAG 评测脚本

**必须创建文件：**

```text
scripts/evaluate_rag.py
```

**核心功能**：
- 加载评测数据集
- 运行检索评测
- 计算指标
- 生成评测报告

---

### 5.3 性能优化

**优化策略**：
- 批处理（batch_size=32）
- 查询缓存（减少 30% 延迟）
- 连接池（减少连接开销）
- 并行检索（Dense + Sparse）

---

### 5.4 监控与告警

**监控指标**：
- 请求数、成功率、错误率
- 延迟（p50、p95、p99）
- 资源使用率（CPU、内存、GPU）
- 缓存命中率

**告警规则**：
- 错误率 > 5%
- p95 延迟 > 2000ms
- 内存使用率 > 90%

---

## 6. 实施路线图

### 第 1 周：评测体系

- [ ] 创建评测数据集（50 个开发集）
- [ ] 实现评测指标模块
- [ ] 建立基线指标

### 第 2 周：文档处理

- [ ] 实现 Markdown 分块器
- [ ] 实现文档路由器
- [ ] 实现分块器基类

### 第 3 周：向量化与存储

- [ ] 实现向量化模块
- [ ] 实现 pgvector 存储层
- [ ] 数据入库脚本

### 第 4 周：检索与重排序

- [ ] 实现混合检索器
- [ ] 实现重排序器
- [ ] 实现引用生成器

### 第 5 周：集成与优化

- [ ] RAG 工具集成
- [ ] 性能优化
- [ ] 完整评测

---

## 7. 验收标准

### 7.1 功能验收

- [ ] 能加载知识库文档并分块
- [ ] 能构建向量索引
- [ ] 能进行混合检索
- [ ] 能重排序结果
- [ ] 能生成引用

### 7.2 性能验收

- [ ] Recall@5 > 0.80
- [ ] MRR > 0.70
- [ ] Latency p95 < 1000ms
- [ ] QPS > 10

### 7.3 质量验收

- [ ] Answer Accuracy > 0.85
- [ ] Task Success Rate > 0.80
- [ ] Hallucination Rate < 0.05

---

**最后更新**: 2026-05-08  
**作者**: Claude (Opus 4.7)  
**审核**: 待用户审核
