# RAG 系统生产级指标体系

> 完整的评测指标、监控指标、业务指标设计

**文档版本**: v1.0  
**更新日期**: 2026-05-08  
**适用场景**: 生产级 RAG 系统

---

## 一、指标体系架构

```
业务指标（最重要）
    ↓
检索质量指标
    ↓
性能指标
    ↓
成本指标
```

---

## 二、业务指标（P0 必须）

### 2.1 答案准确率（Answer Accuracy）

**定义**：LLM 生成的答案是否正确

**计算方式**：人工标注

```python
@dataclass
class AnswerEvaluation:
    """答案评估结果。"""
    
    query: str
    generated_answer: str
    expected_answer: str
    is_correct: bool  # 人工标注
    error_type: str | None  # "hallucination" | "incomplete" | "irrelevant" | None
```

**目标值**：
- 简单查询：> 0.95
- 中等查询：> 0.85
- 困难查询：> 0.70
- **整体**：**> 0.85**

**评估方式**：
```python
# 每周随机抽样 50 个查询
# 人工标注答案是否正确
# 计算准确率
accuracy = correct_count / total_count
```

---

### 2.2 任务成功率（Task Success Rate）

**定义**：用户是否通过系统找到了答案

**计算方式**：用户反馈

```python
@dataclass
class TaskResult:
    """任务结果。"""
    
    query: str
    user_feedback: str  # "success" | "failed" | "partial"
    follow_up_count: int  # 追问次数
```

**目标值**：**> 0.80**

**评估方式**：
```python
# 方案 1：显式反馈（点赞/点踩）
# 方案 2：隐式反馈（是否追问、会话时长）
# 方案 3：定期用户调研

# 计算
success_rate = success_count / total_count
```

---

### 2.3 拒答准确率（Abstention Accuracy）

**定义**：当知识库中没有答案时，系统是否正确拒绝回答

**计算方式**：人工标注

```python
@dataclass
class AbstentionEvaluation:
    """拒答评估结果。"""
    
    query: str
    has_answer_in_kb: bool  # 知识库中是否有答案
    system_abstained: bool  # 系统是否拒绝回答
    is_correct: bool  # 拒答决策是否正确
```

**目标值**：**> 0.90**

**评估方式**：
```python
# 构建"无答案"测试集（50 个查询）
# 评估系统是否正确拒绝回答

# True Positive：知识库无答案，系统拒答
# True Negative：知识库有答案，系统回答
# False Positive：知识库有答案，系统拒答
# False Negative：知识库无答案，系统回答（幻觉）

abstention_accuracy = (TP + TN) / (TP + TN + FP + FN)
```

---

### 2.4 幻觉率（Hallucination Rate）

**定义**：LLM 生成的答案中包含知识库中不存在的信息的比例

**计算方式**：人工标注

**目标值**：**< 0.05**（低于 5%）

**评估方式**：
```python
# 每周随机抽样 50 个答案
# 人工检查是否包含知识库中不存在的信息
hallucination_rate = hallucination_count / total_count
```

---

## 三、检索质量指标（P0 必须）

### 3.1 召回率（Recall@K）

**定义**：前 K 个检索结果中包含相关文档的比例

**计算方式**：

```python
def calculate_recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    """计算 Recall@K。"""
    retrieved_k = set(retrieved_chunk_ids[:k])
    relevant_set = set(relevant_chunk_ids)
    
    hits = len(retrieved_k & relevant_set)
    return hits / len(relevant_set) if relevant_set else 0.0
```

**目标值**：
- **Recall@5**：**> 0.80**（最终结果）
- Recall@20：> 0.90（重排序前）
- Recall@100：> 0.95（检索阶段）

---

### 3.2 平均倒数排名（MRR）

**定义**：第一个相关文档的平均排名倒数

**计算方式**：

```python
def calculate_mrr(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
) -> float:
    """计算 MRR。"""
    relevant_set = set(relevant_chunk_ids)
    
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    
    return 0.0
```

**目标值**：**> 0.70**

**含义**：
- MRR = 1.0：第一个结果就是相关的
- MRR = 0.5：第二个结果是相关的
- MRR = 0.33：第三个结果是相关的

---

### 3.3 归一化折损累积增益（NDCG@K）

**定义**：考虑排序质量的指标

**计算方式**：

```python
import numpy as np

def calculate_ndcg_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    """计算 NDCG@K。"""
    relevant_set = set(relevant_chunk_ids)
    
    # DCG
    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1):
        if chunk_id in relevant_set:
            dcg += 1.0 / np.log2(rank + 1)
    
    # IDCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_chunk_ids))))
    
    return dcg / idcg if idcg > 0 else 0.0
```

**目标值**：**NDCG@5 > 0.80**

---

### 3.4 精确率（Precision@K）

**定义**：前 K 个检索结果中相关文档的比例

**计算方式**：

```python
def calculate_precision_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    """计算 Precision@K。"""
    retrieved_k = set(retrieved_chunk_ids[:k])
    relevant_set = set(relevant_chunk_ids)
    
    hits = len(retrieved_k & relevant_set)
    return hits / k if k > 0 else 0.0
```

**目标值**：**Precision@5 > 0.60**

---

## 四、性能指标（P0 必须）

### 4.1 端到端延迟

**定义**：从用户发起查询到返回答案的总时间

**分解**：

```python
@dataclass
class LatencyBreakdown:
    """延迟分解。"""
    
    query_embedding: float  # 查询向量化
    dense_retrieval: float  # Dense 检索
    sparse_retrieval: float  # Sparse 检索
    fusion: float  # 结果融合
    reranking: float  # 重排序
    llm_generation: float  # LLM 生成
    total: float  # 总延迟
```

**目标值**：
- **p50（中位数）**：**< 500ms**
- **p95**：**< 1000ms**
- **p99**：**< 2000ms**

**各阶段目标**：
- 查询向量化：< 50ms
- 检索（Dense + Sparse）：< 100ms
- 重排序：< 50ms
- LLM 生成：< 500ms

---

### 4.2 吞吐量（QPS）

**定义**：每秒处理的查询数

**目标值**：
- 单机：**> 10 QPS**
- 集群：根据业务需求

**监控方式**：
```python
# 使用 Prometheus + Grafana
qps = total_requests / time_window
```

---

### 4.3 缓存命中率

**定义**：查询缓存命中的比例

**目标值**：**> 20%**

**计算方式**：
```python
cache_hit_rate = cache_hits / total_requests
```

---

## 五、成本指标（P1 推荐）

### 5.1 每次查询成本

**定义**：单次查询的平均成本

**计算方式**：

```python
@dataclass
class QueryCost:
    """查询成本。"""
    
    embedding_cost: float  # 向量化成本（API 或计算资源）
    retrieval_cost: float  # 检索成本（数据库查询）
    reranking_cost: float  # 重排序成本（计算资源）
    llm_cost: float  # LLM 生成成本（API 或计算资源）
    total_cost: float  # 总成本
```

**目标值**：
- 使用本地模型：**< ¥0.01/查询**
- 使用 API：**< ¥0.10/查询**

---

### 5.2 资源利用率

**定义**：CPU、内存、GPU 的使用率

**目标值**：
- CPU 利用率：60-80%
- 内存利用率：< 80%
- GPU 利用率：60-80%（如果使用）

---

## 六、评测数据集设计

### 6.1 数据集规模

```python
# 开发集（Dev Set）：快速迭代
dev_set_size = 50  # 50 个问答对

# 测试集（Test Set）：最终评估
test_set_size = 200  # 200 个问答对

# 无答案测试集：评估拒答能力
no_answer_set_size = 50  # 50 个无答案查询
```

### 6.2 数据集分层

**按难度分层**：
- 简单（40%）：单文档、单段落、事实性问答
- 中等（40%）：多文档、需要推理
- 困难（20%）：多跳推理、复杂条件

**按场景分层**：
- 产品问答（30%）
- 准入规则（30%）
- 授权流程（20%）
- 风险标签（10%）
- 其他（10%）

### 6.3 数据集格式

```python
@dataclass
class EvaluationExample:
    """评测样例。"""
    
    id: str
    category: str  # "产品问答" | "准入规则" | ...
    difficulty: str  # "easy" | "medium" | "hard"
    
    # 查询
    question: str
    
    # Ground Truth
    expected_answer: str
    relevant_chunk_ids: list[str]  # 相关分块 ID
    relevant_doc_ids: list[str]  # 相关文档 ID
    
    # 元数据
    requires_multi_doc: bool  # 是否需要多文档
    requires_reasoning: bool  # 是否需要推理
    has_answer: bool  # 知识库中是否有答案
```

---

## 七、监控指标（P0 必须）

### 7.1 实时监控指标

```python
# 使用 Prometheus + Grafana

# 1. 请求指标
- total_requests：总请求数
- success_requests：成功请求数
- failed_requests：失败请求数
- error_rate：错误率

# 2. 延迟指标
- latency_p50：中位数延迟
- latency_p95：95 分位延迟
- latency_p99：99 分位延迟

# 3. 资源指标
- cpu_usage：CPU 使用率
- memory_usage：内存使用率
- gpu_usage：GPU 使用率（如果使用）

# 4. 缓存指标
- cache_hit_rate：缓存命中率
- cache_size：缓存大小
```

### 7.2 告警规则

```yaml
# Prometheus 告警规则

groups:
  - name: rag_system_alerts
    rules:
      # 错误率告警
      - alert: HighErrorRate
        expr: error_rate > 0.05
        for: 5m
        annotations:
          summary: "错误率超过 5%"
      
      # 延迟告警
      - alert: HighLatency
        expr: latency_p95 > 2000
        for: 5m
        annotations:
          summary: "p95 延迟超过 2 秒"
      
      # 资源告警
      - alert: HighMemoryUsage
        expr: memory_usage > 0.9
        for: 5m
        annotations:
          summary: "内存使用率超过 90%"
```

---

## 八、评测流程

### 8.1 离线评测（每周）

```python
# 1. 准备评测数据集
test_set = load_test_set("test_set_v1.json")

# 2. 运行评测
results = []
for example in test_set:
    # 检索
    retrieved = await retriever.retrieve(example.question, top_k=20)
    
    # 重排序
    reranked = reranker.rerank(example.question, retrieved, top_k=5)
    
    # 计算指标
    recall = calculate_recall_at_k(
        [r.chunk_id for r in reranked],
        example.relevant_chunk_ids,
        k=5,
    )
    mrr = calculate_mrr(
        [r.chunk_id for r in reranked],
        example.relevant_chunk_ids,
    )
    
    results.append({
        "id": example.id,
        "recall@5": recall,
        "mrr": mrr,
    })

# 3. 汇总指标
avg_recall = np.mean([r["recall@5"] for r in results])
avg_mrr = np.mean([r["mrr"] for r in results])

print(f"Recall@5: {avg_recall:.4f}")
print(f"MRR: {avg_mrr:.4f}")
```

### 8.2 在线评测（持续）

```python
# 1. 随机抽样（5% 流量）
if random.random() < 0.05:
    # 记录检索结果
    log_retrieval_result(query, retrieved_chunks)

# 2. 人工标注（每周 50 个）
# 3. 计算指标
# 4. 监控指标变化趋势
```

---

## 九、信易贷项目指标配置

### 9.1 核心指标

```python
# 业务指标
ANSWER_ACCURACY_TARGET = 0.85
TASK_SUCCESS_RATE_TARGET = 0.80
ABSTENTION_ACCURACY_TARGET = 0.90
HALLUCINATION_RATE_TARGET = 0.05

# 检索指标
RECALL_AT_5_TARGET = 0.80
MRR_TARGET = 0.70
NDCG_AT_5_TARGET = 0.80

# 性能指标
LATENCY_P95_TARGET = 1000  # ms
QPS_TARGET = 10
CACHE_HIT_RATE_TARGET = 0.20

# 成本指标
COST_PER_QUERY_TARGET = 0.01  # 元
```

### 9.2 评测数据集

```python
# 初期（快速验证）
DEV_SET_SIZE = 50

# 后期（完整评估）
TEST_SET_SIZE = 200
NO_ANSWER_SET_SIZE = 50
```

---

## 十、总结

### 10.1 指标优先级

**P0（必须实现）**：
- ✅ Answer Accuracy（答案准确率）
- ✅ Recall@5（召回率）
- ✅ MRR（平均倒数排名）
- ✅ Latency p95（延迟）

**P1（推荐实现）**：
- ✅ Task Success Rate（任务成功率）
- ✅ Abstention Accuracy（拒答准确率）
- ✅ NDCG@5（排序质量）
- ✅ Cost per Query（成本）

**P2（可选实现）**：
- ⏳ Hallucination Rate（幻觉率）
- ⏳ Multi-hop Recall（多跳召回率）
- ⏳ Resource Utilization（资源利用率）

### 10.2 实施路线图

**第 1 周**：
- 创建开发集（50 个问答对）
- 实现 Recall@5、MRR 计算
- 建立基线指标

**第 2-4 周**：
- 扩展测试集（200 个问答对）
- 实现完整评测流程
- 部署监控系统

**第 5+ 周**：
- 持续优化
- 收集用户反馈
- 迭代改进

---

**最后更新**: 2026-05-08  
**作者**: Claude (Opus 4.7)  
**审核**: 待用户审核
