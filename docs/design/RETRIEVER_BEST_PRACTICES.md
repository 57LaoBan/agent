# 检索器（Retriever）最佳实践综述

> 基于生产级 RAG 系统的混合检索实战经验

**文档版本**: v1.0  
**更新日期**: 2026-05-08  
**适用场景**: 生产级 RAG 系统检索模块

---

## 一、为什么纯向量检索不够

### 1.1 向量检索的局限性

来源：[Why Pure Vector Search Fails](https://tianpan.co/blog/2026-04-09-production-retrieval-stack-hybrid-search-reranking)

> "用户搜索产品 ID、发票号码、法规代码、拼写错误的竞争对手名称，以及单个 embedding 向量无法在几何上满足的多约束问题。Dense 向量检索在这些场景下会失败。"

**向量检索的失效场景**：

| 场景 | 问题 | 示例 |
|------|------|------|
| **精确关键词** | 返回语义相近但错误的结果 | 搜索"小微税贷"，返回"小微信贷" |
| **稀有术语** | 向量表示不足 | 搜索"XYDAI-2024-001"（文档编号） |
| **专有名词** | 拼写错误无法匹配 | 搜索"信易代"（拼写错误） |
| **多约束条件** | 单个向量无法同时满足 | "利率低于5%且额度超过100万的产品" |

**几何约束限制**：

来源：[A Decision Framework for Hybrid Retrieval](https://tianpan.co/blog/2026-04-17-hybrid-retrieval-architecture-beyond-embeddings)

- 512 维模型：约 50 万文档时开始失效
- 4096 维模型：约 2.5 亿文档时崩溃
- 原因：高维空间中的"维度诅咒"

### 1.2 生产环境的真实需求

来源：[RAG in Production: 10 Real Systems](https://www.looming.tech/post/rag-in-production-lessons-enterprise-deployments)

> "混合 RAG——Dense 向量检索结合 Sparse 关键词检索（BM25 或类似），中间加一个 Reranker。这是大多数生产系统的选择。双重检索提升召回率，Reranker 提升精度。"

**生产系统的检索需求**：
- ✅ 语义理解（同义词、释义）
- ✅ 精确匹配（ID、编号、专有名词）
- ✅ 容错能力（拼写错误、缩写）
- ✅ 多约束查询（价格 + 地区 + 产品类型）

**单一检索方法的问题**：
- Dense only：精确匹配差
- Sparse only：语义理解差
- 需要：**混合检索**

---

## 二、混合检索策略

### 2.1 Dense + Sparse 互补性

来源：[Hybrid Search RAG Complete Guide 2026](https://calmops.com/ai/hybrid-search-rag-complete-guide-2026/)

**Dense 检索（向量检索）**：
- 优势：语义理解、同义词、释义
- 劣势：精确匹配、稀有词、专有名词
- 适合：概念性查询（"如何申请贷款？"）

**Sparse 检索（BM25）**：
- 优势：精确匹配、稀有词、实体名称
- 劣势：语义理解、同义词
- 适合：精确查询（"小微税贷的利率"）

**互补性证明**：

来源：[Why Pure Vector Search Fails](https://tianpan.co/blog/2026-04-09-production-retrieval-stack-hybrid-search-reranking)

> "在 BEIR 基准测试的 18 个数据集中，BM25 在论证检索任务上的表现无人超越。Dense 检索处理释义和同义词，Sparse 检索处理精确匹配和标识符。两者检索的文档集合互补。"

### 2.2 BM25 算法

**核心公式**：

```
BM25(q, d) = Σ IDF(qi) * (f(qi, d) * (k1 + 1)) / (f(qi, d) + k1 * (1 - b + b * |d| / avgdl))
```

**参数说明**：
- `f(qi, d)`：词 qi 在文档 d 中的频率
- `|d|`：文档长度
- `avgdl`：平均文档长度
- `k1`：词频饱和参数（默认 1.2）
- `b`：长度归一化参数（默认 0.75）

**PostgreSQL 实现**（使用全文搜索）：

```sql
-- 创建 tsvector 列
ALTER TABLE rag_chunks ADD COLUMN content_tsv tsvector;

-- 更新 tsvector
UPDATE rag_chunks SET content_tsv = to_tsvector('chinese', content);

-- 创建 GIN 索引
CREATE INDEX idx_chunks_content_tsv ON rag_chunks USING gin(content_tsv);

-- BM25 查询
SELECT
    chunk_id,
    content,
    ts_rank_cd(content_tsv, query) AS bm25_score
FROM rag_chunks, to_tsquery('chinese', '小微 & 税贷') query
WHERE content_tsv @@ query
ORDER BY bm25_score DESC
LIMIT 10;
```

### 2.3 混合检索架构

**并行检索 + 融合**：

```
查询
  ↓
  ├─→ Dense 检索（top-100）
  │     ↓
  │   向量相似度
  │
  └─→ Sparse 检索（top-100）
        ↓
      BM25 分数
        ↓
      融合算法（RRF）
        ↓
      合并结果（top-20）
        ↓
      Reranker
        ↓
      最终结果（top-5）
```

**性能提升**：

来源：[Building RAG Systems with Hybrid Search](https://propelius.tech/blogs/rag-hybrid-search-dense-sparse-retrieval/)

> "混合检索相比 Dense-only 检索提升 20-40% 的准确率。结合 BM25（关键词）和向量 embeddings（语义）效果最佳。"

---

## 三、结果融合算法

### 3.1 Reciprocal Rank Fusion (RRF)

来源：[Hybrid Search for RAG: BM25, SPLADE, and Vector Search](https://blog.premai.io/hybrid-search-for-rag-bm25-splade-and-vector-search-combined/)

**核心公式**：

```
RRF_score(d) = Σ 1 / (k + rank_method(d))
```

**参数说明**：
- `rank_method(d)`：文档 d 在某个检索方法中的排名
- `k`：常数（默认 60），防止除零

**优势**：
- ✅ 只使用排名，不依赖原始分数
- ✅ 避免不同量纲的归一化问题
- ✅ 简单、稳健、效果好

**示例**：

```python
def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """RRF 融合算法。
    
    Args:
        dense_results: Dense 检索结果 [(doc_id, score), ...]
        sparse_results: Sparse 检索结果 [(doc_id, score), ...]
        k: RRF 常数
        
    Returns:
        融合后的结果 [(doc_id, rrf_score), ...]
    """
    rrf_scores = {}
    
    # Dense 检索的 RRF 分数
    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
    
    # Sparse 检索的 RRF 分数
    for rank, (doc_id, _) in enumerate(sparse_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
    
    # 排序
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    
    return sorted_results
```

### 3.2 线性插值融合

**核心公式**：

```
score(d) = α * dense_score(d) + (1 - α) * sparse_score(d)
```

**参数说明**：
- `α`：Dense 权重（默认 0.7）
- `1 - α`：Sparse 权重（默认 0.3）

**优势**：
- ✅ 可调权重
- ✅ 适合已归一化的分数

**劣势**：
- ❌ 需要分数归一化
- ❌ 不同检索方法的分数量纲不同

**示例**：

```python
def linear_interpolation_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    alpha: float = 0.7,
) -> list[tuple[str, float]]:
    """线性插值融合。
    
    Args:
        dense_results: Dense 检索结果（已归一化）
        sparse_results: Sparse 检索结果（已归一化）
        alpha: Dense 权重
        
    Returns:
        融合后的结果
    """
    # 转为字典
    dense_dict = dict(dense_results)
    sparse_dict = dict(sparse_results)
    
    # 合并所有文档 ID
    all_doc_ids = set(dense_dict.keys()) | set(sparse_dict.keys())
    
    # 计算融合分数
    fused_scores = {}
    for doc_id in all_doc_ids:
        dense_score = dense_dict.get(doc_id, 0.0)
        sparse_score = sparse_dict.get(doc_id, 0.0)
        fused_scores[doc_id] = alpha * dense_score + (1 - alpha) * sparse_score
    
    # 排序
    sorted_results = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    
    return sorted_results
```

### 3.3 融合算法对比

| 算法 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **RRF** | 简单、稳健、无需归一化 | 忽略原始分数 | ⭐⭐⭐⭐⭐ |
| **线性插值** | 可调权重、保留分数信息 | 需要归一化 | ⭐⭐⭐⭐ |
| **加权和** | 灵活 | 需要调参 | ⭐⭐⭐ |

**信易贷项目推荐**：**RRF**（简单、稳健）

---

## 四、信易贷项目实现方案

### 4.1 完整实现

**必须创建文件**：

```text
src/xinyidai_agent/rag/retriever.py
```

**完整代码**：

```python
"""RAG 检索器（混合检索）。"""

from __future__ import annotations

from dataclasses import dataclass

from xinyidai_agent.rag.embedder import BGEEmbedder
from xinyidai_agent.rag.storage import PgVectorStore


@dataclass
class RetrievalResult:
    """检索结果。"""
    
    chunk_id: str
    doc_id: str
    content: str
    score: float
    metadata: dict


class DenseRetriever:
    """Dense 检索器（基于向量相似度）。"""
    
    def __init__(
        self,
        embedder: BGEEmbedder,
        store: PgVectorStore,
    ) -> None:
        """初始化检索器。
        
        Args:
            embedder: 向量化器
            store: 存储层
        """
        self._embedder = embedder
        self._store = store
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Dense 检索。
        
        Args:
            query: 查询文本
            top_k: 返回前 K 个结果
            
        Returns:
            检索结果列表
        """
        # 1. 向量化查询
        query_embedding = self._embedder.embed_query(query)
        
        # 2. 向量检索
        results = await self._store.search_similar(
            query_embedding,
            top_k=top_k,
        )
        
        # 3. 转换为 RetrievalResult
        return [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                content=r["content"],
                score=r["similarity"],
                metadata=r["metadata"],
            )
            for r in results
        ]


class SparseRetriever:
    """Sparse 检索器（基于 BM25）。"""
    
    def __init__(self, store: PgVectorStore) -> None:
        """初始化检索器。
        
        Args:
            store: 存储层
        """
        self._store = store
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """BM25 检索。
        
        Args:
            query: 查询文本
            top_k: 返回前 K 个结果
            
        Returns:
            检索结果列表
        """
        # 使用 PostgreSQL 全文搜索
        async with self._store._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    chunk_id,
                    doc_id,
                    content,
                    ts_rank_cd(content_tsv, query) AS bm25_score,
                    metadata
                FROM rag_chunks, to_tsquery('chinese', $1) query
                WHERE content_tsv @@ query
                ORDER BY bm25_score DESC
                LIMIT $2
                """,
                query, top_k,
            )
            
            return [
                RetrievalResult(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    content=row["content"],
                    score=float(row["bm25_score"]),
                    metadata=row["metadata"],
                )
                for row in rows
            ]


class HybridRetriever:
    """混合检索器（Dense + Sparse + RRF）。"""
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        rrf_k: int = 60,
    ) -> None:
        """初始化混合检索器。
        
        Args:
            dense_retriever: Dense 检索器
            sparse_retriever: Sparse 检索器
            rrf_k: RRF 常数
        """
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._rrf_k = rrf_k
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """混合检索。
        
        Args:
            query: 查询文本
            top_k: 返回前 K 个结果
            
        Returns:
            检索结果列表
        """
        # 1. 并行检索
        dense_results = await self._dense_retriever.retrieve(query, top_k=top_k * 2)
        sparse_results = await self._sparse_retriever.retrieve(query, top_k=top_k * 2)
        
        # 2. RRF 融合
        fused_results = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
        )
        
        return fused_results[:top_k]
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """RRF 融合算法。"""
        rrf_scores = {}
        result_map = {}
        
        # Dense 检索的 RRF 分数
        for rank, result in enumerate(dense_results, start=1):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + 1 / (self._rrf_k + rank)
            result_map[result.chunk_id] = result
        
        # Sparse 检索的 RRF 分数
        for rank, result in enumerate(sparse_results, start=1):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + 1 / (self._rrf_k + rank)
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result
        
        # 更新分数并排序
        fused_results = []
        for chunk_id, rrf_score in rrf_scores.items():
            result = result_map[chunk_id]
            result.score = rrf_score
            fused_results.append(result)
        
        fused_results.sort(key=lambda r: r.score, reverse=True)
        
        return fused_results
```

### 4.2 使用示例

**初始化**：

```python
# 初始化组件
embedder = BGEEmbedder()
store = PgVectorStore("postgresql://...")
await store.initialize()

# 创建检索器
dense_retriever = DenseRetriever(embedder, store)
sparse_retriever = SparseRetriever(store)
hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
```

**检索**：

```python
# 混合检索
results = await hybrid_retriever.retrieve("小微税贷的利率是多少？", top_k=5)

for result in results:
    print(f"[{result.score:.4f}] {result.content[:100]}...")
```

### 4.3 性能目标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **召回率 (Recall@5)** | > 0.8 | 前 5 个结果包含答案 |
| **MRR** | > 0.7 | 第一个相关结果的平均排名倒数 |
| **查询延迟 (p95)** | < 100ms | Dense + Sparse + 融合 |
| **准确率提升** | 20-40% | vs Dense-only |

---

## 五、查询优化策略

### 5.1 查询改写

**问题**：用户查询可能不完整或有歧义

**解决方案**：查询改写

```python
class QueryRewriter:
    """查询改写器。"""
    
    def rewrite(self, query: str) -> str:
        """改写查询。
        
        策略：
        1. 扩展缩写（"信易贷" → "信易贷 小微税贷"）
        2. 添加同义词（"利率" → "利率 年化利率 APR"）
        3. 纠正拼写错误
        """
        # 扩展缩写
        query = self._expand_abbreviations(query)
        
        # 添加同义词
        query = self._add_synonyms(query)
        
        return query
```

### 5.2 查询路由

**问题**：不同类型的查询需要不同的检索策略

**解决方案**：查询路由

```python
class QueryRouter:
    """查询路由器。"""
    
    def route(self, query: str) -> str:
        """路由查询到合适的检索器。
        
        Returns:
            "dense" | "sparse" | "hybrid"
        """
        # 精确查询（包含 ID、编号）→ Sparse
        if self._is_exact_query(query):
            return "sparse"
        
        # 概念性查询（"如何"、"为什么"）→ Dense
        if self._is_conceptual_query(query):
            return "dense"
        
        # 默认 → Hybrid
        return "hybrid"
```

### 5.3 元数据过滤

**问题**：需要在特定文档类型中检索

**解决方案**：元数据过滤

```python
async def retrieve_with_filter(
    query: str,
    doc_type: str,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """带元数据过滤的检索。"""
    # 1. 检索（扩大范围）
    results = await hybrid_retriever.retrieve(query, top_k=top_k * 3)
    
    # 2. 过滤
    filtered_results = [
        r for r in results
        if r.metadata.get("doc_type") == doc_type
    ]
    
    return filtered_results[:top_k]
```

---

## 六、常见问题与解决方案

### 6.1 BM25 检索返回空结果

**问题**：Sparse 检索没有返回结果

**原因**：
- 没有创建 tsvector 列
- 没有创建 GIN 索引
- 查询语法错误

**解决方案**：

```sql
-- 检查 tsvector 列
\d rag_chunks

-- 创建 tsvector 列
ALTER TABLE rag_chunks ADD COLUMN content_tsv tsvector;
UPDATE rag_chunks SET content_tsv = to_tsvector('chinese', content);

-- 创建 GIN 索引
CREATE INDEX idx_chunks_content_tsv ON rag_chunks USING gin(content_tsv);

-- 测试查询
SELECT * FROM rag_chunks
WHERE content_tsv @@ to_tsquery('chinese', '小微 & 税贷');
```

### 6.2 混合检索比 Dense-only 慢

**问题**：混合检索延迟 > 200ms

**原因**：
- 串行执行 Dense 和 Sparse
- 没有使用索引

**解决方案**：

```python
# 并行执行
import asyncio

dense_task = asyncio.create_task(dense_retriever.retrieve(query, top_k))
sparse_task = asyncio.create_task(sparse_retriever.retrieve(query, top_k))

dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
```

### 6.3 RRF 融合效果不好

**问题**：融合后的结果不如单一检索

**原因**：
- `k` 参数设置不当
- Dense 和 Sparse 检索质量差异大

**解决方案**：

```python
# 调整 k 参数
hybrid_retriever = HybridRetriever(
    dense_retriever,
    sparse_retriever,
    rrf_k=40,  # 从 60 降至 40，增加高排名文档的权重
)

# 或使用线性插值
fused_score = 0.8 * dense_score + 0.2 * sparse_score  # 增加 Dense 权重
```

---

## 七、总结

### 7.1 核心要点

1. **混合检索**：Dense + Sparse（提升 20-40%）
2. **融合算法**：RRF（简单、稳健）
3. **并行执行**：减少延迟
4. **查询优化**：改写、路由、过滤

### 7.2 信易贷项目配置

**推荐配置**：

```python
# 混合检索器
hybrid_retriever = HybridRetriever(
    dense_retriever=DenseRetriever(embedder, store),
    sparse_retriever=SparseRetriever(store),
    rrf_k=60,  # RRF 常数
)

# 检索
results = await hybrid_retriever.retrieve(
    query="小微税贷的利率是多少？",
    top_k=5,
)
```

**性能目标**：
- Recall@5：> 0.8
- MRR：> 0.7
- 查询延迟 (p95)：< 100ms

### 7.3 实施优先级

**P0（必须）**：
- ✅ Dense 检索（向量相似度）
- ✅ 混合检索（Dense + Sparse + RRF）

**P1（推荐）**：
- ✅ 查询改写
- ✅ 元数据过滤
- ✅ 并行执行

**P2（可选）**：
- ⏳ 查询路由
- ⏳ 多路融合（Dense + Sparse + SPLADE）
- ⏳ 自适应权重

---

## 八、参考资料

### 8.1 技术文章

- [Why Pure Vector Search Fails](https://tianpan.co/blog/2026-04-09-production-retrieval-stack-hybrid-search-reranking)
- [Hybrid Search for RAG: BM25, SPLADE, and Vector Search](https://blog.premai.io/hybrid-search-for-rag-bm25-splade-and-vector-search-combined/)
- [Building RAG Systems with Hybrid Search](https://propelius.tech/blogs/rag-hybrid-search-dense-sparse-retrieval/)
- [A Decision Framework for Hybrid Retrieval](https://tianpan.co/blog/2026-04-17-hybrid-retrieval-architecture-beyond-embeddings)

### 8.2 官方文档

- [PostgreSQL Full Text Search](https://www.postgresql.org/docs/current/textsearch.html)
- [pgvector Documentation](https://github.com/pgvector/pgvector)

---

**最后更新**: 2026-05-08  
**作者**: Claude (Opus 4.7)  
**审核**: 待用户审核

