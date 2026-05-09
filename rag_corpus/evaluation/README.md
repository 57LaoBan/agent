# RAG 评测数据集说明

本目录是信易贷 RAG 系统 P0 评测集，用于在真实检索器、重排序器和答案生成器接入前建立稳定基线。

## 文件说明

- `qa_pairs.json`：端到端问答评测集，覆盖产品问答、准入规则、授权流程、风险标签、企业信息、政策更新等场景。
- `retrieval_test.json`：检索评测集，覆盖精确匹配、语义匹配、多文档召回、负样本和排序测试。
- `results.json`：运行 `scripts/evaluate_rag.py` 后生成的评测结果文件，不要求提交为固定基线。

## 问答数据格式

```json
{
  "id": "qa_001",
  "category": "产品问答",
  "question": "信易贷有哪些贷款产品？",
  "expected_answer": "参考答案",
  "ground_truth_docs": ["credit_products_and_flow.md"],
  "ground_truth_chunks": ["credit_products_and_flow_chunk_1"],
  "difficulty": "easy",
  "metadata": {
    "requires_multi_doc": false,
    "requires_reasoning": false,
    "has_answer": true
  }
}
```

## 检索数据格式

```json
{
  "id": "retrieval_001",
  "query": "经营周转贷适合什么企业？",
  "relevant_docs": ["credit_products_and_flow.md"],
  "relevant_chunks": ["credit_products_and_flow_chunk_1"],
  "irrelevant_docs": ["risk_label_dictionary.md"],
  "expected_rank": ["credit_products_and_flow_chunk_1"],
  "metadata": {
    "query_type": "factual",
    "ambiguity": "low"
  }
}
```

## 当前规模

- 问答对：24 条
- 检索用例：30 条
- 覆盖难度：easy、medium、hard
- 覆盖场景：产品问答、准入规则、授权流程、风险标签、企业信息、政策更新、证据约束

## 验收命令

```powershell
& 'D:\miniconda3\envs\xinyidai-chat-agent\python.exe' scripts\evaluate_rag.py
```
