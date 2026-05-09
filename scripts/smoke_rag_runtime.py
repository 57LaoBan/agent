"""冒烟测试：直接打通 ProductionRAGRetriever，验证生产链路可用。

故意使用同步入口（与 RagSearchTool 保持一致），间接验证 AsyncRuntime
+ pgvector 连接池 + BGE-M3 embedding + Hybrid 检索的端到端串联。
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.config import load_env_file  # noqa: E402
from xinyidai_agent.rag import build_production_rag_runtime  # noqa: E402


def main() -> None:
    """构造生产 RAG 运行时并打一组真实查询。"""
    load_env_file(ROOT / "config" / ".env")

    runtime = build_production_rag_runtime()
    # 首次 embed 在 CPU 上有模型加载冷启动，把同步等待时间放大。
    runtime.retriever._retrieve_timeout_seconds = 180.0
    try:
        for query in [
            "小微税贷的利率是多少？",
            "企业授权与数据范围",
            "风险标签字典里高风险标签有哪些？",
        ]:
            print(f"\n>>> Query: {query}")
            sources, trace = runtime.retriever.retrieve(query, top_k=3)
            print(f"  retriever_type={trace.retriever_type} index_version={trace.index_version}")
            print(f"  results_count={trace.results_count}")
            for step in trace.steps:
                print(f"  step: {step}")
            for idx, source in enumerate(sources, start=1):
                preview = source.content[:80].replace("\n", " ")
                print(f"  [{idx}] score={source.score:.4f} title={source.title} preview={preview}...")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
