"""RAG 质量评测：基于 Ragas 评估检索和生成质量。

评测指标：
- faithfulness: 回答是否忠实于检索到的上下文（防幻觉）
- answer_relevancy: 回答是否切题
- context_precision: 检索结果中有多少是真正相关的
- context_recall: 相关文档是否被检索到

运行方式：
    pytest test/evaluation/test_rag_quality.py -v
    # 或单独运行：
    python -m test.evaluation.test_rag_quality

依赖：
    pip install ragas datasets
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 确保 src 在路径中
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from test.evaluation.dataset import RAG_EVAL_DATASET, EvalSample  # noqa: E402


def _skip_if_no_ragas():
    """检查 ragas 是否可用。"""
    try:
        import ragas  # noqa: F401
        return False
    except ImportError:
        return True


def _skip_if_no_api_key():
    """检查 LLM API key 是否配置（Ragas 需要 LLM 做评判）。"""
    return not (os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"))


class RagQualityTest(unittest.TestCase):
    """RAG 检索和生成质量评测。"""

    @unittest.skipIf(_skip_if_no_ragas(), "ragas 未安装，跳过评测")
    @unittest.skipIf(_skip_if_no_api_key(), "未配置 LLM API key，跳过评测")
    def test_rag_faithfulness_and_relevancy(self):
        """评测 RAG 回答的忠实度和相关性。"""
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset

        # 构造 Ragas 期望的数据格式
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        # 调用 Agent 获取实际回答和检索上下文
        from xinyidai_agent.config import AgentConfig, load_env_file
        from xinyidai_agent.llm import OpenAICompatibleChatModel
        from xinyidai_agent.runtime import ControlledAgentLoop
        from xinyidai_agent.protocol import ChatRequest

        load_env_file(AgentConfig.default_env_path())
        model = OpenAICompatibleChatModel(AgentConfig.from_env())
        loop = ControlledAgentLoop(model=model)

        for sample in RAG_EVAL_DATASET:
            response = loop.answer(ChatRequest(user_message=sample.question))
            # 提取检索到的上下文
            retrieved_contexts = [
                source.content for source in response.sources
            ] if response.sources else [""]

            data["question"].append(sample.question)
            data["answer"].append(response.answer)
            data["contexts"].append(retrieved_contexts)
            data["ground_truth"].append(sample.ground_truth)

        dataset = Dataset.from_dict(data)

        # 配置 Ragas 使用百炼兼容接口
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI

        config = AgentConfig.from_env()
        judge_llm = LangchainLLMWrapper(ChatOpenAI(
            model=config.llm_model,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        ))

        # 执行评测
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_llm,
        )

        print("\n" + "=" * 60)
        print("RAG 质量评测结果")
        print("=" * 60)
        for metric_name, score in result.items():
            print(f"  {metric_name}: {score:.4f}")
        print("=" * 60)

        # 断言基线（首次运行可以先看结果再设阈值）
        self.assertGreaterEqual(result["faithfulness"], 0.5, "忠实度低于基线")
        self.assertGreaterEqual(result["answer_relevancy"], 0.5, "相关性低于基线")


class RagOfflineTest(unittest.TestCase):
    """离线 RAG 评测（不需要 LLM API，只验证检索质量）。"""

    def test_dataset_completeness(self):
        """验证评测数据集完整性。"""
        self.assertGreaterEqual(len(RAG_EVAL_DATASET), 5, "RAG 评测集至少需要 5 条样本")
        for sample in RAG_EVAL_DATASET:
            self.assertTrue(sample.question, "question 不能为空")
            self.assertTrue(sample.ground_truth, "ground_truth 不能为空")
            self.assertTrue(sample.contexts, "contexts 不能为空")

    def test_expected_routes_are_valid(self):
        """验证期望路由是合法的 RouteScene。"""
        from xinyidai_agent.protocol import RouteScene
        from typing import get_args

        valid_scenes = set(get_args(RouteScene))
        for sample in RAG_EVAL_DATASET:
            if sample.expected_route:
                self.assertIn(
                    sample.expected_route,
                    valid_scenes,
                    f"样本 '{sample.question}' 的 expected_route 不合法",
                )


if __name__ == "__main__":
    unittest.main()
