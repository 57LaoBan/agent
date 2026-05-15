"""Agent 行为评测：基于 DeepEval 框架。

评测指标：
- HallucinationMetric: 回答是否与检索上下文矛盾（LLM-as-judge）
- ToolCorrectnessMetric: 工具调用是否与期望一致
- AnswerRelevancyMetric: 回答是否切题

运行方式：
    pytest test/evaluation/test_deepeval_agent.py -v
    # 或使用 deepeval 命令：
    deepeval test run test/evaluation/test_deepeval_agent.py

依赖：
    pip install deepeval
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from test.evaluation.dataset import AGENT_EVAL_DATASET, RAG_EVAL_DATASET  # noqa: E402


def _skip_if_no_deepeval():
    try:
        import deepeval  # noqa: F401
        return False
    except ImportError:
        return True


def _skip_if_no_api_key():
    return not (os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"))


class DeepEvalHallucinationTest(unittest.TestCase):
    """基于 DeepEval HallucinationMetric 的幻觉检测。"""

    @unittest.skipIf(_skip_if_no_deepeval(), "deepeval 未安装，跳过")
    @unittest.skipIf(_skip_if_no_api_key(), "未配置 LLM API key，跳过")
    def test_rag_hallucination(self):
        """检测 RAG 回答是否与检索上下文矛盾。"""
        from deepeval import evaluate
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import HallucinationMetric

        from xinyidai_agent.config import AgentConfig, load_env_file
        from xinyidai_agent.llm import OpenAICompatibleChatModel
        from xinyidai_agent.runtime import ControlledAgentLoop
        from xinyidai_agent.protocol import ChatRequest

        load_env_file(AgentConfig.default_env_path())
        config = AgentConfig.from_env()
        model = OpenAICompatibleChatModel(config)
        loop = ControlledAgentLoop(model=model)

        # 配置 DeepEval 使用百炼兼容接口
        os.environ["OPENAI_API_KEY"] = config.llm_api_key
        os.environ["OPENAI_BASE_URL"] = config.llm_base_url

        test_cases: list[LLMTestCase] = []
        for sample in RAG_EVAL_DATASET:
            response = loop.answer(ChatRequest(user_message=sample.question))
            retrieved_contexts = [
                source.content for source in response.sources
            ] if response.sources else sample.contexts

            test_cases.append(LLMTestCase(
                input=sample.question,
                actual_output=response.answer,
                context=retrieved_contexts,
            ))

        metric = HallucinationMetric(
            threshold=0.5,
            model=config.llm_model,
        )

        results = evaluate(test_cases=test_cases, metrics=[metric])

        print("\n" + "=" * 60)
        print("DeepEval 幻觉检测结果")
        print("=" * 60)
        for result in results.test_results:
            status = "✓" if result.success else "✗"
            print(f"  {status} {result.input[:40]}... score={result.metrics_data[0].score:.2f}")
        print(f"\n  通过率: {results.test_results_pass_rate:.2%}")
        print("=" * 60)

        self.assertGreaterEqual(
            results.test_results_pass_rate, 0.6,
            "幻觉检测通过率低于 60%",
        )


class DeepEvalToolCorrectnessTest(unittest.TestCase):
    """基于 DeepEval ToolCorrectnessMetric 的工具调用评测。"""

    @unittest.skipIf(_skip_if_no_deepeval(), "deepeval 未安装，跳过")
    @unittest.skipIf(_skip_if_no_api_key(), "未配置 LLM API key，跳过")
    def test_tool_correctness(self):
        """检测 Agent 工具调用是否与期望一致。"""
        from deepeval import evaluate
        from deepeval.test_case import LLMTestCase, ToolCall
        from deepeval.metrics import ToolCorrectnessMetric

        from xinyidai_agent.config import AgentConfig, load_env_file
        from xinyidai_agent.llm import OpenAICompatibleChatModel
        from xinyidai_agent.runtime import ControlledAgentLoop
        from xinyidai_agent.protocol import ChatRequest

        load_env_file(AgentConfig.default_env_path())
        config = AgentConfig.from_env()
        model = OpenAICompatibleChatModel(config)
        loop = ControlledAgentLoop(model=model)

        os.environ["OPENAI_API_KEY"] = config.llm_api_key
        os.environ["OPENAI_BASE_URL"] = config.llm_base_url

        test_cases: list[LLMTestCase] = []
        for sample in AGENT_EVAL_DATASET:
            if not sample.expected_tools:
                continue
            response = loop.answer(ChatRequest(user_message=sample.question))
            actual_tools = [
                ToolCall(tool_name=t.tool_name, tool_input=t.output)
                for t in response.tool_trace
            ]
            expected_tools = [
                ToolCall(tool_name=name, tool_input={})
                for name in sample.expected_tools
            ]

            test_cases.append(LLMTestCase(
                input=sample.question,
                actual_output=response.answer,
                tools_called=actual_tools,
                expected_tools=expected_tools,
            ))

        if not test_cases:
            self.skipTest("无可评测的工具调用样本")

        metric = ToolCorrectnessMetric()

        results = evaluate(test_cases=test_cases, metrics=[metric])

        print("\n" + "=" * 60)
        print("DeepEval 工具正确性结果")
        print("=" * 60)
        for result in results.test_results:
            status = "✓" if result.success else "✗"
            print(f"  {status} {result.input[:40]}...")
        print(f"\n  通过率: {results.test_results_pass_rate:.2%}")
        print("=" * 60)

        self.assertGreaterEqual(
            results.test_results_pass_rate, 0.5,
            "工具正确性通过率低于 50%",
        )


class DeepEvalAnswerRelevancyTest(unittest.TestCase):
    """基于 DeepEval AnswerRelevancyMetric 的回答相关性评测。"""

    @unittest.skipIf(_skip_if_no_deepeval(), "deepeval 未安装，跳过")
    @unittest.skipIf(_skip_if_no_api_key(), "未配置 LLM API key，跳过")
    def test_answer_relevancy(self):
        """检测回答是否切题。"""
        from deepeval import evaluate
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import AnswerRelevancyMetric

        from xinyidai_agent.config import AgentConfig, load_env_file
        from xinyidai_agent.llm import OpenAICompatibleChatModel
        from xinyidai_agent.runtime import ControlledAgentLoop
        from xinyidai_agent.protocol import ChatRequest

        load_env_file(AgentConfig.default_env_path())
        config = AgentConfig.from_env()
        model = OpenAICompatibleChatModel(config)
        loop = ControlledAgentLoop(model=model)

        os.environ["OPENAI_API_KEY"] = config.llm_api_key
        os.environ["OPENAI_BASE_URL"] = config.llm_base_url

        test_cases: list[LLMTestCase] = []
        all_samples = [*RAG_EVAL_DATASET, *AGENT_EVAL_DATASET]
        for sample in all_samples:
            response = loop.answer(ChatRequest(user_message=sample.question))
            test_cases.append(LLMTestCase(
                input=sample.question,
                actual_output=response.answer,
            ))

        metric = AnswerRelevancyMetric(
            threshold=0.5,
            model=config.llm_model,
        )

        results = evaluate(test_cases=test_cases, metrics=[metric])

        print("\n" + "=" * 60)
        print("DeepEval 回答相关性结果")
        print("=" * 60)
        print(f"  通过率: {results.test_results_pass_rate:.2%}")
        print("=" * 60)

        self.assertGreaterEqual(
            results.test_results_pass_rate, 0.6,
            "回答相关性通过率低于 60%",
        )


if __name__ == "__main__":
    unittest.main()
