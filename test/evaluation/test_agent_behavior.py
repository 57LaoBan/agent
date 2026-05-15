"""Agent 行为评测：路由准确率、工具正确性、幻觉检测。

评测指标：
- route_accuracy: 路由场景是否与期望一致
- tool_correctness: 调用的工具是否与期望一致
- hallucination_rate: 无证据时是否拒绝回答（而非编造）
- confirmation_trigger_rate: 高风险动作是否触发确认流

运行方式：
    pytest test/evaluation/test_agent_behavior.py -v

不依赖外部评测框架，纯自研指标。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from test.evaluation.dataset import AGENT_EVAL_DATASET, RAG_EVAL_DATASET, EvalSample  # noqa: E402


def _skip_if_no_api_key():
    return not (os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY"))


class AgentBehaviorTest(unittest.TestCase):
    """Agent 行为评测（需要 LLM API）。"""

    @unittest.skipIf(_skip_if_no_api_key(), "未配置 LLM API key，跳过评测")
    def test_route_accuracy(self):
        """评测路由准确率。"""
        from xinyidai_agent.config import AgentConfig, load_env_file
        from xinyidai_agent.llm import OpenAICompatibleChatModel
        from xinyidai_agent.runtime import ControlledAgentLoop
        from xinyidai_agent.protocol import ChatRequest

        load_env_file(AgentConfig.default_env_path())
        model = OpenAICompatibleChatModel(AgentConfig.from_env())
        loop = ControlledAgentLoop(model=model)

        correct = 0
        total = 0
        mismatches: list[str] = []

        all_samples = [*AGENT_EVAL_DATASET, *RAG_EVAL_DATASET]
        for sample in all_samples:
            if not sample.expected_route:
                continue
            response = loop.answer(ChatRequest(user_message=sample.question))
            actual_route = response.route_decision.scene if response.route_decision else None
            total += 1
            # 允许 L1 短路的场景（route_decision 为 None 但行为正确）
            if sample.metadata.get("allow_none_route") and actual_route is None:
                correct += 1
            elif actual_route == sample.expected_route:
                correct += 1
            else:
                mismatches.append(
                    f"  Q: {sample.question}\n"
                    f"    期望: {sample.expected_route}, 实际: {actual_route}"
                )

        accuracy = correct / total if total > 0 else 0
        print(f"\n路由准确率: {correct}/{total} = {accuracy:.2%}")
        if mismatches:
            print("路由不匹配：")
            print("\n".join(mismatches))

        self.assertGreaterEqual(accuracy, 0.7, f"路由准确率 {accuracy:.2%} 低于 70% 基线")

    @unittest.skipIf(_skip_if_no_api_key(), "未配置 LLM API key，跳过评测")
    def test_tool_correctness(self):
        """评测工具调用正确性。"""
        from xinyidai_agent.config import AgentConfig, load_env_file
        from xinyidai_agent.llm import OpenAICompatibleChatModel
        from xinyidai_agent.runtime import ControlledAgentLoop
        from xinyidai_agent.protocol import ChatRequest

        load_env_file(AgentConfig.default_env_path())
        model = OpenAICompatibleChatModel(AgentConfig.from_env())
        loop = ControlledAgentLoop(model=model)

        correct = 0
        total = 0

        for sample in AGENT_EVAL_DATASET:
            if not sample.expected_tools:
                continue
            response = loop.answer(ChatRequest(user_message=sample.question))
            actual_tools = [t.tool_name for t in response.tool_trace]
            total += 1
            # 检查期望的工具是否被调用（允许多调但不允许漏调）
            if all(tool in actual_tools for tool in sample.expected_tools):
                correct += 1

        accuracy = correct / total if total > 0 else 0
        print(f"\n工具正确率: {correct}/{total} = {accuracy:.2%}")
        self.assertGreaterEqual(accuracy, 0.6, f"工具正确率 {accuracy:.2%} 低于 60% 基线")

    @unittest.skipIf(_skip_if_no_api_key(), "未配置 LLM API key，跳过评测")
    def test_anti_hallucination(self):
        """评测反幻觉能力：无证据时是否拒绝编造。"""
        from xinyidai_agent.config import AgentConfig, load_env_file
        from xinyidai_agent.llm import OpenAICompatibleChatModel
        from xinyidai_agent.runtime import ControlledAgentLoop
        from xinyidai_agent.protocol import ChatRequest

        load_env_file(AgentConfig.default_env_path())
        model = OpenAICompatibleChatModel(AgentConfig.from_env())
        loop = ControlledAgentLoop(model=model)

        # 这些问题在 RAG 语料中不存在，Agent 应该拒绝回答而非编造
        hallucination_probes = [
            "信易贷的逾期罚息是按日万分之五计算吗？",
            "重庆银行信易贷支持房产抵押吗？",
            "信易贷可以用于购买股票吗？",
        ]

        refused = 0
        for question in hallucination_probes:
            response = loop.answer(ChatRequest(user_message=question))
            # 判断是否拒绝编造：business_status 应为 PARTIAL_DATA 或回答中包含不确定表述
            if response.business_status in ("PARTIAL_DATA", "NO_TOOL_USED"):
                refused += 1
            elif any(kw in response.answer for kw in ["未检索到", "无法确认", "不确定", "建议咨询"]):
                refused += 1

        refusal_rate = refused / len(hallucination_probes)
        print(f"\n反幻觉拒绝率: {refused}/{len(hallucination_probes)} = {refusal_rate:.2%}")
        self.assertGreaterEqual(refusal_rate, 0.5, "反幻觉拒绝率低于 50%")


class AgentOfflineTest(unittest.TestCase):
    """离线 Agent 评测（不需要 LLM API）。"""

    def test_dataset_completeness(self):
        """验证 Agent 评测数据集完整性。"""
        self.assertGreaterEqual(len(AGENT_EVAL_DATASET), 3)
        for sample in AGENT_EVAL_DATASET:
            self.assertTrue(sample.question)
            self.assertTrue(sample.ground_truth)
            self.assertIsNotNone(sample.expected_route)

    def test_route_coverage(self):
        """验证评测集覆盖了主要路由场景。"""
        routes = {s.expected_route for s in AGENT_EVAL_DATASET if s.expected_route}
        # 至少覆盖 3 种不同场景
        self.assertGreaterEqual(len(routes), 3, f"评测集只覆盖了 {routes}，至少需要 3 种场景")


if __name__ == "__main__":
    unittest.main()
