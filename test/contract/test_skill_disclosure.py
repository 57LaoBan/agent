from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))

from xinyidai_agent.skills import SkillLoader, SkillRegistry  # noqa: E402


class SkillDisclosureContractTest(unittest.TestCase):
    def test_registry_exposes_index_before_detailed_skill_sections(self) -> None:
        registry = SkillRegistry()

        index_section = registry.get_index_prompt_section()
        summary_section = registry.get_prompt_section("DATA_QUERY")
        planning_section = registry.get_prompt_section("DATA_QUERY", "tool_planning")

        self.assertIn("业务 Skill 索引", index_section)
        self.assertIn("`DATA_QUERY`", index_section)
        self.assertIn("技能摘要", summary_section)
        self.assertNotIn("query_credit_amount", summary_section)
        self.assertIn("query_credit_amount", planning_section)
        self.assertIn("## 决策流程", planning_section)
        self.assertIn("## 工具使用规则", planning_section)

    def test_registry_lazy_loads_full_document_only_when_expanded(self) -> None:
        registry = SkillRegistry()

        self.assertEqual(registry._cache, {})

        summary_section = registry.get_prompt_section("KNOWLEDGE_QA")
        self.assertIn("技能摘要", summary_section)
        self.assertEqual(registry._cache, {})

        expanded_section = registry.get_prompt_section("KNOWLEDGE_QA", "tool_planning")
        self.assertIn("rag_search", expanded_section)
        self.assertIn("KNOWLEDGE_QA", registry._cache)

    def test_loader_can_split_markdown_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            skill_path = Path(tmp_dir) / "TEST.md"
            skill_path.write_text(
                "\n".join(
                    [
                        "# 测试技能",
                        "",
                        "## 你的任务",
                        "只暴露这一段作为摘要。",
                        "",
                        "## 决策流程",
                        "先判断，再执行。",
                        "",
                        "## 工具使用规则",
                        "- 调用 test_tool",
                    ]
                ),
                encoding="utf-8",
            )

            loader = SkillLoader(tmp_dir)
            doc = loader.load("TEST")

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.summary, "只暴露这一段作为摘要。")
        self.assertIn("决策流程", doc.sections)
        self.assertEqual(
            doc.select_sections(["工具使用规则"]),
            ["## 工具使用规则\n- 调用 test_tool"],
        )


if __name__ == "__main__":
    unittest.main()
