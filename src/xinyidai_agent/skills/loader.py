"""Skill 文档加载器：从 Markdown 文件加载业务 Skill 文档。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SkillDocument:
    """一份业务 Skill 文档的结构化表示。"""

    skill_id: str
    scene: str
    title: str
    content: str
    summary: str
    file_path: str = ""
    tags: list[str] = field(default_factory=list)

    def to_index_entry(self) -> str:
        """转为 Skill 索引中的一行摘要。"""
        return f"- `{self.scene}`：{self.title}。{self.summary}"

    def to_prompt_section(self, disclosure: str = "summary") -> str:
        """将 Skill 文档按披露层级转为可注入 system prompt 的文本段。"""
        if disclosure == "full":
            body = self.content or self.summary
            return f"## 当前业务技能：{self.title}\n\n{body}"

        section_names = DISCLOSURE_SECTIONS.get(disclosure)
        if section_names is None:
            return self.to_summary_section()

        selected_sections = self.select_sections(section_names)
        if not selected_sections:
            return self.to_summary_section()

        return (
            f"## 当前业务技能：{self.title}\n\n"
            f"### 技能摘要\n{self.summary}\n\n"
            + "\n\n".join(selected_sections)
        )

    def to_summary_section(self) -> str:
        """只暴露当前 Skill 的最小摘要。"""
        return f"## 当前业务技能：{self.title}\n\n### 技能摘要\n{self.summary}"

    def select_sections(self, names: Iterable[str]) -> list[str]:
        """按二级标题选择 Markdown 章节。"""
        sections = self.sections
        return [sections[name] for name in names if name in sections]

    @property
    def sections(self) -> dict[str, str]:
        """按二级标题拆分 Skill 文档。"""
        parsed: dict[str, list[str]] = {}
        current_title = ""

        for line in self.content.splitlines():
            if line.startswith("## "):
                current_title = line.removeprefix("## ").strip()
                parsed[current_title] = [line]
                continue
            if current_title:
                parsed[current_title].append(line)

        return {
            title: "\n".join(lines).strip()
            for title, lines in parsed.items()
            if "\n".join(lines).strip()
        }


class SkillLoader:
    """从指定目录加载 .md 格式的 Skill 文档。

    文件名约定：{scene}.md，例如 DATA_QUERY.md、KNOWLEDGE_QA.md。
    """

    def __init__(self, skill_dir: Path | str | None = None) -> None:
        if skill_dir is None:
            skill_dir = Path(__file__).parent / "documents"
        self._skill_dir = Path(skill_dir)

    def load(self, scene: str) -> SkillDocument | None:
        """按 scene 名加载对应的 Skill 文档。"""
        file_path = self._skill_dir / f"{scene}.md"
        if not file_path.exists():
            return None

        content = file_path.read_text(encoding="utf-8").strip()
        title = self._extract_title(content, scene)
        summary = self._extract_summary(content, title)

        return SkillDocument(
            skill_id=scene.lower(),
            scene=scene,
            title=title,
            content=content,
            summary=summary,
            file_path=str(file_path),
        )

    def load_index(self) -> dict[str, SkillDocument]:
        """加载 Skill 索引，只保留标题和摘要，不把完整正文常驻注册表。"""
        skills: dict[str, SkillDocument] = {}
        if not self._skill_dir.exists():
            return skills

        for file_path in sorted(self._skill_dir.glob("*.md")):
            scene = file_path.stem
            content = file_path.read_text(encoding="utf-8").strip()
            title = self._extract_title(content, scene)
            skills[scene] = SkillDocument(
                skill_id=scene.lower(),
                scene=scene,
                title=title,
                content="",
                summary=self._extract_summary(content, title),
                file_path=str(file_path),
            )

        return skills

    def load_all(self) -> dict[str, SkillDocument]:
        """加载目录下所有完整 Skill 文档，主要用于测试或离线校验。"""
        skills: dict[str, SkillDocument] = {}
        if not self._skill_dir.exists():
            return skills

        for file_path in sorted(self._skill_dir.glob("*.md")):
            scene = file_path.stem
            doc = self.load(scene)
            if doc is not None:
                skills[scene] = doc

        return skills

    def _extract_title(self, content: str, fallback: str) -> str:
        """从 Markdown 内容中提取第一个 # 标题。"""
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        return fallback

    def _extract_summary(self, content: str, fallback: str) -> str:
        """优先从“你的任务”章节提取一段摘要。"""
        lines = content.splitlines()
        in_task_section = False
        summary_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## "):
                if in_task_section:
                    break
                in_task_section = stripped == "## 你的任务"
                continue
            if in_task_section and stripped:
                summary_lines.append(stripped)

        if summary_lines:
            return " ".join(summary_lines)
        return fallback


DISCLOSURE_SECTIONS: dict[str, tuple[str, ...]] = {
    "summary": (),
    "tool_planning": ("你的任务", "决策流程", "工具使用规则"),
    "direct_answer": ("你的任务", "决策流程", "关键约束"),
    "tool_result_answer": ("你的任务", "决策流程", "关键约束"),
}
