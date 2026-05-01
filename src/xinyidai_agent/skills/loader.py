"""Skill 文档加载器：从 Markdown 文件加载业务 Skill 文档。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SkillDocument:
    """一份业务 Skill 文档的结构化表示。"""

    skill_id: str
    scene: str
    title: str
    content: str
    file_path: str = ""
    tags: list[str] = field(default_factory=list)

    def to_prompt_section(self) -> str:
        """将 Skill 文档转为可注入 system prompt 的文本段。"""
        return (
            f"## 当前业务技能：{self.title}\n\n"
            f"{self.content}"
        )


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

        return SkillDocument(
            skill_id=scene.lower(),
            scene=scene,
            title=title,
            content=content,
            file_path=str(file_path),
        )

    def load_all(self) -> dict[str, SkillDocument]:
        """加载目录下所有 Skill 文档，返回 scene → SkillDocument 映射。"""
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
