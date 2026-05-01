"""Skill 注册表：按 scene 查找并返回 Skill 文档。"""

from __future__ import annotations

from xinyidai_agent.skills.loader import SkillDocument, SkillLoader


class SkillRegistry:
    """Skill 注册表，启动时加载所有 Skill 文档，运行时按 scene 查询。"""

    def __init__(self, loader: SkillLoader | None = None) -> None:
        self._loader = loader or SkillLoader()
        self._skills: dict[str, SkillDocument] = self._loader.load_all()

    def get(self, scene: str) -> SkillDocument | None:
        """按 scene 获取对应的 Skill 文档。"""
        return self._skills.get(scene)

    def get_prompt_section(self, scene: str) -> str:
        """按 scene 获取可注入 prompt 的文本段，找不到则返回空字符串。"""
        doc = self.get(scene)
        if doc is None:
            return ""
        return doc.to_prompt_section()

    @property
    def scenes(self) -> list[str]:
        """返回所有已注册 Skill 的 scene 列表。"""
        return list(self._skills.keys())

    def reload(self) -> None:
        """重新加载所有 Skill 文档。"""
        self._skills = self._loader.load_all()
