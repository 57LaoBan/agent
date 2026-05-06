"""Skill 注册表：先暴露索引，再按 scene 和阶段展开 Skill 文档。"""

from __future__ import annotations

from xinyidai_agent.skills.loader import SkillDocument, SkillLoader


class SkillRegistry:
    """Skill 注册表。

    启动时只加载 Skill 索引摘要；运行时按 scene 懒加载完整文档，并按阶段选择章节注入 prompt。
    """

    def __init__(self, loader: SkillLoader | None = None) -> None:
        self._loader = loader or SkillLoader()
        self._index: dict[str, SkillDocument] = self._loader.load_index()
        self._cache: dict[str, SkillDocument] = {}

    def get(self, scene: str) -> SkillDocument | None:
        """按 scene 懒加载完整 Skill 文档。"""
        if scene in self._cache:
            return self._cache[scene]

        if scene not in self._index:
            return None

        doc = self._loader.load(scene)
        if doc is None:
            return None
        self._cache[scene] = doc
        return doc

    def get_index_prompt_section(self) -> str:
        """返回所有 Skill 的最小索引，供模型了解能力边界。"""
        if not self._index:
            return ""

        entries = "\n".join(doc.to_index_entry() for doc in self._index.values())
        return "# 业务 Skill 索引（摘要层）\n\n" + entries

    def get_prompt_section(self, scene: str, disclosure: str = "summary") -> str:
        """按 scene 和披露层级获取可注入 prompt 的文本段。"""
        if disclosure == "summary":
            doc = self._index.get(scene)
        else:
            doc = self.get(scene)
        if doc is None:
            return ""
        return doc.to_prompt_section(disclosure)

    @property
    def scenes(self) -> list[str]:
        """返回所有已注册 Skill 的 scene 列表。"""
        return list(self._index.keys())

    def reload(self) -> None:
        """重新加载 Skill 索引，并清空完整文档缓存。"""
        self._index = self._loader.load_index()
        self._cache.clear()
