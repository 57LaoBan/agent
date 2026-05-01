"""业务 Skill 层：将客户经理的业务处理流程沉淀为结构化文档，注入模型 prompt。"""

from xinyidai_agent.skills.loader import SkillLoader, SkillDocument
from xinyidai_agent.skills.registry import SkillRegistry

__all__ = ["SkillLoader", "SkillDocument", "SkillRegistry"]
