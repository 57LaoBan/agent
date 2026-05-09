"""构造 ReAct 单步的 system 和 user prompt。"""

from __future__ import annotations

import json
from typing import Any

from xinyidai_agent.protocol import ToolExecutionObservation
from xinyidai_agent.react_observation import ReactObservation

REACT_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "thought": {"type": "string", "maxLength": 500},
        "action": {
            "type": "object",
            "properties": {
                "type": {"enum": ["call_tool", "answer", "ask_user", "handoff"]},
                "tool_name": {"type": ["string", "null"]},
                "arguments": {"type": "object"},
                "final_answer": {"type": ["string", "null"]},
                "evidence_used": {"type": "array", "items": {"type": "string"}},
                "message": {"type": ["string", "null"]},
            },
            "required": ["type"],
            "additionalProperties": False,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["thought", "action", "confidence"],
    "additionalProperties": False,
}


class ReactPromptBuilder:
    """ReAct 单步 prompt 构造器。"""

    def build_messages(self, observation: ReactObservation) -> list[dict[str, str]]:
        """构造模型调用消息。"""
        return [
            {"role": "system", "content": self._build_system_prompt(observation)},
            {"role": "user", "content": self._build_user_prompt(observation)},
        ]

    def _build_system_prompt(self, observation: ReactObservation) -> str:
        """构造包含 JSON schema 和安全约束的 system prompt。"""
        return (
            "# 你的角色\n"
            "你是信易贷业务智能助手的 ReAct 推理器。每一轮你只能输出一个 action，"
            "且 action.type 必须是 call_tool / answer / ask_user / handoff 之一。\n\n"
            "# 严格约束\n"
            f"- 当前能力 {observation.capability.capability_id}：{observation.capability.description}\n"
            f"- 当前已执行 {len(observation.executed_tools)} 次工具，最大允许 {observation.max_steps} 步。\n"
            "- 仅能调用 available_tools 中列出的工具。\n"
            "- 同一 (tool_name, arguments) 已经执行过 2 次以上时，禁止再次调用，请改用 answer / ask_user / handoff。\n"
            f"- 当前能力 requires_evidence={observation.capability.requires_evidence}：为 True 时 answer 必须基于工具结果且填 evidence_used。\n"
            "- final_answer 必须基于已成功的工具结果或闲聊场景，禁止编造数据、链接、利率、额度。\n"
            "- 当所需工具未注册时，优先选择 alternative_tools；否则明确告知该能力未上线或转人工。\n"
            "- 风险动作由系统转 pending_action 确认，你只需正常输出 call_tool。\n\n"
            "# 输出格式\n"
            "只输出 JSON 对象，结构必须严格符合：\n"
            f"{json.dumps(REACT_ACTION_SCHEMA, ensure_ascii=False, indent=2)}\n"
            "禁止输出任何 JSON 之外的文字、注释、Markdown 包裹。\n"
        )

    def _build_user_prompt(self, observation: ReactObservation) -> str:
        """构造包含用户问题、路由、工具和历史观察的 user prompt。"""
        parts: list[str] = [
            f"# 用户问题\n{observation.request.user_message}\n",
            (
                "# 当前路由\n"
                f"scene={observation.route.scene}, capability={observation.capability.capability_id}, "
                f"confidence={observation.route.confidence:.2f}\n"
            ),
            f"# 已填槽位\n{json.dumps(dict(observation.route.filled_slots), ensure_ascii=False)}\n",
            "# 可用工具\n" + self._render_tools(observation),
        ]
        if observation.route.missing_slots:
            parts.append(f"# 缺失槽位\n{observation.route.missing_slots}\n")
        if observation.executed_tools:
            parts.append("# 已执行工具历史\n" + self._render_executed(observation.executed_tools))
        if observation.last_observation is not None:
            parts.append("# 你上一步的执行观察\n" + self._render_observation(observation.last_observation))
        if observation.invalid_action_hint:
            parts.append(f"# 上次输出被规则拒绝\n{observation.invalid_action_hint}\n请重新选择 action。")
        parts.append(f"# 当前步数\nstep={observation.step}/{observation.max_steps}")
        return "\n".join(parts)

    @staticmethod
    def _render_tools(observation: ReactObservation) -> str:
        """渲染可用工具列表。"""
        lines = []
        for tool in observation.available_tools:
            lines.append(
                f"- `{tool.name}`（risk={tool.risk_level}, read_only={tool.is_read_only}, "
                f"已执行={tool.already_executed_count} 次）：{tool.description}"
                f"\n  required_arguments: {tool.required_arguments}"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_executed(executed: list[ToolExecutionObservation]) -> str:
        """渲染已执行工具观察。"""
        lines = []
        for index, observation in enumerate(executed, 1):
            arguments = json.dumps(dict(observation.arguments), ensure_ascii=False)[:120]
            detail = observation.summary or observation.reason or ""
            lines.append(f"{index}. [{observation.kind}] {observation.tool_name}({arguments}) → {detail[:200]}")
        return "\n".join(lines)

    @staticmethod
    def _render_observation(observation: ToolExecutionObservation) -> str:
        """渲染最近一次 observation。"""
        return (
            f"kind={observation.kind}\n"
            f"tool={observation.tool_name}\n"
            f"summary={observation.summary}\n"
            f"reason={observation.reason}\n"
            f"hint={observation.hint}\n"
            f"alternative_tools={observation.alternative_tools}\n"
        )
