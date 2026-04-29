from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from xinyidai_agent.llm import ChatModel
from xinyidai_agent.protocol import ChatRequest, RouteDecision
from xinyidai_agent.router.rules import unknown_route


class ModelIntentRouter:
    """模型结构化意图识别器，只产出候选 RouteDecision。"""

    def __init__(self, model: ChatModel) -> None:
        self._model = model

    def route(self, request: ChatRequest) -> RouteDecision:
        raw = self._model.complete(self._build_messages(request))
        try:
            payload = self._parse_json(raw)
            payload = self._normalize_payload(payload)
            return RouteDecision.model_validate(payload)
        except (ValueError, TypeError, ValidationError) as exc:
            return unknown_route(
                request,
                reason=f"模型路由结果无法解析，已转入追问确认，不执行工具：{exc}",
            )

    def _build_messages(self, request: ChatRequest) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "你是信易贷业务意图识别器，只输出 JSON，不要输出解释文字。"
                    "你需要识别 scene、intent、confidence、filled_slots、missing_slots、"
                    "risk_level、allowed_tools、allowed_tool_categories、route_reason、"
                    "should_call_model、should_call_tool。"
                    "scene 只能是 KNOWLEDGE_QA、DATA_QUERY、LOAN_APPLY、AUTHORIZATION、"
                    "APPLICATION_STATUS、SMALLTALK、UNKNOWN。"
                    "risk_level 只能是 read_only、link_create、state_create、state_update、final_submit。"
                    "工具分类只能是 knowledge、data_query、application、authorization、status、utility。"
                    "金融业务中，创建申请、提交申请、生成授权链接等动作必须准确标记风险。"
                    "如果用户输入只是数字、乱码、无业务含义短句，scene 必须返回 UNKNOWN，"
                    "confidence 不得超过 0.5，should_call_tool 必须为 false，missing_slots 写入 [\"user_intent\"]。"
                    "filled_slots 必须是 JSON object；没有槽位时返回 {}，不能返回 []。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户输入：{request.user_message}\n"
                    f"已有上下文 metadata：{json.dumps(request.metadata, ensure_ascii=False)}"
                ),
            },
        ]

    def _parse_json(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("模型未返回 JSON 对象")

        return json.loads(text[start : end + 1])

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        normalized["route_source"] = "model"
        if normalized.get("filled_slots") == []:
            normalized["filled_slots"] = {}
        if normalized.get("missing_slots") == {}:
            normalized["missing_slots"] = []
        if normalized.get("required_slots") == {}:
            normalized["required_slots"] = []
        if normalized.get("allowed_tools") == {}:
            normalized["allowed_tools"] = []
        if normalized.get("allowed_tool_categories") == {}:
            normalized["allowed_tool_categories"] = []
        return normalized
