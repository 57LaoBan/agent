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
                    "你只负责语义理解，不负责工具授权。"
                    "你需要识别 scene、raw_intent、confidence、filled_slots、missing_slots、route_reason。\n\n"
                    "## Scene 分类规则（必须严格遵守）\n"
                    "- KNOWLEDGE_QA：咨询政策、产品介绍、准入条件、利率等知识性问题\n"
                    "- DATA_QUERY：查询企业授信额度、可贷额度等数据（关键词：查、额度、能贷多少）\n"
                    "- LOAN_APPLY：发起贷款申请、创建申请草稿（关键词：申请、办理、要贷款）\n"
                    "- AUTHORIZATION：生成企业授权链接（关键词：授权、授权链接）\n"
                    "- APPLICATION_STATUS：查询申请办理进度（关键词：进度、状态、办到哪了）\n"
                    "- SMALLTALK：问候、闲聊、询问助手能力\n"
                    "- UNKNOWN：意图不明确或无业务含义\n\n"
                    "## 重要区分\n"
                    "- '查额度' → DATA_QUERY（只查数据，不创建申请）\n"
                    "- '申请贷款' → LOAN_APPLY（创建申请草稿）\n"
                    "- '有什么产品' → KNOWLEDGE_QA（咨询知识）\n\n"
                    "raw_intent 是你对用户意图的原始短标签，例如 query_credit_limit。"
                    "missing_slots 只用于 UNKNOWN 场景写入 [\"user_intent\"]；"
                    "业务场景的必填字段由后端能力策略决定，不要发明 loan_purpose、monthly_income、credit_score 等字段。"
                    "不要输出 allowed_tools、allowed_tool_categories、risk_level、confirmation_required，"
                    "这些由后端能力策略决定。"
                    "如果用户输入只是数字、乱码、无业务含义短句，scene 必须返回 UNKNOWN，"
                    "confidence 不得超过 0.5，missing_slots 写入 [\"user_intent\"]。"
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
        if not normalized.get("raw_intent") and normalized.get("intent"):
            normalized["raw_intent"] = normalized["intent"]
        if not normalized.get("intent"):
            normalized["intent"] = normalized.get("raw_intent") or "UNKNOWN"
        normalized.pop("allowed_tools", None)
        normalized.pop("allowed_tool_categories", None)
        normalized.pop("risk_level", None)
        normalized.pop("confirmation_required", None)
        if normalized.get("filled_slots") == []:
            normalized["filled_slots"] = {}
        if normalized.get("missing_slots") == {}:
            normalized["missing_slots"] = []
        if normalized.get("required_slots") == {}:
            normalized["required_slots"] = []
        return normalized
