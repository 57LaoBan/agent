from __future__ import annotations

import json
import re
from typing import Any, get_args

from pydantic import ValidationError

from xinyidai_agent.capabilities.base import CapabilityPolicy
from xinyidai_agent.capabilities.catalog import default_capability_catalog
from xinyidai_agent.llm import JSON_OBJECT_RESPONSE_FORMAT, ChatModel
from xinyidai_agent.protocol import ChatRequest, ModelRouteOutput, RouteDecision, StandardIntent
from xinyidai_agent.router.rules import unknown_route


MODEL_ROUTE_SCHEMA = ModelRouteOutput.model_json_schema()
MODEL_ROUTE_SCHEMA_TEXT = json.dumps(MODEL_ROUTE_SCHEMA, ensure_ascii=False)
STANDARD_INTENTS = set(get_args(StandardIntent))
INTENT_ALIASES = {
    "CREDIT_ELIGIBILITY_QUERY": "CREDIT_LIMIT_QUERY",
    "query_eligibility_criteria": "POLICY_OR_PRODUCT_QA",
    "generate_company_authorization_link": "CREATE_AUTHORIZATION_LINK",
}
SCENE_DEFAULT_INTENTS = {
    "KNOWLEDGE_QA": "POLICY_OR_PRODUCT_QA",
    "DATA_QUERY": "CREDIT_LIMIT_QUERY",
    "LOAN_APPLY": "CREATE_APPLICATION",
    "AUTHORIZATION": "CREATE_AUTHORIZATION_LINK",
    "APPLICATION_STATUS": "APPLICATION_STATUS_QUERY",
    "SMALLTALK": "SMALLTALK",
    "UNKNOWN": "UNKNOWN",
}


class ModelIntentRouter:
    """模型结构化意图识别器，只产出候选 RouteDecision。

    设计原则：模型只在 catalog 预设的 standard_intent 枚举里选一个，
    禁止自由生成；后端按 standard_intent 精确派生 capability，全链路可控。
    """

    def __init__(
        self,
        model: ChatModel,
        catalog: list[CapabilityPolicy] | None = None,
    ) -> None:
        self._model = model
        self._catalog = catalog or default_capability_catalog()
        self._intent_menu_text = self._render_intent_menu(self._catalog)

    def route(self, request: ChatRequest) -> RouteDecision:
        raw = self._model.complete(
            self._build_messages(request),
            response_format=JSON_OBJECT_RESPONSE_FORMAT,
        )
        try:
            payload = self._parse_json(raw)
            payload = self._normalize_model_payload(payload)
            model_output = ModelRouteOutput.model_validate(payload)
            return self._to_route_decision(model_output)
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
                    "你只负责语义理解，不负责工具授权。\n\n"
                    "## 输出字段约束\n"
                    "- scene：从下方枚举里选一个。\n"
                    "- standard_intent：**必须**从下方『可选标准意图清单』里选一个完全相同的字符串，禁止自创、禁止改写大小写。\n"
                    "- raw_intent：你对意图的简短描述，仅用于诊断日志，不参与路由决策；可写也可省略。\n"
                    "- confidence：0.0~1.0；明确的清晰意图必须 >= 0.85；含糊或乱码 <= 0.5。\n"
                    "- filled_slots：JSON object（没有槽位写 {}，不要写 []）。\n"
                    "- missing_slots：仅当 standard_intent=UNKNOWN 时写 [\"user_intent\"]，业务场景的必填字段由后端决定，不要自己发明。\n"
                    "- route_reason：一句话说明为什么这样判定。\n"
                    "- 不要输出 allowed_tools / allowed_tool_categories / risk_level / confirmation_required / capability_id；这些由后端从 standard_intent 派生。\n\n"
                    "## Scene 枚举\n"
                    "- KNOWLEDGE_QA：总结性 / 概念性 / 流程性的知识问答，答案是文字描述、解释、清单或步骤。\n"
                    "- DATA_QUERY：查询具体字段或数值，期望返回结构化数据（数字、状态值、具体取值）。\n"
                    "- LOAN_APPLY：发起贷款申请、创建申请草稿。\n"
                    "- AUTHORIZATION：生成企业授权链接（执行授权动作，不是问怎么办理）。\n"
                    "- APPLICATION_STATUS：查询申请办理进度。\n"
                    "- SMALLTALK：问候、闲聊、询问助手能力。\n"
                    "- UNKNOWN：意图不明确或无业务含义。\n\n"
                    "## 可选标准意图清单（standard_intent 只能取下面之一）\n"
                    f"{self._intent_menu_text}\n\n"
                    "## KNOWLEDGE_QA vs DATA_QUERY 边界（重点）\n"
                    "按答案形态判断：\n"
                    "- 答案是『一段说明 / 字典 / 步骤清单』 → KNOWLEDGE_QA + POLICY_OR_PRODUCT_QA\n"
                    "- 答案是『一个或几个具体数值/状态值』 → DATA_QUERY + (CREDIT_LIMIT_QUERY 或 PRODUCT_TERMS_QUERY)\n\n"
                    "## 决策示例\n"
                    "- '小微税贷的利率是多少？' → scene=DATA_QUERY, standard_intent=PRODUCT_TERMS_QUERY\n"
                    "- '小微税贷的利率是怎么算的？' → scene=KNOWLEDGE_QA, standard_intent=POLICY_OR_PRODUCT_QA\n"
                    "- '高风险标签都包括哪些情况？' → scene=KNOWLEDGE_QA, standard_intent=POLICY_OR_PRODUCT_QA\n"
                    "- 'A 企业当前的风险标签是什么？' → scene=DATA_QUERY, standard_intent=CREDIT_LIMIT_QUERY\n"
                    "- '查一下 A 公司的授信额度' → scene=DATA_QUERY, standard_intent=CREDIT_LIMIT_QUERY\n"
                    "- '怎么办理企业授权？' → scene=KNOWLEDGE_QA, standard_intent=POLICY_OR_PRODUCT_QA\n"
                    "- '帮我给 A 企业生成授权链接' → scene=AUTHORIZATION, standard_intent=CREATE_AUTHORIZATION_LINK\n"
                    "- '我想申请贷款' → scene=LOAN_APPLY, standard_intent=CREATE_APPLICATION\n"
                    "- '我的申请审批到哪一步了' → scene=APPLICATION_STATUS, standard_intent=APPLICATION_STATUS_QUERY\n"
                    "- '你好' → scene=SMALLTALK, standard_intent=SMALLTALK\n"
                    "- '？？？' / 纯数字 / 乱码 → scene=UNKNOWN, standard_intent=UNKNOWN, confidence<=0.5\n\n"
                    f"目标 JSON Schema：{MODEL_ROUTE_SCHEMA_TEXT}"
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

    @staticmethod
    def _render_intent_menu(catalog: list[CapabilityPolicy]) -> str:
        """把 catalog 中所有 standard_intent 渲染成给模型看的菜单。

        同一个 standard_intent 在 catalog 里可能不出现多次（当前一对一），
        但 _render 仍按 standard_intent 去重以防未来扩展。
        """
        seen: set[str] = set()
        lines: list[str] = []
        for policy in catalog:
            intent = policy.standard_intent
            if intent in seen:
                continue
            seen.add(intent)
            lines.append(
                f"- {intent}（scene={policy.scene}）：{policy.description}"
            )
        return "\n".join(lines)

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

    def _normalize_model_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """对模型返回的 payload 做输入兼容和输出收口。

        模型历史输出可能使用 intent 字段或夹带 allowed_tools 等未授权字段；
        这里先转换为后端标准 intent，再交给 Pydantic 严格校验，保证后续只消费
        schema 内字段。
        """
        raw_intent = payload.get("raw_intent")
        candidate_intent = payload.get("standard_intent") or payload.get("intent")
        standard_intent = self._standardize_intent(payload.get("scene"), candidate_intent)
        if raw_intent is None and candidate_intent and candidate_intent != standard_intent:
            raw_intent = str(candidate_intent)

        normalized: dict[str, Any] = {
            "scene": payload.get("scene"),
            "standard_intent": standard_intent,
            "raw_intent": raw_intent,
            "confidence": payload.get("confidence"),
            "filled_slots": payload.get("filled_slots", {}),
            "missing_slots": payload.get("missing_slots", []),
            "route_reason": payload.get("route_reason"),
        }
        if normalized["filled_slots"] == []:
            normalized["filled_slots"] = {}
        if normalized["missing_slots"] == {}:
            normalized["missing_slots"] = []
        return normalized

    def _standardize_intent(self, scene: Any, candidate_intent: Any) -> str:
        """把模型自由 intent 映射为后端 StandardIntent。"""
        if candidate_intent in STANDARD_INTENTS:
            return str(candidate_intent)

        alias = INTENT_ALIASES.get(str(candidate_intent))
        if alias:
            return alias

        scene_default = SCENE_DEFAULT_INTENTS.get(str(scene))
        if scene_default:
            return scene_default

        return "UNKNOWN"

    def _to_route_decision(self, output: ModelRouteOutput) -> RouteDecision:
        """模型输出转 RouteDecision；intent 字段统一用 standard_intent。"""
        return RouteDecision(
            scene=output.scene,
            intent=output.standard_intent,
            raw_intent=output.raw_intent,
            confidence=output.confidence,
            filled_slots=output.filled_slots,
            missing_slots=output.missing_slots,
            route_reason=output.route_reason,
            route_source="model",
        )
