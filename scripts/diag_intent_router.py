"""诊断意图识别：直接调用 ModelIntentRouter 观察模型对真实问题的判定。

用于定位 KNOWLEDGE_QA 类问题被路由到 unknown.clarify 的真因：
- scene 是否被正确识别？
- confidence 是否 >= 0.7？
- raw_intent 是否在 catalog 别名中？
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xinyidai_agent.config import AgentConfig, load_env_file  # noqa: E402
from xinyidai_agent.llm import OpenAICompatibleChatModel  # noqa: E402
from xinyidai_agent.protocol import ChatRequest  # noqa: E402
from xinyidai_agent.router.model_router import ModelIntentRouter  # noqa: E402
from xinyidai_agent.router.policy import RoutePolicy  # noqa: E402


QUERIES = [
    # === 期望走 DATA_QUERY -> product.terms.read（工具未注册，应返回降级提示） ===
    "小微税贷的利率是多少？",
    "小微税贷的额度上限是多少",
    "信易贷产品的期限多长",
    # === 期望走 DATA_QUERY -> credit.limit.read（需要 company_name） ===
    "查一下 ABC 公司的授信额度",
    # === 期望走 KNOWLEDGE_QA -> rag_search ===
    "信易贷的准入条件是什么",
    "高风险标签都包括哪些情况",
    "小微税贷的利率是怎么算的",
    "怎么办理企业授权",
    # === 期望走 AUTHORIZATION ===
    "帮我给 ABC 公司生成授权链接",
]


def main() -> None:
    """对每条 query 输出原始模型路由 + RoutePolicy 收口结果。"""
    load_env_file(AgentConfig.default_env_path())
    model = OpenAICompatibleChatModel(AgentConfig.from_env())
    router = ModelIntentRouter(model)
    policy = RoutePolicy()

    for query in QUERIES:
        request = ChatRequest(session_id="diag", user_message=query)
        raw = router.route(request)
        final = policy.apply(request, raw)
        print(f"\n>>> {query}")
        print(
            f"  raw : scene={raw.scene} intent={raw.raw_intent} "
            f"confidence={raw.confidence:.2f} reason={raw.route_reason}"
        )
        print(
            f"  final: capability={final.capability_id} scene={final.scene} "
            f"intent={final.intent} allowed_tools={final.allowed_tools} "
            f"should_call_tool={final.should_call_tool}"
        )


if __name__ == "__main__":
    main()
