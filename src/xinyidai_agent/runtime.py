from __future__ import annotations

from xinyidai_agent.protocol import ChatRequest, ChatResponse, DiagnosticEvent, SourceDocument
from xinyidai_agent.llm import ChatModel
from xinyidai_agent.rag import EmptyRetriever, Retriever


class ControlledAgentLoop:
    """单 Agent 受控问答流程。

    第一阶段只允许一次检索和一次模型调用，避免模型自驱动工具循环失控。
    """

    def __init__(self, model: ChatModel, retriever: Retriever | None = None) -> None:
        self._model = model
        self._retriever = retriever or EmptyRetriever()

    def answer(self, request: ChatRequest) -> ChatResponse:
        sources, retrieval_trace = self._retriever.retrieve(request.user_message, request.top_k)
        messages = self._build_messages(request.user_message, sources)
        answer = self._model.complete(messages)

        return ChatResponse(
            answer=answer,
            sources=sources,
            retrieval_trace=retrieval_trace,
            diagnostics=[
                DiagnosticEvent(
                    name="controlled_loop",
                    detail={
                        "llm_calls": 1,
                        "retrieval_calls": 1,
                        "mode": "single_agent_rag_first",
                    },
                )
            ],
        )

    def _build_messages(self, user_message: str, sources: list[SourceDocument]) -> list[dict[str, str]]:
        source_text = self._format_sources(sources)
        return [
            {
                "role": "system",
                "content": (
                    "你是信易贷聊天助手。回答必须基于可用证据；"
                    "证据不足时要说明缺口，不要编造政策、产品或准入规则。"
                ),
            },
            {
                "role": "user",
                "content": f"用户问题：{user_message}\n\n可用证据：\n{source_text}",
            },
        ]

    def _format_sources(self, sources: list[SourceDocument]) -> str:
        if not sources:
            return "暂无检索证据。"

        return "\n\n".join(
            f"[{index}] {source.title}\n{source.content}"
            for index, source in enumerate(sources, start=1)
        )
