from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from xinyidai_agent.protocol import RetrievalTrace, RouteDecision, SourceDocument


class EvidenceDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    ready: bool
    reason: str
    business_status: str
    user_message: str = ""
    source_titles: list[str] = Field(default_factory=list)


class EvidencePolicy:
    """RAG evidence boundary: evidence can support answers, but cannot be fabricated."""

    _evidence_required_scenes = {"KNOWLEDGE_QA"}

    def assess(
        self,
        route: RouteDecision,
        sources: list[SourceDocument],
        retrieval_trace: RetrievalTrace | None,
    ) -> EvidenceDecision:
        if route.scene in self._evidence_required_scenes and not sources:
            retriever_note = ""
            if retrieval_trace is not None:
                retriever_note = f" 检索结果数：{retrieval_trace.results_count}。"
            return EvidenceDecision(
                ready=False,
                reason="evidence_not_found",
                business_status="PARTIAL_DATA",
                user_message=f"未检索到依据，不能确认该政策、产品或准入结论。{retriever_note}".strip(),
            )

        return EvidenceDecision(
            ready=True,
            reason="evidence_ready" if sources else "evidence_not_required",
            business_status="RAG_RESULT_READY" if sources else "NO_TOOL_USED",
            source_titles=[source.title for source in sources],
        )

    def attach_citations(self, answer: str, sources: list[SourceDocument]) -> str:
        if not sources:
            return answer

        titles = []
        for source in sources:
            if source.title not in titles:
                titles.append(source.title)
        citation = "参考来源：" + "；".join(titles)
        if "参考来源" in answer:
            return answer
        return f"{answer}\n\n{citation}"
