from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SourceType = Literal["policy", "product", "rule", "web", "internal", "unknown"]


class SourceDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    title: str
    content: str
    source_type: SourceType = "unknown"
    score: float | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    top_k: int
    results_count: int
    rerank_applied: bool = False
    steps: list[dict[str, Any]] = Field(default_factory=list)


class DiagnosticEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_message: str
    session_id: str | None = None
    top_k: int = 5
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    sources: list[SourceDocument] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace | None = None
    diagnostics: list[DiagnosticEvent] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
