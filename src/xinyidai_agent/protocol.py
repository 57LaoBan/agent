from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SourceType = Literal["policy", "product", "rule", "web", "internal", "unknown"]
EventVisibility = Literal["user", "diagnostic"]
PROTOCOL_VERSION = "2026-04-29"
AgentEventType = Literal[
    "turn_started",
    "state_changed",
    "route_started",
    "route_decision",
    "tool_call_proposed",
    "confirmation_required",
    "tool_started",
    "tool_result",
    "assistant_delta",
    "final_answer",
    "turn_finished",
    "error",
]
AgentState = Literal[
    "LISTENING",
    "ROUTING",
    "CLARIFYING",
    "PLANNING_TOOL",
    "WAITING_CONFIRMATION",
    "EXECUTING_TOOL",
    "OBSERVING_RESULT",
    "ANSWERING",
    "FINISHED",
    "REJECTED",
]
RouteScene = Literal[
    "KNOWLEDGE_QA",
    "DATA_QUERY",
    "LOAN_APPLY",
    "AUTHORIZATION",
    "APPLICATION_STATUS",
    "SMALLTALK",
    "UNKNOWN",
]
RiskLevel = Literal["read_only", "link_create", "state_create", "state_update", "final_submit"]
ToolStatus = Literal["proposed", "running", "success", "failed", "blocked"]
ToolCategory = Literal["knowledge", "data_query", "application", "authorization", "status", "utility"]
SlotName = str
BusinessStatus = Literal[
    "NO_TOOL_USED",
    "OK",
    "AUTH_REQUIRED",
    "PARTIAL_DATA",
    "NOT_FOUND",
    "RETRY_LATER",
    "PARTIAL_ERROR",
    "ERROR",
    "RAG_RESULT_READY",
    "CREDIT_AMOUNT_FOUND",
    "CREDIT_AMOUNT_UNAVAILABLE",
    "APPLICATION_DRAFT_CREATED",
    "APPLICATION_SUBMITTED",
    "TOOL_BLOCKED",
    "TOOL_SCHEMA_ERROR",
]
ActionType = Literal["business_action", "open_url", "open_miniprogram", "contact_service"]
NextStepType = Literal["none", "suggest_tool", "ask_user", "stop_with_action"]
DecisionAction = Literal["finish", "continue_tool", "ask_user", "reject"]


class AuditInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str | None = None
    session_id: str | None = None
    action_id: str | None = None
    idempotency_key: str | None = None
    risk_level: RiskLevel = "read_only"
    policy_decision: str = "allow"
    reason: str = ""


class AgentAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: ActionType
    name: str
    label: str
    risk_level: RiskLevel = "read_only"
    confirmation_required: bool = False
    reason: str | None = None
    url: str | None = None
    payload: dict[str, Any] | None = None
    expires_in_seconds: int | None = None
    audit: AuditInfo | None = None


class ToolNextStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: NextStepType = "none"
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    reason: str | None = None
    allowed_tool_categories: list[ToolCategory] = Field(default_factory=list)
    risk_level: RiskLevel = "read_only"


class EvidenceState(BaseModel):
    model_config = ConfigDict(frozen=True)

    ready: bool = False
    reason: str = "unknown"
    signals: dict[str, Any] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)


class ModelDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    should_finish: bool = False
    next_action: DecisionAction = "ask_user"
    confidence: float = 0.0
    reason: str = "unknown"
    missing_info: list[str] = Field(default_factory=list)
    ask_user_question: str = ""
    source: str = "runtime"
    task_stage: str | None = None
    attempt: int | None = None


class PerformanceStage(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    duration_ms: float = 0.0
    percentage: float = 0.0
    start_offset_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PerformanceSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_ms: float = 0.0
    stages: list[PerformanceStage] = Field(default_factory=list)


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


class RouteDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    scene: RouteScene
    intent: str
    confidence: float
    required_slots: list[str] = Field(default_factory=list)
    filled_slots: dict[str, Any] = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    allowed_tool_categories: list[ToolCategory] = Field(default_factory=list)
    risk_level: RiskLevel = "read_only"
    route_reason: str
    route_source: str = "unknown"
    should_call_model: bool = True
    should_call_tool: bool = False


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    tool_category: ToolCategory | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel = "read_only"
    confirmation_required: bool = False
    reason: str = ""


class ActionLink(BaseModel):
    model_config = ConfigDict(frozen=True)

    link_type: Literal["apply", "authorization", "status", "other"]
    url: str
    label: str
    expires_in_seconds: int | None = None


class ToolResultEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    status: BusinessStatus
    code: int | None = None
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    next_step: ToolNextStep | None = None
    actions: list[AgentAction] = Field(default_factory=list)
    audit: AuditInfo | None = None


class PendingAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    action_id: str
    tool_call: ToolCall
    title: str
    summary: str
    confirm_label: str = "确认"
    cancel_label: str = "取消"
    details: list[dict[str, str]] = Field(default_factory=list)


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    tool_category: ToolCategory | None = None
    status: ToolStatus
    output: dict[str, Any] = Field(default_factory=dict)
    envelope: ToolResultEnvelope | None = None
    error_message: str | None = None
    business_status: str | None = None
    code: int | None = None
    message: str | None = None
    next_required_action: str | None = None
    next_step: ToolNextStep | None = None
    confirmation_required: bool = False
    terminal: bool = False
    allowed_next_tools: list[str] = Field(default_factory=list)
    action_links: list[ActionLink] = Field(default_factory=list)
    actions: list[AgentAction] = Field(default_factory=list)
    audit: AuditInfo | None = None
    model_observation: str = ""
    user_visible_message: str | None = None


class StateTransition(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_state: AgentState
    to_state: AgentState
    reason: str


class AgentEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    turn_id: str
    sequence: int
    event_type: AgentEventType
    visibility: EventVisibility
    timestamp: str
    payload: dict[str, Any] = Field(default_factory=dict)


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

    protocol_version: str = PROTOCOL_VERSION
    answer: str
    stop_reason: str | None = None
    business_status: str = "NO_TOOL_USED"
    task_stage: str = "understand"
    sources: list[SourceDocument] = Field(default_factory=list)
    route_decision: RouteDecision | None = None
    retrieval_trace: RetrievalTrace | None = None
    tool_trace: list[ToolResult] = Field(default_factory=list)
    pending_action: PendingAction | None = None
    evidence: EvidenceState = Field(default_factory=EvidenceState)
    model_decision: ModelDecision = Field(default_factory=ModelDecision)
    performance: PerformanceSummary = Field(default_factory=PerformanceSummary)
    actions: list[AgentAction] = Field(default_factory=list)
    events: list[AgentEvent] = Field(default_factory=list)
    diagnostics: list[DiagnosticEvent] = Field(default_factory=list)
    error: str | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
