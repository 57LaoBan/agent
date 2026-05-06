from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from xinyidai_agent.protocol import AgentEvent, SessionStateSnapshot


ConversationRole = Literal["user", "assistant", "tool", "system_tool", "diagnostic"]
ConversationVisibility = Literal["user", "diagnostic"]


class ChatSessionRecord(BaseModel):
    """会话列表和会话详情页使用的会话元信息。"""

    model_config = ConfigDict(frozen=True)

    session_id: str
    title: str | None = None
    status: str = "active"
    created_at: str
    updated_at: str
    message_count: int = 0
    last_message_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    """前端聊天窗口可恢复展示的一条会话消息。"""

    model_config = ConfigDict(frozen=True)

    message_id: str
    session_id: str
    sequence: int
    role: ConversationRole
    content: str
    payload: dict[str, Any] = Field(default_factory=dict)
    visibility: ConversationVisibility = "user"
    event_type: str | None = None
    request_id: str | None = None
    turn_id: str | None = None
    created_at: str


class ConversationAuditEvent(BaseModel):
    """用于审计追溯的单条运行事件记录。"""

    model_config = ConfigDict(frozen=True)

    event_id: str
    session_id: str
    sequence: int
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None
    turn_id: str | None = None
    created_at: str


class ConversationTranscript(BaseModel):
    """从持久化存储恢复出的完整会话视图。"""

    model_config = ConfigDict(frozen=True)

    session: ChatSessionRecord
    messages: list[ConversationMessage] = Field(default_factory=list)
    session_state: SessionStateSnapshot | None = None


class ConversationStore(Protocol):
    """会话记录持久化存储接口。"""

    def ensure_schema(self) -> None:
        """初始化或升级会话记录相关表结构。"""
        ...

    def ensure_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> ChatSessionRecord:
        """确保指定会话存在，并返回最新会话元信息。"""
        ...

    def append_message(
        self,
        *,
        session_id: str,
        role: ConversationRole,
        content: str,
        payload: dict[str, Any] | None = None,
        visibility: ConversationVisibility = "user",
        event_type: str | None = None,
        request_id: str | None = None,
        turn_id: str | None = None,
        message_id: str | None = None,
    ) -> ConversationMessage:
        """追加一条面向用户或诊断用途的会话消息。"""
        ...

    def append_audit_event(
        self,
        *,
        session_id: str,
        event: AgentEvent,
        request_id: str | None = None,
    ) -> ConversationAuditEvent:
        """追加一条不可用于前端展示、但可用于审计追溯的运行事件。"""
        ...

    def get_transcript(self, session_id: str, limit: int = 200) -> ConversationTranscript | None:
        """按顺序读取会话消息和最新状态，用于网页端恢复聊天窗口。"""
        ...

    def list_sessions(self, limit: int = 50, offset: int = 0) -> list[ChatSessionRecord]:
        """分页读取最近更新的会话列表。"""
        ...
