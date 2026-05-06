from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from xinyidai_agent.memory.conversation import (
    ChatSessionRecord,
    ConversationAuditEvent,
    ConversationMessage,
    ConversationTranscript,
    ConversationVisibility,
    ConversationRole,
)
from xinyidai_agent.memory.session import SessionStore
from xinyidai_agent.protocol import AgentEvent, SessionStateSnapshot


SCHEMA_VERSION = 1


class PostgresSessionStore(SessionStore):
    """基于 PostgreSQL 的单窗口会话状态快照存储。"""

    def __init__(self, dsn: str, ensure_schema: bool = True) -> None:
        """创建状态存储，并按需初始化表结构。"""
        self._dsn = dsn
        if ensure_schema:
            PostgresConversationStore(dsn, ensure_schema=False).ensure_schema()

    def get(self, session_id: str) -> SessionStateSnapshot | None:
        """按会话 ID 读取最新状态快照。"""
        with _connect(self._dsn) as conn:
            row = conn.execute(
                "SELECT state FROM session_states WHERE session_id = %s",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return SessionStateSnapshot.model_validate(row["state"])

    def save(self, state: SessionStateSnapshot) -> None:
        """保存最新状态快照，并维护快照版本号。"""
        payload = state.model_dump(mode="json")
        with _connect(self._dsn) as conn:
            with conn.transaction():
                _ensure_session_row(conn, state.session_id, metadata=None)
                conn.execute(
                    """
                    INSERT INTO session_states (session_id, state, version, updated_at)
                    VALUES (%s, %s, 1, now())
                    ON CONFLICT (session_id) DO UPDATE
                    SET state = EXCLUDED.state,
                        version = session_states.version + 1,
                        updated_at = now()
                    """,
                    (state.session_id, Jsonb(payload)),
                )


class PostgresConversationStore:
    """基于 PostgreSQL 的会话记录和审计事件存储。"""

    def __init__(self, dsn: str, ensure_schema: bool = True) -> None:
        """创建会话记录存储，并按需初始化表结构。"""
        self._dsn = dsn
        if ensure_schema:
            self.ensure_schema()

    def ensure_schema(self) -> None:
        """幂等创建会话、消息、状态和审计事件表。"""
        with _connect(self._dsn) as conn:
            with conn.transaction():
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_schema_migrations (
                        name TEXT PRIMARY KEY,
                        version INTEGER NOT NULL,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        status TEXT NOT NULL DEFAULT 'active',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        message_count INTEGER NOT NULL DEFAULT 0,
                        last_message_at TIMESTAMPTZ,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id UUID PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                        sequence BIGINT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool', 'system_tool', 'diagnostic')),
                        content TEXT NOT NULL DEFAULT '',
                        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        visibility TEXT NOT NULL DEFAULT 'user' CHECK (visibility IN ('user', 'diagnostic')),
                        event_type TEXT,
                        request_id TEXT,
                        turn_id TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        UNIQUE (session_id, sequence)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_states (
                        session_id TEXT PRIMARY KEY REFERENCES chat_sessions(id) ON DELETE CASCADE,
                        state JSONB NOT NULL,
                        version BIGINT NOT NULL DEFAULT 1,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_audit_events (
                        id UUID PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                        sequence BIGINT NOT NULL,
                        event_type TEXT NOT NULL,
                        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        request_id TEXT,
                        turn_id TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        UNIQUE (session_id, sequence)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_sequence ON chat_messages(session_id, sequence)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_messages_request ON chat_messages(request_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated ON chat_sessions(updated_at DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_audit_session_sequence ON session_audit_events(session_id, sequence)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_audit_request ON session_audit_events(request_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_sessions_metadata_gin ON chat_sessions USING GIN (metadata)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_messages_payload_gin ON chat_messages USING GIN (payload)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_states_state_gin ON session_states USING GIN (state)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_audit_payload_gin ON session_audit_events USING GIN (payload)"
                )
                conn.execute(
                    """
                    INSERT INTO agent_schema_migrations (name, version)
                    VALUES ('session_persistence', %s)
                    ON CONFLICT (name) DO UPDATE
                    SET version = GREATEST(agent_schema_migrations.version, EXCLUDED.version),
                        applied_at = now()
                    """,
                    (SCHEMA_VERSION,),
                )

    def ensure_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> ChatSessionRecord:
        """确保会话元信息存在，并合并本次传入的元数据。"""
        with _connect(self._dsn) as conn:
            with conn.transaction():
                _ensure_session_row(conn, session_id, metadata)
                row = _get_session_row(conn, session_id)
        return _session_record(row)

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
        """追加聊天消息，并用会话级锁保证同一会话内序号连续。"""
        resolved_message_id = message_id or str(uuid4())
        with _connect(self._dsn) as conn:
            with conn.transaction():
                _ensure_session_row(conn, session_id, metadata=None)
                sequence = _next_sequence(conn, "chat_messages", session_id)
                row = conn.execute(
                    """
                    INSERT INTO chat_messages (
                        id, session_id, sequence, role, content, payload, visibility,
                        event_type, request_id, turn_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET content = EXCLUDED.content
                    RETURNING id::text, session_id, sequence, role, content, payload,
                              visibility, event_type, request_id, turn_id,
                              created_at::text
                    """,
                    (
                        resolved_message_id,
                        session_id,
                        sequence,
                        role,
                        content,
                        Jsonb(payload or {}),
                        visibility,
                        event_type,
                        request_id,
                        turn_id,
                    ),
                ).fetchone()
                conn.execute(
                    """
                    UPDATE chat_sessions
                    SET updated_at = now(),
                        last_message_at = now(),
                        message_count = (
                            SELECT count(*) FROM chat_messages WHERE session_id = %s
                        )
                    WHERE id = %s
                    """,
                    (session_id, session_id),
                )
        return _message(row)

    def append_audit_event(
        self,
        *,
        session_id: str,
        event: AgentEvent,
        request_id: str | None = None,
    ) -> ConversationAuditEvent:
        """追加运行审计事件，保留原始事件负载用于追溯。"""
        with _connect(self._dsn) as conn:
            with conn.transaction():
                _ensure_session_row(conn, session_id, metadata=None)
                sequence = _next_sequence(conn, "session_audit_events", session_id)
                row = conn.execute(
                    """
                    INSERT INTO session_audit_events (
                        id, session_id, sequence, event_type, payload, request_id, turn_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id::text, session_id, sequence, event_type, payload,
                              request_id, turn_id, created_at::text
                    """,
                    (
                        str(uuid4()),
                        session_id,
                        sequence,
                        event.event_type,
                        Jsonb(event.model_dump(mode="json")),
                        request_id,
                        event.turn_id,
                    ),
                ).fetchone()
        return _audit_event(row)

    def get_transcript(self, session_id: str, limit: int = 200) -> ConversationTranscript | None:
        """读取前端恢复聊天窗口需要的会话、消息和状态。"""
        with _connect(self._dsn) as conn:
            session_row = _get_session_row(conn, session_id)
            if session_row is None:
                return None
            state_row = conn.execute(
                "SELECT state FROM session_states WHERE session_id = %s",
                (session_id,),
            ).fetchone()
            rows = conn.execute(
                """
                SELECT id::text, session_id, sequence, role, content, payload,
                       visibility, event_type, request_id, turn_id, created_at::text
                FROM chat_messages
                WHERE session_id = %s AND visibility = 'user'
                ORDER BY sequence ASC
                LIMIT %s
                """,
                (session_id, limit),
            ).fetchall()
        session_state = SessionStateSnapshot.model_validate(state_row["state"]) if state_row else None
        return ConversationTranscript(
            session=_session_record(session_row),
            messages=[_message(row) for row in rows],
            session_state=session_state,
        )

    def list_sessions(self, limit: int = 50, offset: int = 0) -> list[ChatSessionRecord]:
        """按最近更新时间倒序分页读取会话列表。"""
        with _connect(self._dsn) as conn:
            rows = conn.execute(
                """
                SELECT id, title, status, created_at::text, updated_at::text,
                       message_count, last_message_at::text, metadata
                FROM chat_sessions
                ORDER BY updated_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            ).fetchall()
        return [_session_record(row) for row in rows]


@contextmanager
def _connect(dsn: str) -> Iterator[psycopg.Connection]:
    """创建一次短连接，并统一使用字典行格式。"""
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        yield conn


def _ensure_session_row(conn: psycopg.Connection, session_id: str, metadata: dict[str, Any] | None) -> None:
    """插入或更新会话基础行，避免消息和状态出现孤儿记录。"""
    conn.execute(
        """
        INSERT INTO chat_sessions (id, metadata)
        VALUES (%s, %s)
        ON CONFLICT (id) DO UPDATE
        SET updated_at = now(),
            metadata = chat_sessions.metadata || EXCLUDED.metadata
        """,
        (session_id, Jsonb(metadata or {})),
    )


def _get_session_row(conn: psycopg.Connection, session_id: str) -> dict[str, Any] | None:
    """读取单条会话基础行。"""
    return conn.execute(
        """
        SELECT id, title, status, created_at::text, updated_at::text,
               message_count, last_message_at::text, metadata
        FROM chat_sessions
        WHERE id = %s
        """,
        (session_id,),
    ).fetchone()


def _next_sequence(conn: psycopg.Connection, table_name: str, session_id: str) -> int:
    """分配会话内连续序号，事务级 advisory lock 防止并发冲突。"""
    conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"{table_name}:{session_id}",))
    row = conn.execute(
        f"SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM {table_name} WHERE session_id = %s",
        (session_id,),
    ).fetchone()
    return int(row["next_sequence"])


def _session_record(row: dict[str, Any]) -> ChatSessionRecord:
    """将数据库行转换为会话元信息模型。"""
    return ChatSessionRecord(
        session_id=row["id"],
        title=row["title"],
        status=row["status"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        message_count=int(row["message_count"]),
        last_message_at=str(row["last_message_at"]) if row["last_message_at"] else None,
        metadata=dict(row["metadata"] or {}),
    )


def _message(row: dict[str, Any]) -> ConversationMessage:
    """将数据库行转换为会话消息模型。"""
    return ConversationMessage(
        message_id=row["id"],
        session_id=row["session_id"],
        sequence=int(row["sequence"]),
        role=row["role"],
        content=row["content"],
        payload=dict(row["payload"] or {}),
        visibility=row["visibility"],
        event_type=row["event_type"],
        request_id=row["request_id"],
        turn_id=row["turn_id"],
        created_at=str(row["created_at"]),
    )


def _audit_event(row: dict[str, Any]) -> ConversationAuditEvent:
    """将数据库行转换为审计事件模型。"""
    return ConversationAuditEvent(
        event_id=row["id"],
        session_id=row["session_id"],
        sequence=int(row["sequence"]),
        event_type=row["event_type"],
        payload=dict(row["payload"] or {}),
        request_id=row["request_id"],
        turn_id=row["turn_id"],
        created_at=str(row["created_at"]),
    )
