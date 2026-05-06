from __future__ import annotations

from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from xinyidai_agent.config import AgentConfig, SessionStorageConfig, load_env_file
from xinyidai_agent.llm import OpenAICompatibleChatModel
from xinyidai_agent.memory import (
    ChatSessionRecord,
    ConversationTranscript,
    MemoryManager,
    PostgresConversationStore,
    PostgresSessionStore,
)
from xinyidai_agent.protocol import AgentEvent, ChatRequest, ChatResponse, ConfirmActionRequest
from xinyidai_agent.runtime import ControlledAgentLoop


def create_app(loop: ControlledAgentLoop | None = None) -> FastAPI:
    """创建 FastAPI 应用，并暴露聊天、确认和会话恢复接口。"""
    app = FastAPI(title="信易贷聊天 Agent", version="0.1.0")
    agent_loop = loop or _build_default_loop()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """健康检查接口。"""
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        """执行一轮普通聊天。"""
        return agent_loop.answer(request)

    @app.post("/chat/confirm", response_model=ChatResponse)
    def chat_confirm(request: ConfirmActionRequest) -> ChatResponse:
        """确认或取消上一轮生成的待确认业务动作。"""
        return agent_loop.confirm(request)

    @app.post("/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        """执行一轮流式聊天，并在流结束后记录完整事件。"""
        return StreamingResponse(
            _to_sse(agent_loop.run(request), agent_loop),
            media_type="text/event-stream",
        )

    @app.get("/sessions", response_model=list[ChatSessionRecord])
    def list_sessions(limit: int = 50, offset: int = 0) -> list[ChatSessionRecord]:
        """分页返回最近更新的会话，供网页端展示历史会话入口。"""
        store = getattr(agent_loop, "conversation_store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="session persistence is not configured")
        return store.list_sessions(limit=min(max(limit, 1), 200), offset=max(offset, 0))

    @app.get("/sessions/{session_id}", response_model=ConversationTranscript)
    def get_session(session_id: str, limit: int = 200) -> ConversationTranscript:
        """按会话 ID 恢复聊天消息和当前窗口状态。"""
        store = getattr(agent_loop, "conversation_store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="session persistence is not configured")
        transcript = store.get_transcript(session_id, limit=min(max(limit, 1), 500))
        if transcript is None:
            raise HTTPException(status_code=404, detail="session not found")
        return transcript

    return app


def _build_default_loop() -> ControlledAgentLoop:
    """按环境配置创建默认 Agent 循环和会话存储。"""
    load_env_file(AgentConfig.default_env_path())
    model = OpenAICompatibleChatModel(AgentConfig.from_env())
    storage = SessionStorageConfig.from_env()
    if storage.backend == "memory":
        return ControlledAgentLoop(model=model)
    if storage.backend != "postgres":
        raise RuntimeError(f"unsupported SESSION_STORAGE_BACKEND: {storage.backend}")
    conversation_store = PostgresConversationStore(storage.dsn)
    session_store = PostgresSessionStore(storage.dsn, ensure_schema=False)
    return ControlledAgentLoop(
        model=model,
        memory_manager=MemoryManager(session_store),
        conversation_store=conversation_store,
    )


def _to_sse(events: Iterator[AgentEvent], loop: ControlledAgentLoop | None = None) -> Iterator[str]:
    """把 Agent 事件转换成 Server-Sent Events 文本流。"""
    emitted: list[AgentEvent] = []
    for event in events:
        emitted.append(event)
        payload = event.model_dump_json()
        yield f"event: {event.event_type}\n"
        yield f"data: {payload}\n\n"
    if loop is not None and hasattr(loop, "record_events"):
        loop.record_events(emitted)
