from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from xinyidai_agent.config import AgentConfig, SessionStorageConfig, load_env_file
from xinyidai_agent.capabilities.catalog import CapabilityCatalog, default_capability_catalog
from xinyidai_agent.evidence_policy import EvidencePolicy
from xinyidai_agent.llm import OpenAICompatibleChatModel
from xinyidai_agent.memory import (
    ChatSessionRecord,
    ConversationTranscript,
    MemoryManager,
    PostgresConversationStore,
    PostgresSessionStore,
)
from xinyidai_agent.protocol import AgentEvent, ChatRequest, ChatResponse, ConfirmActionRequest
from xinyidai_agent.rag import (
    EmptyRetriever,
    ProductionRAGRuntime,
    Retriever,
    build_production_rag_runtime,
)
from xinyidai_agent.router import ControlledIntentRouter
from xinyidai_agent.router.rules_guard import RulesGuard
from xinyidai_agent.router.scene_direct import SceneDirectDispatcher
from xinyidai_agent.runtime import ControlledAgentLoop
from xinyidai_agent.runtime.fast_path import FastPathRunner
from xinyidai_agent.runtime.react_engine import ReactStepEngine
from xinyidai_agent.runtime.react_loop import ReactRuntime
from xinyidai_agent.runtime.react_observation_builder import ReactObservationBuilder
from xinyidai_agent.tools.registry import default_tool_registry


_LOGGER = logging.getLogger(__name__)


def create_app(loop: ControlledAgentLoop | None = None) -> FastAPI:
    """创建 FastAPI 应用，并暴露聊天、确认和会话恢复接口。"""
    # Windows 上 psycopg 异步驱动与默认 ProactorEventLoop 不兼容，需在进程初始阶段切换。
    if sys.platform.startswith("win"):
        import asyncio
        if not isinstance(
            asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy
        ):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    rag_runtime: ProductionRAGRuntime | None = None
    if loop is None:
        agent_loop, rag_runtime = _build_default_loop_with_runtime()
    else:
        agent_loop = loop

    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        """FastAPI 生命周期：启动不做额外动作，关闭时释放 RAG 资源。"""
        try:
            yield
        finally:
            if rag_runtime is not None:
                _LOGGER.info("关闭 RAG 运行时资源")
                rag_runtime.close()

    app = FastAPI(title="信易贷聊天 Agent", version="0.1.0", lifespan=_lifespan)
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
    """保留原接口以兼容外部调用者；内部委托给 _build_default_loop_with_runtime。"""
    agent_loop, _ = _build_default_loop_with_runtime()
    return agent_loop


def _build_default_loop_with_runtime() -> tuple[ControlledAgentLoop, ProductionRAGRuntime | None]:
    """按环境配置创建 Agent 循环、会话存储与可选的生产 RAG 运行时。

    RAG_ENABLED 默认开启；设为 false 则回退到 EmptyRetriever，保证未配置数据库时仍可起服。
    """
    load_env_file(AgentConfig.default_env_path())
    model = OpenAICompatibleChatModel(AgentConfig.from_env())

    rag_enabled = _env_flag("RAG_ENABLED", default=True)
    rag_runtime: ProductionRAGRuntime | None = None
    retriever: Retriever
    if rag_enabled:
        try:
            rag_runtime = build_production_rag_runtime()
            retriever = rag_runtime.retriever
            _LOGGER.info(
                "生产 RAG 检索器已就绪：prefix=%s，version=%s",
                rag_runtime.store.table_prefix,
                rag_runtime.retriever._index_version if hasattr(rag_runtime.retriever, "_index_version") else "unknown",
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("生产 RAG 检索器初始化失败，回退到 EmptyRetriever：%s", exc)
            rag_runtime = None
            retriever = EmptyRetriever()
    else:
        retriever = EmptyRetriever()
        _LOGGER.warning("RAG_ENABLED=false，使用 EmptyRetriever 占位。")

    tool_registry = default_tool_registry(retriever=retriever)

    capability_catalog = CapabilityCatalog(default_capability_catalog())
    router = ControlledIntentRouter(
        model=model,
        guard=RulesGuard(),
        scene_direct=SceneDirectDispatcher(capability_catalog),
    )
    memory_manager: MemoryManager
    conversation_store = None

    storage = SessionStorageConfig.from_env()
    if storage.backend == "memory":
        memory_manager = MemoryManager()
    elif storage.backend == "postgres":
        conversation_store = PostgresConversationStore(storage.dsn)
        session_store = PostgresSessionStore(storage.dsn, ensure_schema=False)
        memory_manager = MemoryManager(session_store)
    else:
        if rag_runtime is not None:
            rag_runtime.close()
        raise RuntimeError(f"unsupported SESSION_STORAGE_BACKEND: {storage.backend}")

    react_runtime = ReactRuntime(
        model=model,
        router=router,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        capability_catalog=capability_catalog,
        fast_path=FastPathRunner(tool_registry, model),
        react_engine=ReactStepEngine(model, tool_registry),
        observation_builder=ReactObservationBuilder(tool_registry),
        evidence_policy=EvidencePolicy(),
    )
    agent_loop = ControlledAgentLoop(
        model=model,
        router=router,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        conversation_store=conversation_store,
        capability_catalog=capability_catalog,
        react_runtime=react_runtime,
    )

    return agent_loop, rag_runtime


def _env_flag(name: str, default: bool) -> bool:
    """读取布尔开关，兼容 1/0/true/false/yes/no 等常见写法。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


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
