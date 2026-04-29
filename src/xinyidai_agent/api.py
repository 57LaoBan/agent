from __future__ import annotations

from collections.abc import Iterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from xinyidai_agent.config import AgentConfig, load_env_file
from xinyidai_agent.llm import OpenAICompatibleChatModel
from xinyidai_agent.protocol import AgentEvent, ChatRequest, ChatResponse
from xinyidai_agent.runtime import ControlledAgentLoop


def create_app(loop: ControlledAgentLoop | None = None) -> FastAPI:
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
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        return agent_loop.answer(request)

    @app.post("/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        return StreamingResponse(
            _to_sse(agent_loop.run(request)),
            media_type="text/event-stream",
        )

    return app


def _build_default_loop() -> ControlledAgentLoop:
    load_env_file(AgentConfig.default_env_path())
    return ControlledAgentLoop(model=OpenAICompatibleChatModel(AgentConfig.from_env()))


def _to_sse(events: Iterator[AgentEvent]) -> Iterator[str]:
    for event in events:
        payload = event.model_dump_json()
        yield f"event: {event.event_type}\n"
        yield f"data: {payload}\n\n"
