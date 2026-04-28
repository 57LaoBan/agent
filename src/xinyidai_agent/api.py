from __future__ import annotations

from fastapi import FastAPI

from xinyidai_agent.config import AgentConfig, load_env_file
from xinyidai_agent.llm import OpenAICompatibleChatModel
from xinyidai_agent.protocol import ChatRequest, ChatResponse
from xinyidai_agent.runtime import ControlledAgentLoop


def create_app(loop: ControlledAgentLoop | None = None) -> FastAPI:
    app = FastAPI(title="信易贷聊天 Agent", version="0.1.0")
    agent_loop = loop or _build_default_loop()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        return agent_loop.answer(request)

    return app


def _build_default_loop() -> ControlledAgentLoop:
    load_env_file(AgentConfig.default_env_path())
    return ControlledAgentLoop(model=OpenAICompatibleChatModel(AgentConfig.from_env()))
