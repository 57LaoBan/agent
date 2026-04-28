from __future__ import annotations

from typing import Protocol

from openai import OpenAI

from xinyidai_agent.config import AgentConfig


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        """根据消息列表生成回答。"""


class OpenAICompatibleChatModel:
    def __init__(self, config: AgentConfig) -> None:
        if not config.llm_api_key:
            raise RuntimeError("缺少 LLM_API_KEY 或 DASHSCOPE_API_KEY。")

        self._model = config.llm_model
        self._client = OpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            timeout=config.request_timeout_seconds,
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return completion.choices[0].message.content or ""
