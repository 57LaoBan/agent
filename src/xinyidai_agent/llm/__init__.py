"""LLM 网关层。

对外暴露 ChatModel Protocol 和具体实现，上层模块只依赖 Protocol。
"""

from xinyidai_agent.llm.protocol import ChatModel, JSON_OBJECT_RESPONSE_FORMAT, ResponseFormat
from xinyidai_agent.llm.gateway import LiteLLMGateway, OpenAICompatibleChatModel

__all__ = [
    "ChatModel",
    "JSON_OBJECT_RESPONSE_FORMAT",
    "LiteLLMGateway",
    "OpenAICompatibleChatModel",
    "ResponseFormat",
]
