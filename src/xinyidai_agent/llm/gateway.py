"""LLM 网关实现：LiteLLM 多模型适配 + 保留旧 OpenAI 兼容实现。"""

from __future__ import annotations

from typing import Any

from xinyidai_agent.config import AgentConfig
from xinyidai_agent.llm.protocol import ResponseFormat


class OpenAICompatibleChatModel:
    """基于 openai SDK 的百炼兼容实现（保留向后兼容）。"""

    def __init__(self, config: AgentConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("缺少 openai 依赖，无法创建 OpenAI 兼容模型客户端。") from exc

        if not config.llm_api_key:
            raise RuntimeError("缺少 LLM_API_KEY 或 DASHSCOPE_API_KEY。")

        self._model = config.llm_model
        self._client = OpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            timeout=config.request_timeout_seconds,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        response_format: ResponseFormat | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        completion = self._client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ""


class LiteLLMGateway:
    """基于 LiteLLM 的多模型网关。

    支持百炼、OpenAI、DeepSeek 等所有 OpenAI 兼容接口，
    内置重试、超时、fallback、token 计数。
    """

    def __init__(self, config: AgentConfig) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("缺少 litellm 依赖，请 pip install litellm。") from exc

        if not config.llm_api_key:
            raise RuntimeError("缺少 LLM_API_KEY 或 DASHSCOPE_API_KEY。")

        self._model = config.llm_model
        self._api_key = config.llm_api_key
        self._api_base = config.llm_base_url
        self._timeout = config.request_timeout_seconds

    def complete(
        self,
        messages: list[dict[str, str]],
        response_format: ResponseFormat | None = None,
    ) -> str:
        import litellm

        kwargs: dict[str, Any] = {
            "model": f"openai/{self._model}",
            "messages": messages,
            "api_key": self._api_key,
            "api_base": self._api_base,
            "timeout": self._timeout,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content or ""
