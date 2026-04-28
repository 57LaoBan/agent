from __future__ import annotations

import argparse
from pathlib import Path

from xinyidai_agent.config import AgentConfig, load_env_file
from xinyidai_agent.llm import OpenAICompatibleChatModel
from xinyidai_agent.protocol import ChatRequest
from xinyidai_agent.runtime import ControlledAgentLoop


def main() -> int:
    parser = argparse.ArgumentParser(description="信易贷 Agent 本地单轮问答")
    parser.add_argument("message", help="用户问题")
    parser.add_argument("--env", default="config/.env", help="本地配置文件路径")
    args = parser.parse_args()

    load_env_file(Path(args.env))
    loop = ControlledAgentLoop(model=OpenAICompatibleChatModel(AgentConfig.from_env()))
    response = loop.answer(ChatRequest(user_message=args.message))
    print(response.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
