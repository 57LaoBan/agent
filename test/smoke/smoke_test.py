from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from openai import OpenAI


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_local_config() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_env_file(project_root / "config" / ".env")


def build_client() -> OpenAI:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 LLM_API_KEY，请先在 config/.env 中配置。")

    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
        timeout=60,
    )


def main() -> int:
    configure_stdout()
    load_local_config()

    parser = argparse.ArgumentParser(description="阿里云百炼 OpenAI 兼容接口连通性烟测")
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL") or os.getenv("DASHSCOPE_MODEL", DEFAULT_MODEL),
        help="模型名称",
    )
    parser.add_argument("--prompt", default="你是谁？", help="测试问题")
    args = parser.parse_args()

    try:
        client = build_client()
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": args.prompt},
            ],
        )
        print("模型调用成功")
        print(f"model: {args.model}")
        print("answer:")
        print(completion.choices[0].message.content)
        return 0
    except Exception as exc:
        print(f"错误信息：{exc}")
        print("请参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
