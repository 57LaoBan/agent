from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "e2e_cases.json"
DEFAULT_API_URL = "http://127.0.0.1:8000"


def main() -> int:
    parser = argparse.ArgumentParser(description="信易贷 Agent 后端端到端测试")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="已启动的 Agent API 地址")
    parser.add_argument("--start-server", action="store_true", help="由脚本启动 uvicorn 后端")
    parser.add_argument("--case", help="只运行指定 case id")
    args = parser.parse_args()

    load_env(ROOT / "config" / ".env")
    assert_live_model_config()

    server = None
    try:
        if args.start_server:
            server = start_backend(args.api_url)
            wait_for_health(args.api_url)

        cases = load_cases(args.case)
        for case in cases:
            run_case(args.api_url, case)

        print(f"[e2e] 后端端到端测试通过：{len(cases)} 个 case")
        return 0
    finally:
        if server:
            server.terminate()
            server.wait(timeout=10)


def load_env(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def assert_live_model_config() -> None:
    if os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY"):
        return

    raise RuntimeError(
        "缺少 LLM_API_KEY 或 DASHSCOPE_API_KEY。端到端测试默认要求真实模型调用，"
        "请先在 config/.env 配置模型密钥。"
    )


def start_backend(api_url: str) -> subprocess.Popen[str]:
    port = api_url.rstrip("/").rsplit(":", 1)[-1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "xinyidai_agent.api:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            port,
        ],
        cwd=ROOT,
        env=env,
        text=True,
    )


def wait_for_health(api_url: str) -> None:
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{api_url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.5)

    raise TimeoutError(f"后端未在 30 秒内就绪：{api_url}")


def load_cases(case_id: str | None) -> list[dict[str, Any]]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    cases = payload["backend_cases"]
    if case_id:
        cases = [case for case in cases if case["id"] == case_id]
    if not cases:
        raise ValueError(f"未找到后端 e2e case：{case_id}")
    return cases


def run_case(api_url: str, case: dict[str, Any]) -> None:
    request_payload = {
        "user_message": case["message"],
        "top_k": case.get("top_k", 5),
        "metadata": case.get("metadata", {}),
    }
    response = post_json(f"{api_url}/chat", request_payload)

    assert_equal(response["route_decision"]["scene"], case["expected_scene"], case, "scene")
    if expected_allowed_tools := case.get("expected_allowed_tools"):
        assert_equal(
            response["route_decision"]["allowed_tools"],
            expected_allowed_tools,
            case,
            "allowed_tools",
        )
    assert_equal(
        response["route_decision"]["allowed_tool_categories"],
        case["expected_allowed_tool_categories"],
        case,
        "allowed_tool_categories",
    )

    if expected_tool := case.get("expected_tool"):
        tool_result = first_tool_result(response, expected_tool)
        assert_equal(tool_result["tool_category"], case["expected_tool_category"], case, "tool_category")
        for key, expected_value in case.get("expected_output", {}).items():
            assert_equal(tool_result["output"].get(key), expected_value, case, f"output.{key}")

    if expected_pending_tool := case.get("expected_pending_tool"):
        pending_action = response.get("pending_action")
        if not pending_action:
            raise AssertionError(f"{case['id']} 期望 pending_action，但响应为空")
        tool_call = pending_action["tool_call"]
        assert_equal(tool_call["tool_name"], expected_pending_tool, case, "pending_tool")
        assert_equal(
            tool_call["tool_category"],
            case["expected_pending_tool_category"],
            case,
            "pending_tool_category",
        )

    if case.get("requires_live_model") and not response.get("answer"):
        raise AssertionError(f"{case['id']} 期望真实模型生成 answer，但 answer 为空")

    print(f"[e2e] PASS backend/{case['id']}")


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def first_tool_result(response: dict[str, Any], tool_name: str) -> dict[str, Any]:
    for result in response["tool_trace"]:
        if result["tool_name"] == tool_name:
            return result
    raise AssertionError(f"未找到工具结果：{tool_name}")


def assert_equal(actual: Any, expected: Any, case: dict[str, Any], field: str) -> None:
    if actual != expected:
        raise AssertionError(
            f"{case['id']} 字段 {field} 不匹配：expected={expected!r}, actual={actual!r}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
