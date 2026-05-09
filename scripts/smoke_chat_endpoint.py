"""自包含 /chat 冒烟验收：覆盖 Phase 9 的 11 条 query 路径。"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from xinyidai_agent.protocol import ChatRequest  # noqa: E402
from test.e2e.test_react_loop_matrix import CASES, build_matrix_agent  # noqa: E402


def main() -> None:
    """执行 11 条 query 并在路径不符合预期时返回非零退出码。"""
    agent = build_matrix_agent()
    passed = 0
    failures: list[str] = []

    for case_id, query, expected, has_sources, expected_path in CASES:
        response = agent.answer(ChatRequest(user_message=query, session_id=f"smoke-{case_id}"))
        stop_ok = response.stop_reason in expected if isinstance(expected, tuple) else response.stop_reason == expected
        source_ok = len(response.sources) >= 1 if has_sources else True
        path_ok = _path_matches(expected_path, response)
        ok = stop_ok and source_ok and path_ok
        if ok:
            passed += 1
        else:
            failures.append(
                (
                    f"{case_id}: stop={response.stop_reason}, sources={len(response.sources)}, "
                    f"path={expected_path}, business={response.business_status}, answer={response.answer[:80]}"
                )
            )
        print(
            f"[{'OK' if ok else 'FAIL'}] {case_id}: "
            f"stop={response.stop_reason} sources={len(response.sources)} path={expected_path}"
        )

    if failures:
        print("\nsmoke_chat_endpoint failures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print(f"\nsmoke_chat_endpoint: {passed}/{len(CASES)} query 路径正确")


def _path_matches(expected_path: str, response) -> bool:
    """根据事件和工具轨迹判断路径是否匹配。"""
    event_types = [event.event_type for event in response.events]
    if expected_path in {"L0", "L1"}:
        return "route_decision" not in event_types and response.tool_trace == []
    if expected_path == "L3-fast":
        return bool(response.tool_trace and response.tool_trace[0].tool_name == "rag_search")
    if expected_path == "L3-react":
        return bool(response.route_decision)
    return False


if __name__ == "__main__":
    main()
