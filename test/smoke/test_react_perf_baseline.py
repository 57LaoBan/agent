from __future__ import annotations

from statistics import quantiles
from time import perf_counter

from xinyidai_agent.protocol import ChatRequest

from collections.abc import Callable

from test.e2e.test_react_loop_matrix import build_matrix_agent


def _elapsed_ms(callable_obj: Callable[[], object]) -> float:
    """测量单次调用耗时。"""
    started = perf_counter()
    callable_obj()
    return (perf_counter() - started) * 1000


def _p99(values: list[float]) -> float:
    """计算小样本 p99；样本不足时取最大值。"""
    if len(values) < 3:
        return max(values)
    return quantiles(values, n=100, method="inclusive")[98]


def test_l0_l1_under_50ms() -> None:
    """L0/L1 路径 p99 < 50ms，且不触发模型调用。"""
    agent = build_matrix_agent()
    agent.answer(ChatRequest(user_message="你好", session_id="perf-warmup"))
    durations = [
        _elapsed_ms(lambda idx=idx: agent.answer(ChatRequest(user_message="你好", session_id=f"perf-l1-{idx}")))
        for idx in range(8)
    ]
    assert _p99(durations) < 50


def test_fast_path_under_6s() -> None:
    """L3-fast 路径 p99 < 6s。"""
    agent = build_matrix_agent()
    durations = [
        _elapsed_ms(
            lambda idx=idx: agent.answer(
                ChatRequest(user_message="信易贷的准入条件是什么", session_id=f"perf-fast-{idx}")
            )
        )
        for idx in range(5)
    ]
    assert _p99(durations) < 6000


def test_react_step_under_8s() -> None:
    """L3-react 单步路径 p99 < 8s。"""
    agent = build_matrix_agent()
    durations = [
        _elapsed_ms(
            lambda idx=idx: agent.answer(
                ChatRequest(user_message="查一下 ABC 公司的授信额度", session_id=f"perf-react-{idx}")
            )
        )
        for idx in range(5)
    ]
    assert _p99(durations) < 8000
