"""异步运行时桥接器。

把 async-only 的 RAG 链路（pgvector 连接池、BGE-M3 embedding、Reranker）
接到同步的工具协议和 FastAPI sync handler 上。所有协程都被提交到一个
后台线程独占的 event loop，避免 FastAPI threadpool 每次新建 loop 时与
asyncio.Queue / Future 跨 loop 冲突。

设计要点：
- 单例后台线程，loop.run_forever 长期常驻；
- run_coroutine 走 asyncio.run_coroutine_threadsafe，多个调用方线程并发安全；
- close 阶段先停 loop 再 join 线程，保证连接池等资源能在同一 loop 内释放。
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar


T = TypeVar("T")


def _create_event_loop() -> asyncio.AbstractEventLoop:
    """创建与 psycopg 异步驱动兼容的 event loop。

    Windows 默认的 ProactorEventLoop 不支持 psycopg 的异步 socket 操作，
    必须切到 SelectorEventLoop；其它平台使用 asyncio 默认实现。
    """
    if sys.platform.startswith("win"):
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()


class AsyncRuntime:
    """后台线程 + 独占 event loop，对外提供同步 run_coroutine 接口。

    用法：

        runtime = AsyncRuntime()
        try:
            runtime.run_coroutine(store.initialize())
            result = runtime.run_coroutine(retriever.retrieve_async(query, top_k))
        finally:
            runtime.close()
    """

    _DEFAULT_THREAD_NAME = "xinyidai-async-runtime"

    def __init__(self, thread_name: str | None = None) -> None:
        """启动后台线程并在其中创建独立 event loop。"""
        self._loop: asyncio.AbstractEventLoop = _create_event_loop()
        self._ready = threading.Event()
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name or self._DEFAULT_THREAD_NAME,
            daemon=True,
        )
        self._thread.start()
        # 等待后台线程把 loop 绑定到自身，再返回，防止首次 submit 抢跑。
        self._ready.wait(timeout=5.0)
        if not self._ready.is_set():
            raise RuntimeError("AsyncRuntime 后台 event loop 启动超时")

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """对外暴露后台 loop，供调用方（如 ProductionRAGRetriever）做相等性判断。"""
        return self._loop

    def run_coroutine(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        """同步等待协程在后台 loop 里执行完成并返回结果。

        Args:
            coro: 待执行协程。
            timeout: 等待秒数，None 代表无超时（生产中应由调用方传入合理值）。
        """
        if self._closed:
            raise RuntimeError("AsyncRuntime 已关闭")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def close(self, join_timeout: float = 10.0) -> None:
        """优雅关闭：停止 loop 并等待线程退出。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                # loop 已经 stop，忽略。
                pass
        self._thread.join(timeout=join_timeout)
        if not self._loop.is_closed():
            self._loop.close()

    def _run(self) -> None:
        """后台线程入口：把 loop 绑定到自身并 run_forever。"""
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            # 清理待完成任务，避免 close 时报 warning。
            pending = [task for task in asyncio.all_tasks(self._loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
