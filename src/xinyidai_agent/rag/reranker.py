"""Cross-Encoder 重排序器。"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import threading
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from xinyidai_agent.rag.retriever import RetrievalResult


class CrossEncoderReranker:
    """基于本地 Cross-Encoder 模型的生产重排序器。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        cache_dir: str | None = None,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 32,
        result_cache_size: int = 1000,
        model: Any | None = None,
    ) -> None:
        """初始化重排序模型和结果缓存。"""
        if max_length <= 0:
            raise ValueError("max_length 必须大于 0")
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if result_cache_size < 0:
            raise ValueError("result_cache_size 不能小于 0")
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size
        self._model = model
        self._result_cache_size = result_cache_size
        self._score_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._lock = threading.RLock()
        self._metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "total_duration": 0.0,
            "errors": 0,
        }

    @property
    def model_name(self) -> str:
        """返回当前重排序模型名称。"""
        return self._model_name

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """对候选检索结果进行 Cross-Encoder 重排序。"""
        if top_k <= 0 or not results:
            return []

        normalized_query = self._normalize_query(query)
        normalized_results = list(results)
        cache_key = self._cache_key(normalized_query, normalized_results)
        start_time = perf_counter()

        with self._lock:
            cached_scores = self._score_cache.get(cache_key)
            if cached_scores is not None:
                self._score_cache.move_to_end(cache_key)
                self._metrics["total_requests"] += 1
                self._metrics["cache_hits"] += 1
                return self._rank_with_scores(normalized_results, cached_scores, top_k)

        pairs = [(normalized_query, result.content[: self._max_length]) for result in normalized_results]
        try:
            scores = self._predict_scores(pairs)
        except Exception as exc:
            with self._lock:
                self._metrics["errors"] += 1
            raise RuntimeError(f"Cross-Encoder 重排序失败：{exc}") from exc

        score_tuple = tuple(float(score) for score in scores)
        with self._lock:
            self._metrics["total_requests"] += 1
            self._metrics["total_duration"] += perf_counter() - start_time
            self._cache_scores(cache_key, score_tuple)
        return self._rank_with_scores(normalized_results, score_tuple, top_k)

    def rerank_batch(
        self,
        queries: Sequence[str],
        results_list: Sequence[Sequence[RetrievalResult]],
        top_k: int = 5,
    ) -> list[list[RetrievalResult]]:
        """批量重排序多组候选结果。"""
        if len(queries) != len(results_list):
            raise ValueError("queries 和 results_list 长度必须一致")
        return [self.rerank(query, results, top_k=top_k) for query, results in zip(queries, results_list)]

    def get_metrics(self) -> dict[str, float | int]:
        """获取重排序器指标快照。"""
        with self._lock:
            total_requests = int(self._metrics["total_requests"])
            cache_hits = int(self._metrics["cache_hits"])
            total_duration = float(self._metrics["total_duration"])
            errors = int(self._metrics["errors"])
            cache_size = len(self._score_cache)

        return {
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / total_requests if total_requests else 0.0,
            "avg_duration_ms": total_duration / total_requests * 1000 if total_requests else 0.0,
            "errors": errors,
            "error_rate": errors / total_requests if total_requests else 0.0,
            "result_cache_size": cache_size,
        }

    @property
    def _model_instance(self) -> Any:
        """懒加载 sentence-transformers CrossEncoder。"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("CrossEncoderReranker 需要安装 sentence-transformers。") from exc
            kwargs: dict[str, Any] = {
                "max_length": self._max_length,
                "device": self._device,
            }
            if self._cache_dir:
                kwargs["cache_folder"] = self._cache_dir
            self._model = CrossEncoder(self._model_name, **kwargs)
        return self._model

    def _predict_scores(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """调用模型并规范化输出分数。"""
        scores = self._model_instance.predict(pairs, batch_size=self._batch_size)
        array = np.asarray(scores, dtype=np.float32)
        if array.ndim != 1 or array.shape[0] != len(pairs):
            raise ValueError("重排序模型输出维度不符合候选数量")
        return array

    def _rank_with_scores(
        self,
        results: list[RetrievalResult],
        scores: Sequence[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """将重排序分数写入元数据并返回排序结果。"""
        reranked: list[RetrievalResult] = []
        for result, score in zip(results, scores):
            metadata = {
                **result.metadata,
                "retrieval_score": result.score,
                "rerank_score": float(score),
            }
            reranked.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    content=result.content,
                    score=float(score),
                    metadata=metadata,
                    document_title=result.document_title,
                    source_path=result.source_path,
                    start_char=result.start_char,
                    end_char=result.end_char,
                    retrieval_source=f"{result.retrieval_source}+rerank",
                )
            )
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked[:top_k]

    def _normalize_query(self, query: str) -> str:
        """校验并规范化查询文本。"""
        normalized = " ".join(str(query or "").split())
        if not normalized:
            raise ValueError("重排序查询不能为空")
        return normalized

    def _cache_scores(self, cache_key: str, scores: tuple[float, ...]) -> None:
        """写入 LRU 重排序结果缓存。"""
        if self._result_cache_size == 0:
            return
        self._score_cache[cache_key] = scores
        self._score_cache.move_to_end(cache_key)
        while len(self._score_cache) > self._result_cache_size:
            self._score_cache.popitem(last=False)

    def _cache_key(self, query: str, results: Sequence[RetrievalResult]) -> str:
        """为查询和候选集合生成稳定缓存键。"""
        parts = [query]
        parts.extend(f"{result.chunk_id}:{result.score:.8f}" for result in results)
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
