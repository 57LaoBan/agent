"""向量化模块（基于本地 BGE-M3 模型）。"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import threading
import time
from typing import Any

import numpy as np


class BGEEmbedder:
    """BGE-M3 向量化器。

    特性：
    - 本地 BGE-M3 模型，默认读取 HuggingFace 缓存。
    - 查询向量 LRU 缓存，降低在线重复查询延迟。
    - 文档批量向量化，适配离线入库。
    - 输出向量统一归一化，可直接用于 pgvector cosine 检索。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_dir: str | None = None,
        device: str = "cpu",
        max_length: int = 8192,
        query_cache_size: int = 1000,
        model: Any | None = None,
    ) -> None:
        """初始化向量化器。

        Args:
            cache_dir: 模型缓存目录，传 None 时使用 HuggingFace 默认路径
                （`HF_HOME` 环境变量或 `~/.cache/huggingface/hub`），以命中系统已下载的模型。
        """
        if max_length <= 0:
            raise ValueError("max_length 必须大于 0")
        if query_cache_size < 0:
            raise ValueError("query_cache_size 不能小于 0")

        self._model_name = model_name
        self._cache_dir = cache_dir
        self._device = device
        self._max_length = max_length
        self._query_cache_size = query_cache_size
        self._model = model
        self._dimension = 1024
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.RLock()
        self._metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "total_duration": 0.0,
            "errors": 0,
        }

    def embed_query(self, query: str) -> np.ndarray:
        """向量化查询文本，并缓存归一化结果。"""
        normalized_query = self._normalize_text(query)
        cache_key = self._cache_key(normalized_query)
        start_time = time.perf_counter()

        with self._lock:
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                self._query_cache.move_to_end(cache_key)
                self._metrics["total_requests"] += 1
                self._metrics["cache_hits"] += 1
                return cached.copy()

        try:
            embedding = self._encode_one(normalized_query)
        except Exception as exc:
            with self._lock:
                self._metrics["errors"] += 1
            raise RuntimeError(f"查询向量化失败：{exc}") from exc

        with self._lock:
            self._metrics["total_requests"] += 1
            self._metrics["total_duration"] += time.perf_counter() - start_time
            self._cache_query(cache_key, embedding)

        return embedding.copy()

    def embed_documents(
        self,
        documents: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """批量向量化文档文本。"""
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        texts = [self._normalize_text(document) for document in documents]
        start_time = time.perf_counter()

        try:
            embeddings = self._model_instance.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
            )
            array = self._ensure_2d_float32(embeddings)
        except Exception as exc:
            with self._lock:
                self._metrics["errors"] += len(texts)
            raise RuntimeError(f"批量文档向量化失败：{exc}") from exc

        with self._lock:
            self._metrics["total_requests"] += len(texts)
            self._metrics["total_duration"] += time.perf_counter() - start_time

        return array

    @property
    def dimension(self) -> int:
        """向量维度。"""
        return self._dimension

    def get_metrics(self) -> dict[str, float | int]:
        """获取性能指标快照。"""
        with self._lock:
            total_requests = int(self._metrics["total_requests"])
            cache_hits = int(self._metrics["cache_hits"])
            errors = int(self._metrics["errors"])
            total_duration = float(self._metrics["total_duration"])

        avg_duration = total_duration / total_requests if total_requests > 0 else 0.0
        cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
        error_rate = errors / total_requests if total_requests > 0 else 0.0
        return {
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "avg_duration_ms": avg_duration * 1000,
            "errors": errors,
            "error_rate": error_rate,
            "query_cache_size": len(self._query_cache),
        }

    def clear_cache(self) -> None:
        """清空查询向量缓存。"""
        with self._lock:
            self._query_cache.clear()

    @property
    def _model_instance(self) -> Any:
        """懒加载 SentenceTransformer 模型。"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("BGEEmbedder 需要安装 sentence-transformers。") from exc

            kwargs: dict[str, Any] = {"device": self._device}
            if self._cache_dir:
                kwargs["cache_folder"] = self._cache_dir
            self._model = SentenceTransformer(self._model_name, **kwargs)
        return self._model

    def _encode_one(self, text: str) -> np.ndarray:
        """向量化单条文本。"""
        embedding = self._model_instance.encode(text, normalize_embeddings=True)
        return self._ensure_1d_float32(embedding)

    def _normalize_text(self, text: str) -> str:
        """校验并截断文本。"""
        if text is None:
            raise ValueError("向量化文本不能为 None")
        normalized = str(text).strip()
        if not normalized:
            raise ValueError("向量化文本不能为空")
        return normalized[: self._max_length]

    def _cache_query(self, cache_key: str, embedding: np.ndarray) -> None:
        """写入 LRU 查询缓存。"""
        if self._query_cache_size == 0:
            return
        self._query_cache[cache_key] = embedding.copy()
        self._query_cache.move_to_end(cache_key)
        while len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)

    def _cache_key(self, text: str) -> str:
        """生成稳定缓存键。"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _ensure_1d_float32(self, embedding: Any) -> np.ndarray:
        """规范化单条向量形状。"""
        array = np.asarray(embedding, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"期望一维查询向量，实际维度：{array.ndim}")
        self._ensure_dimension(array)
        return array

    def _ensure_2d_float32(self, embeddings: Any) -> np.ndarray:
        """规范化批量向量形状。"""
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"期望二维文档向量矩阵，实际维度：{array.ndim}")
        if array.shape[0] == 0:
            return array.reshape(0, self._dimension)
        self._ensure_dimension(array[0])
        return array

    def _ensure_dimension(self, embedding: np.ndarray) -> None:
        """首次从模型输出确认向量维度，后续保持一致。"""
        actual_dimension = int(embedding.shape[-1])
        if actual_dimension <= 0:
            raise ValueError("向量维度必须大于 0")
        if self._dimension != actual_dimension:
            if self._dimension == 1024 and self._model is not None:
                self._dimension = actual_dimension
                return
            raise ValueError(f"向量维度不一致：期望 {self._dimension}，实际 {actual_dimension}")
