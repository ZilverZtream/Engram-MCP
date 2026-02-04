from __future__ import annotations

import asyncio
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

import numpy as np


# -----------------
# Process-pool worker
# -----------------
_WORKER_MODEL: Any = None
_WORKER_MODEL_NAME: Optional[str] = None
_WORKER_DEVICE: Optional[str] = None


def _worker_init(model_name: str, device: str) -> None:
    global _WORKER_MODEL, _WORKER_MODEL_NAME, _WORKER_DEVICE
    _WORKER_MODEL_NAME = model_name
    _WORKER_DEVICE = device

    from sentence_transformers import SentenceTransformer

    # Each worker loads the model once.
    _WORKER_MODEL = SentenceTransformer(model_name, device=device)


def _embed_texts_worker(texts: List[str]) -> np.ndarray:
    if _WORKER_MODEL is None:
        raise RuntimeError("Embedding worker not initialized")
    # SentenceTransformer returns np.ndarray
    vecs = _WORKER_MODEL.encode(
        texts,
        batch_size=max(1, min(128, len(texts))),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)


@dataclass
class Embedder:
    """Async-friendly embedding client.

    - CPU: uses ProcessPoolExecutor to avoid blocking the event loop and bypass the GIL.
    - CUDA: by default uses ThreadPoolExecutor to avoid CUDA context issues with forking.

    If you *really* want a process pool on CUDA, set prefer_thread_for_cuda=False (but expect more
    platform-specific edge cases).
    """

    model_name: str
    device: str
    prefer_thread_for_cuda: bool = True
    prefer_thread_for_cpu: bool = True
    max_workers: int = 2
    shared: bool = True

    def __post_init__(self) -> None:
        self._proc_pool: Optional[ProcessPoolExecutor] = None
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._shared_key: Optional[Tuple[str, Tuple[Any, ...]]] = None

    _shared_lock = threading.Lock()
    _shared_thread_pools: Dict[Tuple[str, str], Tuple[ThreadPoolExecutor, int]] = {}
    _shared_proc_pools: Dict[Tuple[str, str, int], Tuple[ProcessPoolExecutor, int]] = {}
    _shared_models: Dict[Tuple[str, str], Any] = {}

    def _ensure_executor(self) -> Tuple[str, Any]:
        use_thread = (self.device.startswith("cuda") and self.prefer_thread_for_cuda) or (
            not self.device.startswith("cuda") and self.prefer_thread_for_cpu
        )
        if use_thread:
            if self.shared:
                key = (self.model_name, self.device)
                with self._shared_lock:
                    if self._shared_key is None:
                        pool, ref = self._shared_thread_pools.get(key, (None, 0))
                        if pool is None:
                            pool = ThreadPoolExecutor(max_workers=1)
                        self._shared_thread_pools[key] = (pool, ref + 1)
                        self._shared_key = ("thread", key)
                    pool = self._shared_thread_pools[key][0]
                return "thread", pool
            if self._thread_pool is None:
                self._thread_pool = ThreadPoolExecutor(max_workers=1)
            return "thread", self._thread_pool

        if self.shared:
            key = (self.model_name, self.device, int(self.max_workers))
            with self._shared_lock:
                if self._shared_key is None:
                    pool, ref = self._shared_proc_pools.get(key, (None, 0))
                    if pool is None:
                        mp_context = multiprocessing.get_context("spawn")
                        pool = ProcessPoolExecutor(
                            max_workers=self.max_workers,
                            mp_context=mp_context,
                            initializer=_worker_init,
                            initargs=(self.model_name, self.device),
                        )
                    self._shared_proc_pools[key] = (pool, ref + 1)
                    self._shared_key = ("process", key)
                pool = self._shared_proc_pools[key][0]
            return "process", pool

        if self._proc_pool is None:
            mp_context = multiprocessing.get_context("spawn")
            self._proc_pool = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=mp_context,
                initializer=_worker_init,
                initargs=(self.model_name, self.device),
            )
        return "process", self._proc_pool

    async def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, 0), dtype=np.float32)

        mode, ex = self._ensure_executor()
        loop = asyncio.get_running_loop()

        if mode == "thread":
            # Thread mode: model lives in-process.
            def _embed_in_process() -> np.ndarray:
                from sentence_transformers import SentenceTransformer

                # Lazy init model in this process.
                model = None
                if self.shared:
                    key = (self.model_name, self.device)
                    with self._shared_lock:
                        model = self._shared_models.get(key)
                        if model is None:
                            model = SentenceTransformer(self.model_name, device=self.device)
                            self._shared_models[key] = model
                else:
                    if not hasattr(self, "_inproc_model"):
                        self._inproc_model = SentenceTransformer(self.model_name, device=self.device)
                    model = self._inproc_model
                vecs = model.encode(
                    texts_list,
                    batch_size=max(1, min(128, len(texts_list))),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return np.asarray(vecs, dtype=np.float32)

            return await loop.run_in_executor(ex, _embed_in_process)

        # Process pool
        return await loop.run_in_executor(ex, _embed_texts_worker, texts_list)

    async def embed_one(self, text: str) -> np.ndarray:
        vecs = await self.embed_texts([text])
        return vecs[0]

    async def embed_texts_batched(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 256,
    ) -> AsyncIterator[np.ndarray]:
        texts_list = list(texts)
        if not texts_list:
            return
        size = max(1, int(batch_size))
        for i in range(0, len(texts_list), size):
            yield await self.embed_texts(texts_list[i:i + size])

    def close(self) -> None:
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=False, cancel_futures=True)
        if self._proc_pool is not None:
            self._proc_pool.shutdown(wait=False, cancel_futures=True)
        if self.shared and self._shared_key is not None:
            pool_type, key = self._shared_key
            with self._shared_lock:
                if pool_type == "thread":
                    pool, ref = self._shared_thread_pools.get(key, (None, 0))
                    if pool and ref <= 1:
                        pool.shutdown(wait=False, cancel_futures=True)
                        self._shared_thread_pools.pop(key, None)
                        self._shared_models.pop((key[0], key[1]), None)
                    else:
                        self._shared_thread_pools[key] = (pool, ref - 1)
                else:
                    pool, ref = self._shared_proc_pools.get(key, (None, 0))
                    if pool and ref <= 1:
                        pool.shutdown(wait=False, cancel_futures=True)
                        self._shared_proc_pools.pop(key, None)
                    else:
                        self._shared_proc_pools[key] = (pool, ref - 1)
                self._shared_key = None


@dataclass
class _EmbeddingRequest:
    texts: List[str]
    model_name: str
    device: str
    future: asyncio.Future


class EmbeddingService:
    def __init__(
        self,
        *,
        prefer_thread_for_cuda: bool,
        worker_count: int = 2,
        max_workers_per_model: int = 2,
    ) -> None:
        self._prefer_thread_for_cuda = prefer_thread_for_cuda
        self._worker_count = max(1, int(worker_count))
        self._max_workers_per_model = max(1, int(max_workers_per_model))
        self._queue: asyncio.Queue[_EmbeddingRequest] = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._started = asyncio.Event()
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._start_lock:
            if self._workers:
                return
            for _ in range(self._worker_count):
                self._workers.append(asyncio.create_task(self._worker_loop()))
            self._started.set()

    async def _worker_loop(self) -> None:
        embedder_cache: Dict[Tuple[str, str], Embedder] = {}
        try:
            while True:
                req = await self._queue.get()
                key = (req.model_name, req.device)
                embedder = embedder_cache.get(key)
                if embedder is None:
                    embedder = Embedder(
                        model_name=req.model_name,
                        device=req.device,
                        prefer_thread_for_cuda=self._prefer_thread_for_cuda,
                        max_workers=self._max_workers_per_model,
                    )
                    embedder_cache[key] = embedder
                try:
                    vectors = await embedder.embed_texts(req.texts)
                except Exception as exc:
                    if not req.future.done():
                        req.future.set_exception(exc)
                else:
                    if not req.future.done():
                        req.future.set_result(vectors)
        except asyncio.CancelledError:
            for embedder in embedder_cache.values():
                embedder.close()
            raise

    async def embed_texts(self, texts: Sequence[str], *, model_name: str, device: str) -> np.ndarray:
        await self.start()
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, 0), dtype=np.float32)
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put(
            _EmbeddingRequest(texts=text_list, model_name=model_name, device=device, future=future)
        )
        return await future

    async def embed_one(self, text: str, *, model_name: str, device: str) -> np.ndarray:
        vecs = await self.embed_texts([text], model_name=model_name, device=device)
        return vecs[0]

    async def close(self) -> None:
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
