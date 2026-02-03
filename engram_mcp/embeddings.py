from __future__ import annotations

import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional, Sequence, Tuple

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
    max_workers: int = 2

    def __post_init__(self) -> None:
        self._proc_pool: Optional[ProcessPoolExecutor] = None
        self._thread_pool: Optional[ThreadPoolExecutor] = None

    def _ensure_executor(self) -> Tuple[str, Any]:
        use_thread = self.device.startswith("cuda") and self.prefer_thread_for_cuda
        if use_thread:
            if self._thread_pool is None:
                self._thread_pool = ThreadPoolExecutor(max_workers=1)
            return "thread", self._thread_pool

        if self._proc_pool is None:
            # Use spawn context to avoid PyTorch/CUDA fork issues
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
                if not hasattr(self, "_inproc_model"):
                    self._inproc_model = SentenceTransformer(self.model_name, device=self.device)
                vecs = self._inproc_model.encode(
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
