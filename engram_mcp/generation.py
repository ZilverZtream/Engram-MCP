from __future__ import annotations

import asyncio
import multiprocessing
import os
import threading
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Tuple


INSIGHT_PROMPT_TEMPLATE = """You are an insight engine that connects two related contexts.

Context A:
{context_a}

Context B:
{context_b}

Task:
Write a single, concise insight that links the two contexts. Be specific and avoid repetition.
If the snippets are unrelated or the connection is trivial, respond with ONLY "NO_INSIGHT".

Insight:
"""


# -----------------
# Process-pool worker (transformers / local model)
# -----------------
_WORKER_MODEL: Any = None
_WORKER_TOKENIZER: Any = None
_WORKER_DEVICE: Optional[str] = None
_WORKER_INIT_ERROR: Optional[str] = None


def _worker_init(model_name: str, device: str) -> None:
    global _WORKER_MODEL, _WORKER_TOKENIZER, _WORKER_DEVICE, _WORKER_INIT_ERROR
    _WORKER_DEVICE = device

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _WORKER_MODEL = AutoModelForCausalLM.from_pretrained(model_name)
        if device.startswith("cuda"):
            _WORKER_MODEL.to(device)
        _WORKER_MODEL.eval()
    except Exception as exc:
        _WORKER_MODEL = None
        _WORKER_TOKENIZER = None
        _WORKER_INIT_ERROR = f"{type(exc).__name__}: {exc}"
        raise


def _generate_worker(
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    if _WORKER_INIT_ERROR:
        raise RuntimeError(f"Generation worker failed to initialize: {_WORKER_INIT_ERROR}")
    if _WORKER_MODEL is None or _WORKER_TOKENIZER is None:
        raise RuntimeError("Generation worker not initialized")

    import torch

    inputs = _WORKER_TOKENIZER(prompt, return_tensors="pt")
    if _WORKER_DEVICE and _WORKER_DEVICE.startswith("cuda"):
        inputs = {k: v.to(_WORKER_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = _WORKER_MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(0.0, float(temperature)),
            top_p=float(top_p),
            pad_token_id=_WORKER_TOKENIZER.eos_token_id,
        )
    text = _WORKER_TOKENIZER.decode(output[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


@dataclass
class Generator:
    """Async-friendly text generation client using transformers."""

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

    _shared_pool_lock: ClassVar[Optional[asyncio.Lock]] = None
    _shared_thread_pools: ClassVar[Dict[Tuple[str, str], Tuple[ThreadPoolExecutor, int]]] = {}
    _shared_proc_pools: ClassVar[Dict[Tuple[str, str, int], Tuple[ProcessPoolExecutor, int]]] = {}
    _shared_model_lock: ClassVar[threading.Lock] = threading.Lock()
    _shared_models: ClassVar[Dict[Tuple[str, str], Tuple[Any, Any]]] = {}

    @classmethod
    def _get_shared_pool_lock(cls) -> asyncio.Lock:
        if cls._shared_pool_lock is None:
            cls._shared_pool_lock = asyncio.Lock()
        return cls._shared_pool_lock

    async def _ensure_executor(self) -> Tuple[str, Any]:
        use_thread = (self.device.startswith("cuda") and self.prefer_thread_for_cuda) or (
            not self.device.startswith("cuda") and self.prefer_thread_for_cpu
        )
        _thread_workers = max(1, self.max_workers)
        if use_thread:
            if self.shared:
                key = (self.model_name, self.device)
                async with self._get_shared_pool_lock():
                    if self._shared_key is None:
                        pool, ref = self._shared_thread_pools.get(key, (None, 0))
                        if pool is None:
                            pool = ThreadPoolExecutor(max_workers=_thread_workers)
                        self._shared_thread_pools[key] = (pool, ref + 1)
                        self._shared_key = ("thread", key)
                    pool = self._shared_thread_pools[key][0]
                return "thread", pool
            if self._thread_pool is None:
                self._thread_pool = ThreadPoolExecutor(max_workers=_thread_workers)
            return "thread", self._thread_pool

        _proc_workers = min(self.max_workers, 2)

        if self.shared:
            key = (self.model_name, self.device, _proc_workers)
            async with self._get_shared_pool_lock():
                if self._shared_key is None:
                    pool, ref = self._shared_proc_pools.get(key, (None, 0))
                    if pool is None:
                        mp_context = multiprocessing.get_context("spawn")
                        pool = ProcessPoolExecutor(
                            max_workers=_proc_workers,
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
                max_workers=_proc_workers,
                mp_context=mp_context,
                initializer=_worker_init,
                initargs=(self.model_name, self.device),
            )
        return "process", self._proc_pool

    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        mode, ex = await self._ensure_executor()
        loop = asyncio.get_running_loop()

        if mode == "thread":
            def _generate_in_process() -> str:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                if self.shared:
                    key = (self.model_name, self.device)
                    with self._shared_model_lock:
                        model_tokenizer = self._shared_models.get(key)
                        if model_tokenizer is None:
                            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                            model = AutoModelForCausalLM.from_pretrained(self.model_name)
                            if self.device.startswith("cuda"):
                                model.to(self.device)
                            model.eval()
                            model_tokenizer = (model, tokenizer)
                            self._shared_models[key] = model_tokenizer
                else:
                    if not hasattr(self, "_inproc_model"):
                        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                        model = AutoModelForCausalLM.from_pretrained(self.model_name)
                        if self.device.startswith("cuda"):
                            model.to(self.device)
                        model.eval()
                        self._inproc_model = (model, tokenizer)
                    model_tokenizer = self._inproc_model

                model, tokenizer = model_tokenizer
                inputs = tokenizer(prompt, return_tensors="pt")
                if self.device.startswith("cuda"):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0,
                        temperature=max(0.0, float(temperature)),
                        top_p=float(top_p),
                        pad_token_id=tokenizer.eos_token_id,
                    )
                text = tokenizer.decode(output[0], skip_special_tokens=True)
                if text.startswith(prompt):
                    text = text[len(prompt):]
                return text.strip()

            return await loop.run_in_executor(ex, _generate_in_process)

        return await loop.run_in_executor(
            ex,
            _generate_worker,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def close(self) -> None:
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True, cancel_futures=True)
        if self._proc_pool is not None:
            self._proc_pool.shutdown(wait=True, cancel_futures=True)
        if self.shared and self._shared_key is not None:
            pool_type, key = self._shared_key
            with self._shared_model_lock:
                if pool_type == "thread":
                    pool, ref = self._shared_thread_pools.get(key, (None, 0))
                    if pool and ref <= 1:
                        pool.shutdown(wait=True, cancel_futures=True)
                        self._shared_thread_pools.pop(key, None)
                        self._shared_models.pop((key[0], key[1]), None)
                    else:
                        self._shared_thread_pools[key] = (pool, ref - 1)
                else:
                    pool, ref = self._shared_proc_pools.get(key, (None, 0))
                    if pool and ref <= 1:
                        pool.shutdown(wait=True, cancel_futures=True)
                        self._shared_proc_pools.pop(key, None)
                    else:
                        self._shared_proc_pools[key] = (pool, ref - 1)
                self._shared_key = None


@dataclass
class _GenerationRequest:
    prompt: str
    model_name: str
    device: str
    max_new_tokens: int
    temperature: float
    top_p: float
    future: asyncio.Future


class GenerationService:
    def __init__(
        self,
        *,
        prefer_thread_for_cuda: bool = True,
        worker_count: int = 2,
        max_workers_per_model: int = 2,
        max_cached_models: int = 2,
        gpu_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> None:
        self._prefer_thread_for_cuda = prefer_thread_for_cuda
        self._worker_count = max(1, int(worker_count))
        self._max_workers_per_model = max(1, int(max_workers_per_model))
        self._max_cached_models = max(1, int(max_cached_models))
        self._gpu_semaphore = gpu_semaphore
        self._queue: asyncio.Queue[_GenerationRequest] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
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
        generator_cache: "OrderedDict[Tuple[str, str], Generator]" = OrderedDict()
        try:
            while True:
                req = await self._queue.get()
                key = (req.model_name, req.device)
                generator = generator_cache.get(key)
                if generator is None:
                    generator = Generator(
                        model_name=req.model_name,
                        device=req.device,
                        prefer_thread_for_cuda=self._prefer_thread_for_cuda,
                        max_workers=self._max_workers_per_model,
                    )
                    generator_cache[key] = generator
                    generator_cache.move_to_end(key)
                    while len(generator_cache) > self._max_cached_models:
                        old_key, old_generator = generator_cache.popitem(last=False)
                        old_generator.close()
                else:
                    generator_cache.move_to_end(key)
                try:
                    # Use GPU semaphore if available (low-VRAM mode)
                    if self._gpu_semaphore is not None:
                        async with self._gpu_semaphore:
                            result = await generator.generate(
                                req.prompt,
                                max_new_tokens=req.max_new_tokens,
                                temperature=req.temperature,
                                top_p=req.top_p,
                            )
                    else:
                        result = await generator.generate(
                            req.prompt,
                            max_new_tokens=req.max_new_tokens,
                            temperature=req.temperature,
                            top_p=req.top_p,
                        )
                except Exception as exc:
                    if not req.future.done():
                        req.future.set_exception(exc)
                else:
                    if not req.future.done():
                        req.future.set_result(result)
        except asyncio.CancelledError:
            for generator in generator_cache.values():
                generator.close()
            raise

    async def generate(
        self,
        prompt: str,
        *,
        model_name: str,
        device: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put(
            _GenerationRequest(
                prompt=prompt,
                model_name=model_name,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                future=future,
            )
        )
        return await future

    async def generate_insight(
        self,
        context_a: str,
        context_b: str,
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        prompt = build_insight_prompt(context_a, context_b)
        model = model_name or os.getenv("ENGRAM_GENERATION_MODEL", "gpt2")
        device_name = device or os.getenv("ENGRAM_GENERATION_DEVICE", "cpu")
        return await self.generate(
            prompt,
            model_name=model,
            device=device_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    async def close(self, *, graceful: bool = True) -> None:
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()


def build_insight_prompt(context_a: str, context_b: str) -> str:
    return INSIGHT_PROMPT_TEMPLATE.format(context_a=context_a, context_b=context_b)


async def generate_insight_async(
    context_a: str,
    context_b: str,
    *,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    service: Optional[GenerationService] = None,
) -> str:
    prompt = build_insight_prompt(context_a, context_b)
    model = model_name or os.getenv("ENGRAM_GENERATION_MODEL", "gpt2")
    device_name = device or os.getenv("ENGRAM_GENERATION_DEVICE", "cpu")

    owns_service = False
    if service is None:
        service = GenerationService()
        owns_service = True

    try:
        return await service.generate(
            prompt,
            model_name=model,
            device=device_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    finally:
        if owns_service:
            await service.close()


def generate_insight(context_a: str, context_b: str) -> str:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(generate_insight_async(context_a, context_b))
    raise RuntimeError("generate_insight must be called from sync code. Use generate_insight_async instead.")
