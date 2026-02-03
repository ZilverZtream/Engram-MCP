from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional


@dataclass
class Job:
    job_id: str
    kind: str
    created_at: float
    task: asyncio.Task


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self, kind: str, coro: Awaitable[Any]) -> Job:
        job_id = f"{kind}_{int(time.time()*1000)}"
        task = asyncio.create_task(coro)
        job = Job(job_id=job_id, kind=kind, created_at=time.time(), task=task)
        async with self._lock:
            self._jobs[job_id] = job
        # Cleanup when done
        task.add_done_callback(lambda _t: asyncio.create_task(self._cleanup(job_id)))
        return job

    async def _cleanup(self, job_id: str) -> None:
        async with self._lock:
            self._jobs.pop(job_id, None)

    async def cancel(self, job_id: str) -> bool:
        async with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return False
        job.task.cancel()
        return True

    async def list(self) -> Dict[str, Dict[str, Any]]:
        async with self._lock:
            jobs = list(self._jobs.values())
        out: Dict[str, Dict[str, Any]] = {}
        for j in jobs:
            out[j.job_id] = {
                "kind": j.kind,
                "age_s": round(time.time() - j.created_at, 1),
                "done": j.task.done(),
                "cancelled": j.task.cancelled(),
            }
        return out
