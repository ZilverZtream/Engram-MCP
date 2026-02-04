from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Dict

from . import db as dbmod


@dataclass
class Job:
    job_id: str
    kind: str
    created_at: float
    task: asyncio.Task


class JobManager:
    def __init__(
        self,
        db_path: str,
        max_concurrent_jobs: int = 5,
        max_queue_size: int = 1000,
    ) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()
        # Semaphore to limit concurrent jobs and prevent DoS
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self._max_queue_size = max(1, int(max_queue_size))
        self._db_path = db_path
        self._reconciled = False
        self._reconcile_lock = asyncio.Lock()

    async def _ensure_reconciled(self) -> None:
        if self._reconciled:
            return
        async with self._reconcile_lock:
            if self._reconciled:
                return
            await dbmod.init_db(self._db_path)
            await dbmod.mark_stale_jobs_failed(
                self._db_path,
                error="Job interrupted by server restart.",
            )
            self._reconciled = True

    async def create(self, kind: str, coro: Awaitable[Any]) -> Job:
        await self._ensure_reconciled()
        async with self._lock:
            if len(self._jobs) >= self._max_queue_size:
                raise RuntimeError("Job queue is full; try again later.")
        job_id = f"{kind}_{int(time.time()*1000)}"
        await dbmod.init_db(self._db_path)
        await dbmod.create_job(self._db_path, job_id=job_id, kind=kind, status="QUEUED")

        # Wrap coroutine with semaphore to limit concurrency
        async def _wrapped_coro():
            async with self._semaphore:
                await dbmod.update_job_status(self._db_path, job_id=job_id, status="PROCESSING")
                try:
                    result = await coro
                except asyncio.CancelledError:
                    await dbmod.update_job_status(self._db_path, job_id=job_id, status="CANCELLED")
                    raise
                except Exception as exc:
                    await dbmod.update_job_status(
                        self._db_path,
                        job_id=job_id,
                        status="FAILED",
                        error=str(exc),
                    )
                    raise
                await dbmod.update_job_status(self._db_path, job_id=job_id, status="COMPLETED")
                return result

        task = asyncio.create_task(_wrapped_coro())
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
        await self._ensure_reconciled()
        await dbmod.init_db(self._db_path)
        rows = await dbmod.list_jobs(self._db_path)
        out: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            created_at = row.get("created_at") or ""
            out[str(row.get("job_id"))] = {
                "kind": row.get("kind"),
                "status": row.get("status"),
                "created_at": created_at,
                "updated_at": row.get("updated_at"),
                "error": row.get("error"),
            }
        return out
