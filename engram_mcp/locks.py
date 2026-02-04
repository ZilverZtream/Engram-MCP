from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, AsyncIterator


_project_locks: Dict[str, asyncio.Lock] = {}
_locks_lock = asyncio.Lock()
_project_rwlocks: Dict[str, "RWLock"] = {}


async def get_project_lock(project_id: str) -> asyncio.Lock:
    """Get or create a lock for a specific project."""
    async with _locks_lock:
        if project_id not in _project_locks:
            _project_locks[project_id] = asyncio.Lock()
        return _project_locks[project_id]


async def remove_project_lock(project_id: str) -> None:
    """Remove a project lock once the project is deleted."""
    async with _locks_lock:
        _project_locks.pop(project_id, None)


class RWLock:
    def __init__(self) -> None:
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0
        self._cond = asyncio.Condition()

    async def acquire_read(self) -> None:
        async with self._cond:
            while self._writer or self._writers_waiting > 0:
                await self._cond.wait()
            self._readers += 1

    async def release_read(self) -> None:
        async with self._cond:
            self._readers = max(0, self._readers - 1)
            if self._readers == 0:
                self._cond.notify_all()

    async def acquire_write(self) -> None:
        async with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers > 0:
                    await self._cond.wait()
                self._writer = True
            finally:
                self._writers_waiting = max(0, self._writers_waiting - 1)

    async def release_write(self) -> None:
        async with self._cond:
            self._writer = False
            self._cond.notify_all()

    @asynccontextmanager
    async def read_lock(self) -> AsyncIterator[None]:
        await self.acquire_read()
        try:
            yield
        finally:
            await self.release_read()

    @asynccontextmanager
    async def write_lock(self) -> AsyncIterator[None]:
        await self.acquire_write()
        try:
            yield
        finally:
            await self.release_write()


async def get_project_rwlock(project_id: str) -> RWLock:
    async with _locks_lock:
        if project_id not in _project_rwlocks:
            _project_rwlocks[project_id] = RWLock()
        return _project_rwlocks[project_id]


async def remove_project_rwlock(project_id: str) -> None:
    async with _locks_lock:
        _project_rwlocks.pop(project_id, None)
