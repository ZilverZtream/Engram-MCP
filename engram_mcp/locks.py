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
        self._writer_task: asyncio.Task | None = None
        self._writer_depth = 0

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
        # Pre-register write intent BEFORE acquiring the condition lock so
        # that any reader entering acquire_read after this point will see
        # _writers_waiting > 0 and block.  This narrows the starvation window
        # that exists when many reader coroutines are queued ahead of a writer
        # in the event-loop schedule.
        self._writers_waiting += 1
        _decremented = False
        try:
            async with self._cond:
                current = asyncio.current_task()
                if self._writer and current is not None and current == self._writer_task:
                    # Reentrant acquire: undo the early increment and deepen.
                    self._writers_waiting -= 1
                    _decremented = True
                    self._writer_depth += 1
                    return
                try:
                    while self._writer or self._readers > 0:
                        await self._cond.wait()
                    self._writer = True
                    self._writer_task = current
                    self._writer_depth = 1
                    self._writers_waiting -= 1
                    _decremented = True
                except BaseException:
                    # Cancelled while waiting: wake any blocked readers/writers
                    # so they are not stuck waiting for a writer that gave up.
                    self._writers_waiting -= 1
                    _decremented = True
                    self._cond.notify_all()
                    raise
        finally:
            if not _decremented:
                # Safety net: if the condition lock itself raised (e.g. the
                # event loop is closing), ensure the counter is corrected.
                self._writers_waiting -= 1

    async def release_write(self) -> None:
        async with self._cond:
            current = asyncio.current_task()
            if self._writer and current is not None and current == self._writer_task:
                self._writer_depth = max(0, self._writer_depth - 1)
                if self._writer_depth > 0:
                    return
            self._writer = False
            self._writer_task = None
            self._writer_depth = 0
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
