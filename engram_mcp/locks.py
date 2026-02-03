from __future__ import annotations

import asyncio
from typing import Dict


_project_locks: Dict[str, asyncio.Lock] = {}
_locks_lock = asyncio.Lock()


async def get_project_lock(project_id: str) -> asyncio.Lock:
    """Get or create a lock for a specific project."""
    async with _locks_lock:
        if project_id not in _project_locks:
            _project_locks[project_id] = asyncio.Lock()
        return _project_locks[project_id]
