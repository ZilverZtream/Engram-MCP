from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, Optional, Tuple

from fastmcp import FastMCP

from engram_mcp.config import load_config
from engram_mcp.embeddings import Embedder
from engram_mcp.indexing import Indexer
from engram_mcp.locks import get_project_lock
from engram_mcp.jobs import JobManager
from engram_mcp.search import SearchEngine
from engram_mcp.security import PathNotAllowed, enforce_allowed_roots
from engram_mcp import db as dbmod


cfg = load_config()

mcp = FastMCP(name="Engram MCP")

jobs = JobManager()
indexer = Indexer(cfg)
search_engine = SearchEngine(db_path=cfg.db_path, index_dir=cfg.index_dir)
_embedder_cache: Dict[str, Tuple[str, str, Embedder]] = {}
_embedder_lock = asyncio.Lock()


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _sanitize_project_id(project_id: str) -> str:
    sanitized = os.path.basename(project_id)
    if not re.match(r"^[a-zA-Z0-9_-]+$", sanitized):
        raise ValueError(f"Invalid project_id: {project_id}. Only alphanumeric characters, underscores, and hyphens are allowed.")
    return sanitized


async def _get_embedder_for_project(project_id: str) -> Embedder:
    proj = await dbmod.get_project(cfg.db_path, project_id)
    if not proj:
        raise ValueError(f"Project not found: {project_id}")

    md = proj.get("metadata") or {}
    model_name = md.get("model_name") or cfg.model_name_text
    device = md.get("device") or _device()
    async with _embedder_lock:
        cached = _embedder_cache.get(project_id)
        if cached and cached[0] == model_name and cached[1] == device:
            return cached[2]
        if cached:
            cached[2].close()
        embedder = Embedder(
            model_name=model_name,
            device=device,
            prefer_thread_for_cuda=cfg.prefer_thread_for_cuda,
        )
        _embedder_cache[project_id] = (model_name, device, embedder)
        return embedder


@mcp.tool
async def index_project(
    directory: str,
    project_name: str,
    project_type: str,
) -> str:
    """Index a directory (within allowed_roots) for hybrid search."""
    try:
        directory = enforce_allowed_roots(directory, cfg.allowed_roots)
    except PathNotAllowed as e:
        return f"❌ {e}"

    async def _run() -> str:
        pid = await indexer.index_project(directory=directory, project_name=project_name, project_type=project_type)
        return f"✅ Indexed '{project_name}' as project_id: {pid}"

    job = await jobs.create("index", _run())
    # Wait for completion (Claude expects tool result), but the event loop remains responsive.
    try:
        return await job.task
    except asyncio.CancelledError:
        return f"🛑 Index job cancelled: {job.job_id}"


@mcp.tool
async def update_project(project_id: str) -> str:
    """Update an existing project by scanning its stored directory."""

    async def _run() -> str:
        changes = await indexer.update_project(project_id=project_id)
        return f"✅ Updated {project_id}: +{changes['added']} added, -{changes['deleted']} deleted"

    job = await jobs.create("update", _run())
    try:
        return await job.task
    except asyncio.CancelledError:
        return f"🛑 Update job cancelled: {job.job_id}"


@mcp.tool
async def list_projects() -> Dict[str, Any]:
    """List indexed projects."""
    await dbmod.init_db(cfg.db_path)
    return {"projects": await dbmod.list_projects(cfg.db_path)}


@mcp.tool
async def project_info(project_id: str) -> Dict[str, Any]:
    """Get detailed info about a project."""
    await dbmod.init_db(cfg.db_path)
    proj = await dbmod.get_project(cfg.db_path, project_id)
    if not proj:
        return {"error": f"Project not found: {project_id}"}
    return proj


@mcp.tool
async def search_memory(
    query: str,
    project_id: str,
    max_results: int = 10,
    use_mmr: bool = True,
) -> Dict[str, Any]:
    """Hybrid search: SQLite FTS5 (lexical) + FAISS (vector) with RRF, optional MMR."""
    await dbmod.init_db(cfg.db_path)
    proj = await dbmod.get_project(cfg.db_path, project_id)
    if not proj:
        return {"error": f"Project not found: {project_id}"}

    embedder = await _get_embedder_for_project(project_id)
    qv = await embedder.embed_one(query)

    results = await search_engine.search(
        project_id=project_id,
        query=query,
        query_vec=qv,
        fts_top_k=cfg.fts_top_k,
        vector_top_k=cfg.vector_top_k,
        return_k=max(1, min(50, int(max_results))),
        enable_mmr=bool(use_mmr and cfg.enable_mmr),
        mmr_lambda=float(cfg.mmr_lambda),
    )

    # Compact response (clients can format)
    return {
        "query": query,
        "project_id": project_id,
        "count": len(results),
        "results": [
            {
                "id": r.get("id"),
                "score": r.get("relevance_score"),
                "token_count": r.get("token_count"),
                "metadata": r.get("metadata"),
                "content": r.get("content"),
            }
            for r in results
        ],
    }


@mcp.tool
async def list_jobs() -> Dict[str, Any]:
    """List active background jobs (index/update)."""
    return await jobs.list()


@mcp.tool
async def cancel_job(job_id: str) -> str:
    """Cancel an in-progress index/update job by job_id."""
    ok = await jobs.cancel(job_id)
    return "✅ Cancelled" if ok else "❌ Job not found"


@mcp.tool
async def delete_project(project_id: str) -> str:
    """Delete a project and remove its indexes from disk."""
    await dbmod.init_db(cfg.db_path)
    proj = await dbmod.get_project(cfg.db_path, project_id)
    if not proj:
        return f"❌ Project not found: {project_id}"

    try:
        safe_id = _sanitize_project_id(project_id)
    except ValueError as e:
        return f"❌ {e}"
    lock = await get_project_lock(project_id)
    async with lock:
        await dbmod.delete_project(cfg.db_path, project_id)
        index_paths = [
            os.path.join(cfg.index_dir, f"{safe_id}.index"),
        ]
        shard_prefix = os.path.join(cfg.index_dir, f"{safe_id}.shard")
        for entry in os.listdir(cfg.index_dir):
            if entry.startswith(os.path.basename(shard_prefix)) and entry.endswith(".index"):
                index_paths.append(os.path.join(cfg.index_dir, entry))
        for path in index_paths:
            if os.path.exists(path):
                os.unlink(path)

    async with _embedder_lock:
        cached = _embedder_cache.pop(project_id, None)
        if cached:
            cached[2].close()
    return f"✅ Deleted {project_id}"


if __name__ == "__main__":
    # Stdio transport by default
    mcp.run()
