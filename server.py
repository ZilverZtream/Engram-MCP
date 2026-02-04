from __future__ import annotations

import atexit
import asyncio
import logging
import os
import signal
import time
from typing import Any, Dict, Optional

from fastmcp import FastMCP

from engram_mcp.config import load_config
from engram_mcp.embeddings import EmbeddingService
from engram_mcp.indexing import Indexer, ensure_index_permissions
from engram_mcp.locks import get_project_lock, get_project_rwlock, remove_project_lock, remove_project_rwlock
from engram_mcp.jobs import JobManager
from engram_mcp.search import SearchEngine, configure_numba
from engram_mcp.security import PathContext, PathNotAllowed, ProjectID, validate_project_field
from engram_mcp import db as dbmod
from engram_mcp import chunking


cfg = load_config()
dbmod.ensure_db_permissions(cfg.db_path)
configure_numba(bool(cfg.enable_numba), warmup=bool(cfg.enable_numba))

mcp = FastMCP(name="Engram MCP")

project_path_context = PathContext(cfg.allowed_roots)
index_path_context = PathContext([cfg.index_dir])
# Fix #11: honour the configured embedding back-end
embedding_service = EmbeddingService(
    prefer_thread_for_cuda=cfg.prefer_thread_for_cuda,
    worker_count=max(1, min(4, os.cpu_count() or 4)),
    embedding_backend=cfg.embedding_backend,
    ollama_model=cfg.ollama_model,
    ollama_url=cfg.ollama_url,
    openai_api_key=cfg.openai_api_key,
    openai_embedding_model=cfg.openai_embedding_model,
)
jobs = JobManager(cfg.db_path)
indexer = Indexer(cfg, embedding_service, project_path_context, index_path_context)
search_engine = SearchEngine(
    db_path=cfg.db_path,
    index_dir=cfg.index_dir,
    index_path_context=index_path_context,
)


async def _startup_tasks() -> None:
    await dbmod.init_db(cfg.db_path)
    await ensure_index_permissions(index_path_context, cfg.index_dir, db_path=cfg.db_path)
    await dbmod.prune_embedding_cache(
        cfg.db_path,
        ttl_s=int(cfg.embedding_cache_ttl_s),
        max_rows=int(cfg.embedding_cache_max_rows),
    )


def _schedule_startup_tasks() -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_startup_tasks())
        return
    loop.create_task(_startup_tasks())


_schedule_startup_tasks()


# ---------------------------------------------------------------------------
# Fix #6: coordinated shutdown – ensure embedding pools (thread + process)
# and the DB pool are torn down, and active jobs are cancelled.
# ---------------------------------------------------------------------------
_shutdown_lock = asyncio.Lock()
_shutdown_started = False


async def _shutdown(reason: str, *, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    global _shutdown_started
    async with _shutdown_lock:
        if _shutdown_started:
            return
        _shutdown_started = True
    logging.info("Shutdown initiated (%s)", reason)
    try:
        await asyncio.wait_for(jobs.shutdown(timeout_s=5.0), timeout=6.0)
        await asyncio.wait_for(embedding_service.close(graceful=True), timeout=10.0)
        await asyncio.wait_for(dbmod.close_db_pool(), timeout=5.0)
        loop = loop or asyncio.get_running_loop()
        if loop.is_running():
            pending = [
                task
                for task in asyncio.all_tasks(loop)
                if task is not asyncio.current_task(loop)
            ]
            for task in pending:
                task.cancel()
            try:
                await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=10.0)
            except asyncio.TimeoutError:
                logging.warning("Shutdown timed out waiting for pending tasks.")
    except Exception:
        logging.warning("Shutdown cleanup failed", exc_info=True)
    finally:
        if loop and loop.is_running():
            loop.stop()


def _sync_cleanup() -> None:
    """Best-effort shutdown of embedding pools and DB connections on exit."""
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(_shutdown("atexit", loop=loop), loop)
            try:
                fut.result(timeout=15.0)
            except Exception:
                logging.warning("Shutdown cleanup timed out", exc_info=True)
        else:
            asyncio.run(_shutdown("atexit"))
    except Exception:
        logging.warning("Shutdown cleanup failed", exc_info=True)


def _register_signal_handlers() -> bool:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return False
    if not hasattr(loop, "add_signal_handler"):
        return False
    try:
        loop.add_signal_handler(
            signal.SIGTERM,
            lambda: asyncio.create_task(_shutdown("SIGTERM", loop=loop)),
        )
        return True
    except (NotImplementedError, RuntimeError):
        return False


_uses_asyncio_signals = _register_signal_handlers()
atexit.register(_sync_cleanup)

if not _uses_asyncio_signals:
    _original_sigterm = signal.getsignal(signal.SIGTERM)

    def _sigterm_handler(sig: int, frame: Any) -> None:
        _sync_cleanup()
        if callable(_original_sigterm):
            _original_sigterm(sig, frame)  # type: ignore[arg-type]
        else:
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@mcp.tool
async def index_project(
    directory: str,
    project_name: str,
    project_type: str,
    wait: bool = True,
    dedupe_by_directory: bool = True,
) -> str:
    """Index a directory (within allowed_roots) for hybrid search."""
    try:
        directory = str(project_path_context.resolve_path(directory))
    except PathNotAllowed as e:
        return f"❌ {e}"
    try:
        project_name = validate_project_field(project_name, field_name="project_name")
        project_type = validate_project_field(project_type, field_name="project_type")
    except ValueError as e:
        return f"❌ {e}"

    async def _run() -> str:
        start = time.time()
        pid = await indexer.index_project(
            directory=directory,
            project_name=project_name,
            project_type=project_type,
            dedupe_by_directory=bool(dedupe_by_directory),
        )
        logging.info(
            "index_project",
            extra={
                "project_id": pid,
                "operation": "index",
                "project_name": project_name,
                "project_type": project_type,
                "duration_ms": int((time.time() - start) * 1000),
            },
        )
        return f"✅ Indexed '{project_name}' as project_id: {pid}"

    job = await jobs.create("index", _run())
    # Wait for completion (Claude expects tool result), but the event loop remains responsive.
    try:
        if not wait:
            return f"🆔 Index job queued: {job.job_id}"
        return await job.task
    except asyncio.CancelledError:
        return f"🛑 Index job cancelled: {job.job_id}"


@mcp.tool
async def update_project(project_id: str, wait: bool = True) -> str:
    """Update an existing project by scanning its stored directory."""

    async def _run() -> str:
        lock = await get_project_lock(project_id)
        rwlock = await get_project_rwlock(project_id)
        async with lock:
            async with rwlock.write_lock():
                start = time.time()
                changes = await indexer.update_project(project_id=project_id)
                logging.info(
                    "update_project",
                    extra={
                        "project_id": project_id,
                        "operation": "update",
                        "duration_ms": int((time.time() - start) * 1000),
                        "added": changes.get("added"),
                        "deleted": changes.get("deleted"),
                    },
                )
        return f"✅ Updated {project_id}: +{changes['added']} added, -{changes['deleted']} deleted"

    job = await jobs.create("update", _run())
    try:
        if not wait:
            return f"🆔 Update job queued: {job.job_id}"
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
    fts_mode: str = "strict",
    include_content: bool = True,
    max_content_chars_per_result: int = 1200,
) -> Dict[str, Any]:
    """Hybrid search: SQLite FTS5 (lexical) + FAISS (vector) with RRF, optional MMR.

    fts_mode: "strict" (AND), "any" (OR), or "phrase".
    """
    await dbmod.init_db(cfg.db_path)
    proj = await dbmod.get_project(cfg.db_path, project_id)
    if not proj:
        return {"error": f"Project not found: {project_id}"}
    max_chars = int(getattr(cfg, "max_query_chars", 4096))
    max_tokens = int(getattr(cfg, "max_query_tokens", 256))
    if len(query) > max_chars:
        return {"error": f"Query too long ({len(query)} chars). Max is {max_chars}."}
    token_count = chunking.token_count(query)
    if token_count > max_tokens:
        return {"error": f"Query too long ({token_count} tokens). Max is {max_tokens}."}

    md = proj.get("metadata") or {}
    model_name = md.get("model_name") or cfg.model_name_text
    device = md.get("device") or _device()
    rwlock = await get_project_rwlock(project_id)
    start = time.time()
    vector_enabled = bool(search_engine.faiss_available)
    async with rwlock.read_lock():
        qv = None
        if vector_enabled:
            qv = await embedding_service.embed_one(query, model_name=model_name, device=device)

        try:
            results = await search_engine.search(
                project_id=project_id,
                query=query,
                query_vec=qv,
                fts_top_k=cfg.fts_top_k,
                vector_top_k=cfg.vector_top_k,
                return_k=max(1, min(50, int(max_results))),
                enable_mmr=bool(use_mmr and cfg.enable_mmr and vector_enabled),
                mmr_lambda=float(cfg.mmr_lambda),
                fts_mode=fts_mode,
            )
        except ValueError as exc:
            return {"error": str(exc)}
    logging.info(
        "search_memory",
        extra={
            "project_id": project_id,
            "operation": "search",
            "duration_ms": int((time.time() - start) * 1000),
            "result_count": len(results),
        },
    )

    # Compact response (clients can format)
    def _trim_content(text: str) -> str:
        if not include_content:
            return ""
        max_len = max(0, int(max_content_chars_per_result))
        if max_len and len(text) > max_len:
            return text[:max_len] + "…"
        return text

    return {
        "query": query,
        "project_id": project_id,
        "vector_enabled": vector_enabled,
        "count": len(results),
        "results": [
            {
                "id": r.get("id"),
                "score": r.get("relevance_score"),
                "token_count": r.get("token_count"),
                "metadata": r.get("metadata"),
                **({"content": _trim_content(r.get("content") or "")} if include_content else {}),
            }
            for r in results
        ],
    }


@mcp.tool
async def get_chunk(project_id: str, chunk_id: str, include_content: bool = True) -> Dict[str, Any]:
    """Fetch a single chunk by ID (full content by default)."""
    await dbmod.init_db(cfg.db_path)
    chunk = await dbmod.fetch_chunk_by_id(cfg.db_path, project_id=project_id, chunk_id=chunk_id)
    if not chunk:
        return {"error": f"Chunk not found: {chunk_id}"}
    if not include_content:
        chunk = {k: v for k, v in chunk.items() if k != "content"}
    return chunk


@mcp.tool
async def project_health(project_id: str) -> Dict[str, Any]:
    """Report index/DB health for a project."""
    await dbmod.init_db(cfg.db_path)
    proj = await dbmod.get_project(cfg.db_path, project_id)
    if not proj:
        return {"error": f"Project not found: {project_id}"}
    if not search_engine.faiss_available:
        return {
            "project_id": project_id,
            "index_dirty": bool(proj.get("index_dirty")),
            "faiss_index_uuid": proj.get("faiss_index_uuid"),
            "chunk_count_db": int(proj.get("chunk_count") or 0),
            "faiss_ntotal": 0,
            "missing_or_unreadable_indexes": [],
            "uuid_mismatches": [],
            "vector_status": "vector search unavailable (faiss not installed)",
            "updated_at": proj.get("updated_at"),
        }
    rwlock = await get_project_rwlock(project_id)
    async with rwlock.read_lock():
        metadata = proj.get("metadata") or {}
        shard_count = int(metadata.get("shard_count") or 1)
        index_uuid = proj.get("faiss_index_uuid")
        index_paths = []
        if shard_count <= 1:
            index_paths.append(os.path.join(cfg.index_dir, f"{project_id}.index.current"))
        else:
            index_paths.extend(
                os.path.join(cfg.index_dir, f"{project_id}.shard{shard_id}.index.current")
                for shard_id in range(shard_count)
            )

        faiss_totals = []
        uuid_mismatches = []
        missing_files = []
        for path in index_paths:
            if not index_path_context.exists(path):
                missing_files.append(path)
                continue
            uuid_path = path + ".uuid"
            if index_path_context.exists(uuid_path) and index_uuid:
                try:
                    with index_path_context.open_file(uuid_path, "r", encoding="utf-8") as uf:
                        on_disk_uuid = uf.read().strip()
                    if on_disk_uuid and on_disk_uuid != index_uuid:
                        uuid_mismatches.append({"path": path, "on_disk": on_disk_uuid, "expected": index_uuid})
                except Exception as exc:
                    uuid_mismatches.append({"path": path, "error": str(exc)})
            try:
                import faiss

                index = faiss.read_index(path, faiss.IO_FLAG_MMAP)
                faiss_totals.append(int(getattr(index, "ntotal", 0)))
            except Exception as exc:
                missing_files.append(f"{path} ({exc})")
                continue

    return {
        "project_id": project_id,
        "index_dirty": bool(proj.get("index_dirty")),
        "faiss_index_uuid": index_uuid,
        "chunk_count_db": int(proj.get("chunk_count") or 0),
        "faiss_ntotal": int(sum(faiss_totals)) if faiss_totals else 0,
        "missing_or_unreadable_indexes": missing_files,
        "uuid_mismatches": uuid_mismatches,
        "vector_status": "ok",
        "updated_at": proj.get("updated_at"),
    }


@mcp.tool
async def repair_project(project_id: str) -> str:
    """Rebuild the FAISS index from DB state for a project."""
    if not search_engine.faiss_available:
        return "❌ Vector search unavailable (faiss not installed). Install faiss and retry."
    await dbmod.init_db(cfg.db_path)
    proj = await dbmod.get_project(cfg.db_path, project_id)
    if not proj:
        return f"❌ Project not found: {project_id}"
    metadata = proj.get("metadata") or {}
    lock = await get_project_lock(project_id)
    rwlock = await get_project_rwlock(project_id)
    async with lock:
        async with rwlock.write_lock():
            await indexer._rebuild_index_from_db(project_id=project_id, metadata=metadata)
    return f"✅ Rebuilt FAISS index for {project_id}"


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
        ProjectID(project_id)
    except ValueError as e:
        return f"❌ {e}"
    lock = await get_project_lock(project_id)
    rwlock = await get_project_rwlock(project_id)
    async with lock:
        async with rwlock.write_lock():
            await dbmod.set_project_deleting(cfg.db_path, project_id, True)
            artifacts = await dbmod.list_project_artifacts(cfg.db_path, project_id)
            failures = []
            for artifact in artifacts:
                path = artifact.get("artifact_path")
                if not path:
                    continue
                if index_path_context.exists(path):
                    try:
                        index_path_context.unlink(path)
                    except Exception as exc:
                        failures.append(f"{path} ({exc})")
            try:
                for path in index_path_context.iter_files(cfg.index_dir):
                    name = path.name
                    if name.startswith(f"{project_id}.") and ".index.current" in name:
                        try:
                            index_path_context.unlink(path)
                        except Exception as exc:
                            failures.append(f"{path} ({exc})")
            except Exception as exc:
                failures.append(f"fallback cleanup ({exc})")
            if failures:
                return "❌ Failed to delete index artifacts: " + ", ".join(failures)
            await dbmod.delete_project(cfg.db_path, project_id)
    await remove_project_lock(project_id)
    await remove_project_rwlock(project_id)
    return f"✅ Deleted {project_id}"


if __name__ == "__main__":
    # Stdio transport by default
    mcp.run()
