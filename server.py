from __future__ import annotations

import atexit
import asyncio
import hashlib
import importlib.util
import logging
import os
import signal
import stat
import time
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from engram_mcp.config import load_config
from engram_mcp.embeddings import EmbeddingService
from engram_mcp.indexing import Indexer, ensure_index_permissions
from engram_mcp.locks import get_project_lock, get_project_rwlock, remove_project_lock, remove_project_rwlock
from engram_mcp.jobs import JobManager
from engram_mcp.search import (
    SearchEngine,
    configure_numba,
    configure_search_cache,
    invalidate_search_cache_project,
    is_faiss_available,
)
from engram_mcp.dreaming import identify_candidates
from engram_mcp.generation import GenerationService
from engram_mcp.security import PathContext, PathNotAllowed, ProjectID, validate_project_field
from engram_mcp import db as dbmod
from engram_mcp import chunking
from engram_mcp.parsing import SUPPORTED_DOC_EXTS, SUPPORTED_TEXT_EXTS, iter_files


cfg = load_config()
dbmod.ensure_db_permissions(cfg.db_path)
if cfg.enable_numba:
    logging.info("Numba enabled; compiling optimization kernels...")
configure_numba(bool(cfg.enable_numba), warmup=bool(cfg.enable_numba))
configure_search_cache(ttl_s=float(cfg.search_cache_ttl_s), max_items=int(cfg.search_cache_max_items))

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
    openai_api_key=cfg.openai_api_key.get_secret_value(),
    openai_embedding_model=cfg.openai_embedding_model,
)
jobs = JobManager(cfg.db_path, max_queue_per_key=cfg.max_jobs_per_project)
indexer = Indexer(cfg, embedding_service, project_path_context, index_path_context)

_AUTO_DREAM_SEARCH_INTERVAL = 25
_AUTO_DREAM_MIN_INTERVAL_S = 300.0
_AUTO_DREAM_DAY_S = 24 * 60 * 60
MAX_DREAM_PAIRS = 100
DEFAULT_MAX_CONTENT_CHARS = 1200
MAX_CONTENT_CHARS = 100_000
_auto_dream_state: Dict[str, Dict[str, float]] = {}
_auto_dream_lock = asyncio.Lock()


def _faiss_gpu_available() -> bool:
    if not is_faiss_available():
        return False
    import faiss

    if hasattr(faiss, "get_num_gpus"):
        return faiss.get_num_gpus() > 0
    return False


def _resolve_vector_backend() -> tuple[str, bool, str]:
    requested = str(cfg.vector_backend or "auto").lower()
    if requested == "fts":
        return "fts", False, "disabled by config"
    if not is_faiss_available():
        return "fts", False, "faiss not installed"
    if requested == "faiss_gpu":
        if _faiss_gpu_available():
            return "faiss_gpu", True, "GPU enabled"
        return "faiss_cpu", True, "GPU unavailable; using CPU"
    if requested == "faiss_cpu":
        return "faiss_cpu", True, "CPU enabled"
    if _faiss_gpu_available():
        return "faiss_gpu", True, "auto (GPU available)"
    return "faiss_cpu", True, "auto (CPU)"


_vector_backend, _vector_enabled, _vector_reason = _resolve_vector_backend()
search_engine = SearchEngine(
    db_path=cfg.db_path,
    index_dir=cfg.index_dir,
    index_path_context=index_path_context,
    faiss_available=_vector_enabled,
    vector_backend=_vector_backend,
)


async def _startup_tasks() -> None:
    await dbmod.init_db(cfg.db_path)
    await ensure_index_permissions(index_path_context, cfg.index_dir, db_path=cfg.db_path)
    await dbmod.prune_embedding_cache(
        cfg.db_path,
        ttl_s=int(cfg.embedding_cache_ttl_s),
        max_rows=int(cfg.embedding_cache_max_rows),
    )
    _log_startup_status()


def _log_startup_status() -> None:
    logging.info("Storage: db=%s index_dir=%s", cfg.db_path, cfg.index_dir)
    if _vector_enabled:
        logging.info("Vector search: enabled (%s).", _vector_backend)
    else:
        logging.info(
            "Vector search: disabled (%s). Install: pipx install \"engram-mcp[cpu]\"",
            _vector_reason,
        )
    if cfg.enable_numba:
        logging.info("Numba optimizations: enabled.")
    else:
        logging.info("Numba optimizations: disabled (set enable_numba: true to enable).")
    if cfg.search_cache_ttl_s and cfg.search_cache_max_items:
        logging.info(
            "Search cache: enabled (ttl=%ss max_items=%s).",
            cfg.search_cache_ttl_s,
            cfg.search_cache_max_items,
        )
    else:
        logging.info("Search cache: disabled.")


def _schedule_startup_tasks() -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_startup_tasks())
        return

    task = loop.create_task(_startup_tasks())

    def _on_startup_done(t: "asyncio.Task[None]") -> None:
        exc = t.exception()
        if exc is None:
            return
        # Log with full traceback so the operator sees *why* startup failed,
        # then stop the loop so the process does not linger in a half-
        # initialised "zombie" state where all tool calls fail.
        logging.critical(
            "Startup tasks failed; the server cannot serve requests.",
            exc_info=exc,
        )
        loop.stop()

    task.add_done_callback(_on_startup_done)


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
    if importlib.util.find_spec("torch") is None:
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _scan_project_limits(directory: str) -> tuple[int, int]:
    max_files = int(cfg.max_project_files)
    max_bytes = int(cfg.max_project_bytes)
    total_files = 0
    total_bytes = 0
    for path in iter_files(project_path_context, directory, cfg.ignore_patterns):
        ext = path.suffix.lower()
        if ext not in SUPPORTED_TEXT_EXTS and ext not in SUPPORTED_DOC_EXTS:
            continue
        stat_result = project_path_context.stat(path)
        total_files += 1
        total_bytes += stat_result.st_size
        if max_files > 0 and total_files > max_files:
            raise ValueError(
                f"Project exceeds file limit ({max_files} files). "
                "Reduce the directory size or increase max_project_files."
            )
        if max_bytes > 0 and total_bytes > max_bytes:
            raise ValueError(
                f"Project exceeds size limit ({max_bytes} bytes). "
                "Reduce the directory size or increase max_project_bytes."
            )
    return total_files, total_bytes


async def _enforce_project_limits(directory: str) -> None:
    max_files = int(cfg.max_project_files)
    max_bytes = int(cfg.max_project_bytes)
    if max_files <= 0 and max_bytes <= 0:
        return
    await asyncio.to_thread(_scan_project_limits, directory)


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

    try:
        await _enforce_project_limits(directory)
    except ValueError as e:
        return f"❌ {e}"
    job = await jobs.create("index", _run(), queue_key=directory)
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

    await dbmod.init_db(cfg.db_path)
    proj = await dbmod.get_project(cfg.db_path, project_id)
    directory = proj.get("directory") if proj else None
    if directory:
        try:
            directory = str(project_path_context.resolve_path(directory))
            await _enforce_project_limits(directory)
        except (PathNotAllowed, ValueError) as e:
            return f"❌ {e}"
    job = await jobs.create("update", _run(), queue_key=project_id)
    try:
        if not wait:
            return f"🆔 Update job queued: {job.job_id}"
        return await job.task
    except asyncio.CancelledError:
        return f"🛑 Update job cancelled: {job.job_id}"


async def _run_dream_cycle(project_id: str, *, max_pairs: int = 10) -> str:
    candidates = await identify_candidates(project_id)
    if not candidates:
        return "ℹ️ No dream candidates found."
    max_pairs_int = min(MAX_DREAM_PAIRS, max(1, int(max_pairs)))
    candidates = candidates[:max_pairs_int]

    unique_ids: List[str] = []
    seen_ids: set[str] = set()
    for first, second, _count in candidates:
        for cid in (str(first), str(second)):
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            unique_ids.append(cid)

    chunk_rows = await dbmod.fetch_chunks_by_ids(
        cfg.db_path, project_id=project_id, chunk_ids=unique_ids
    )
    chunk_map = {str(row["id"]): row for row in chunk_rows}

    worker_count = max(1, min(4, os.cpu_count() or 4))
    generation = GenerationService(
        prefer_thread_for_cuda=cfg.prefer_thread_for_cuda,
        worker_count=worker_count,
    )
    insights: List[chunking.Chunk] = []
    semaphore = asyncio.Semaphore(worker_count)
    max_context_window = max(256, int(os.getenv("ENGRAM_DREAM_CONTEXT_WINDOW", "2048")))
    per_context_budget = max(64, int(max_context_window * 0.75) // 2)
    try:
        async def _generate_single_insight(candidate: tuple[str, str, int]) -> Optional[chunking.Chunk]:
            first, second, _count = candidate
            cid_a, cid_b = str(first), str(second)
            row_a = chunk_map.get(cid_a)
            row_b = chunk_map.get(cid_b)
            if not row_a or not row_b:
                return None
            context_a = row_a.get("content") or ""
            context_b = row_b.get("content") or ""
            if not context_a.strip() or not context_b.strip():
                return None
            context_a = chunking.truncate_text_to_tokens(context_a, per_context_budget)
            context_b = chunking.truncate_text_to_tokens(context_b, per_context_budget)
            if not context_a.strip() or not context_b.strip():
                return None
            async with semaphore:
                try:
                    insight = await asyncio.wait_for(
                        generation.generate_insight(
                            context_a,
                            context_b,
                            model_name=cfg.dream_model_name,
                            device=_device(),
                        ),
                        timeout=30,
                    )
                except asyncio.TimeoutError:
                    return None
            insight = (insight or "").strip()
            if not insight or "NO_INSIGHT" in insight.upper():
                return None
            sorted_ids = sorted([cid_a, cid_b])
            insight_id = chunking.make_chunk_id("insight", project_id, *sorted_ids)
            return chunking.Chunk(
                insight_id,
                insight,
                chunking.token_count(insight),
                {"type": "insight", "source_chunks": sorted_ids},
            )

        results = await asyncio.gather(
            *[_generate_single_insight(candidate) for candidate in candidates],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logging.warning("Dream insight generation failed: %s", result)
                continue
            if result is None:
                continue
            insights.append(result)
    finally:
        await generation.close()

    if not insights:
        return "ℹ️ No insights generated."

    vfs_path = "vfs://insights/dream_cycle.md"
    added = await indexer.add_virtual_file_chunks(
        project_id=project_id,
        file_path=vfs_path,
        chunks=insights,
    )
    return f"✅ Dreamed {added} insights from {len(candidates)} pairs."


async def _maybe_auto_trigger_dream(project_id: str) -> None:
    if not cfg.enable_dreaming or not cfg.auto_dream_enabled:
        return
    now = time.time()
    async with _auto_dream_lock:
        state = _auto_dream_state.setdefault(
            project_id, {"count": 0.0, "last": 0.0, "day": 0.0, "runs": 0.0}
        )
        day_marker = float(int(now // _AUTO_DREAM_DAY_S))
        if state["day"] != day_marker:
            state["day"] = day_marker
            state["runs"] = 0.0
        state["count"] += 1.0
        if state["count"] < _AUTO_DREAM_SEARCH_INTERVAL:
            return
        if now - state["last"] < _AUTO_DREAM_MIN_INTERVAL_S:
            return
        if state["runs"] >= float(cfg.auto_dream_max_runs_per_day):
            return
        state["count"] = 0.0
        state["last"] = now
        state["runs"] += 1.0
    try:
        ProjectID(project_id)
    except ValueError:
        return
    await dbmod.init_db(cfg.db_path)
    proj = await dbmod.get_project(cfg.db_path, project_id)
    if not proj:
        return
    max_pairs = min(MAX_DREAM_PAIRS, max(1, int(cfg.auto_dream_max_pairs)))
    job = await jobs.create(
        "dream",
        _run_dream_cycle(project_id, max_pairs=max_pairs),
        queue_key=f"dream:{project_id}",
    )
    logging.info(
        "auto_dream_triggered",
        extra={
            "project_id": project_id,
            "job_id": job.job_id,
            "interval": _AUTO_DREAM_SEARCH_INTERVAL,
        },
    )


@mcp.tool
async def dream_project(project_id: str, wait: bool = False, max_pairs: int = 10) -> str:
    """Generate insight chunks from recent search co-occurrences."""
    if not cfg.enable_dreaming:
        return "❌ Dreaming is disabled in the configuration."
    try:
        ProjectID(project_id)
    except ValueError as e:
        return f"❌ {e}"
    await dbmod.init_db(cfg.db_path)
    proj = await dbmod.get_project(cfg.db_path, project_id)
    if not proj:
        return f"❌ Project not found: {project_id}"

    job = await jobs.create(
        "dream",
        _run_dream_cycle(project_id, max_pairs=max_pairs),
        queue_key=f"dream:{project_id}",
    )
    try:
        if not wait:
            return f"🆔 Dream job queued: {job.job_id}"
        return await job.task
    except asyncio.CancelledError:
        return f"🛑 Dream job cancelled: {job.job_id}"


@mcp.tool
async def trigger_rem_cycle(project_id: str) -> str:
    """Enqueue a REM-style dream cycle for a project."""
    if not cfg.enable_dreaming:
        return "❌ Dreaming is disabled in the configuration."
    try:
        ProjectID(project_id)
    except ValueError as e:
        return f"❌ {e}"
    await dbmod.init_db(cfg.db_path)
    proj = await dbmod.get_project(cfg.db_path, project_id)
    if not proj:
        return f"❌ Project not found: {project_id}"
    job = await jobs.create("dream", _run_dream_cycle(project_id), queue_key=f"dream:{project_id}")
    return f"🆔 Dream job queued: {job.job_id}"


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

    max_content_chars = int(max_content_chars_per_result)
    if max_content_chars < 0:
        max_content_chars = DEFAULT_MAX_CONTENT_CHARS
    max_content_chars = min(max_content_chars, MAX_CONTENT_CHARS)

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
    try:
        await _maybe_auto_trigger_dream(project_id)
    except Exception:
        logging.warning("Auto dream trigger failed", exc_info=True)

    # Compact response (clients can format)
    def _trim_content(text: str) -> str:
        if not include_content:
            return ""
        if max_content_chars and len(text) > max_content_chars:
            return text[:max_content_chars] + "…"
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
    if include_content:
        max_chars = MAX_CONTENT_CHARS
        content = chunk.get("content")
        if content and len(content) > max_chars:
            chunk = dict(chunk)
            chunk["content"] = content[:max_chars] + "…"
    else:
        chunk = {k: v for k, v in chunk.items() if k != "content"}
    return chunk


@mcp.tool
async def query_graph_nodes(
    project_id: str,
    node_type: str = "",
    name_pattern: str = "",
    file_path: str = "",
    limit: int = 100,
) -> Dict[str, Any]:
    """List functions/classes extracted from the code graph.

    node_type   – filter to ``function`` or ``class`` (empty = both).
    name_pattern – SQL LIKE pattern; e.g. ``%parse%`` matches any name
                   containing "parse".  Wrap in ``%`` for substring match.
    file_path   – restrict results to a single source file.
    limit       – max rows returned (cap 500).
    """
    await dbmod.init_db(cfg.db_path)
    proj = await dbmod.get_project(cfg.db_path, project_id)
    if not proj:
        return {"error": f"Project not found: {project_id}"}
    limit = max(1, min(500, int(limit)))
    nodes = await dbmod.query_graph_nodes(
        cfg.db_path,
        project_id=project_id,
        node_type=node_type or None,
        name_pattern=name_pattern or None,
        file_path=file_path or None,
        limit=limit,
    )
    return {"project_id": project_id, "count": len(nodes), "nodes": nodes}


@mcp.tool
async def find_references(project_id: str, node_id: str) -> Dict[str, Any]:
    """Find all graph edges (callers / callees / containment) for a node.

    node_id is the value returned by ``query_graph_nodes`` (format
    ``<file_path>:<node_type>:<name>``).

    Returns:
      outgoing – edges where this node is the *source* (e.g. calls)
      incoming – edges where this node is the *target* (e.g. called_by)

    Each edge entry includes the peer node_id and edge_type.  Resolve
    peer details by passing the peer node_id back into ``query_graph_nodes``
    with a matching name_pattern.
    """
    await dbmod.init_db(cfg.db_path)
    proj = await dbmod.get_project(cfg.db_path, project_id)
    if not proj:
        return {"error": f"Project not found: {project_id}"}
    edges = await dbmod.fetch_graph_edges(cfg.db_path, project_id=project_id, node_id=node_id)
    # Resolve peer node details for all referenced node_ids in one batch.
    peer_ids: List[str] = []
    seen: set[str] = set()
    for e in edges["outgoing"]:
        tid = e["target_id"]
        if tid not in seen:
            peer_ids.append(tid)
            seen.add(tid)
    for e in edges["incoming"]:
        sid = e["source_id"]
        if sid not in seen:
            peer_ids.append(sid)
            seen.add(sid)
    # Look up each peer.  node_id == "<file>:<type>:<name>" – extract name
    # and do an exact match via the name_pattern filter.
    peer_map: Dict[str, Dict[str, Any]] = {}
    for pid in peer_ids:
        parts = pid.split(":", 2)
        if len(parts) == 3:
            fp, _nt, _nm = parts
            peers = await dbmod.query_graph_nodes(
                cfg.db_path, project_id=project_id, file_path=fp, limit=500
            )
            for p in peers:
                peer_map[p["node_id"]] = p
    # Annotate edges with resolved peer info.
    for e in edges["outgoing"]:
        peer = peer_map.get(e["target_id"])
        if peer:
            e["target"] = peer
    for e in edges["incoming"]:
        peer = peer_map.get(e["source_id"])
        if peer:
            e["source"] = peer
    return {
        "project_id": project_id,
        "node_id": node_id,
        "outgoing": edges["outgoing"],
        "incoming": edges["incoming"],
    }


@mcp.tool
async def graph_search(
    project_id: str,
    query: str,
    max_results: int = 10,
    symbol_boost: float = 0.03,
) -> Dict[str, Any]:
    """Search that combines text search with code-graph symbol awareness.

    1. Extracts potential symbol names from *query*.
    2. Looks up those names in graph_nodes to find which files contain
       matching symbols.
    3. Runs a normal FTS search.
    4. Boosts the RRF score of any chunk whose source file contains a
       matching symbol by *symbol_boost* (default 0.03, ~15x the insight
       bonus used elsewhere).
    5. Returns results with a ``graph_match`` flag and, where applicable,
       the matched node details.
    """
    await dbmod.init_db(cfg.db_path)
    proj = await dbmod.get_project(cfg.db_path, project_id)
    if not proj:
        return {"error": f"Project not found: {project_id}"}

    # --- Extract candidate symbol tokens (alphanumeric + _ sequences that
    # look like identifiers; skip very short or all-lowercase common words).
    import re as _re

    raw_tokens = _re.findall(r'[A-Za-z_]\w*', query)
    # Keep tokens that are either camelCase, contain underscore, start with
    # upper-case, or are long enough to be interesting (>= 4 chars).
    candidates = [
        t for t in raw_tokens
        if len(t) >= 4
        or '_' in t
        or (len(t) >= 2 and t[0].isupper())
        or any(c.isupper() for c in t[1:])  # camelCase interior caps
    ]

    # --- Graph: find nodes whose names match any candidate ---
    symbol_files: Dict[str, List[Dict[str, Any]]] = {}  # file_path → [node, …]
    for sym in candidates:
        pattern = f"%{sym}%"
        nodes = await dbmod.query_graph_nodes(
            cfg.db_path, project_id=project_id, name_pattern=pattern, limit=50
        )
        for n in nodes:
            symbol_files.setdefault(n["file_path"], []).append(n)

    # --- Text search (FTS, no vector needed – keeps this tool lightweight) ---
    max_chars = int(getattr(cfg, "max_query_chars", 4096))
    if len(query) > max_chars:
        return {"error": f"Query too long ({len(query)} chars). Max is {max_chars}."}

    fts_results = await dbmod.fts_search(
        cfg.db_path,
        project_id=project_id,
        query=query,
        limit=max(1, min(50, int(max_results) * 3)),  # over-fetch for re-ranking
        mode="any",
    )

    # --- Apply symbol boost to chunks from files with matching symbols ---
    boosted: List[Dict[str, Any]] = []
    for r in fts_results:
        meta = r.get("metadata") or {}
        file_path = meta.get("item_name") or meta.get("path") or ""
        # Normalise: item_name is the relative path stored during chunking.
        matched_nodes = symbol_files.get(file_path, [])
        base_score = float(r.get("lex_score", 0.0))
        # bm25 scores from FTS5 are negative (lower = better); invert for
        # a positive relevance score, then add the boost.
        relevance = -base_score + (symbol_boost if matched_nodes else 0.0)
        boosted.append(
            {
                "id": r.get("id"),
                "score": round(relevance, 6),
                "content": r.get("content", ""),
                "metadata": meta,
                "graph_match": bool(matched_nodes),
                "matched_symbols": [
                    {"name": n["name"], "node_type": n["node_type"], "node_id": n["node_id"]}
                    for n in matched_nodes
                ] if matched_nodes else [],
            }
        )

    # Sort descending by score, trim to max_results.
    boosted.sort(key=lambda x: x["score"], reverse=True)
    boosted = boosted[: max(1, min(50, int(max_results)))]

    # Trim content for response size.
    max_content = 1200
    for r in boosted:
        c = r.get("content") or ""
        if len(c) > max_content:
            r["content"] = c[:max_content] + "…"

    return {
        "query": query,
        "project_id": project_id,
        "symbol_files_matched": len(symbol_files),
        "count": len(boosted),
        "results": boosted,
    }


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
        import faiss

        for path in index_paths:
            try:
                safe_path = index_path_context.ensure_allowed_nofollow(path)
                st = index_path_context.lstat(safe_path)
            except Exception as exc:
                missing_files.append(f"{path} ({exc})")
                continue
            if stat.S_ISLNK(st.st_mode):
                missing_files.append(f"{safe_path} (symlink not allowed)")
                continue
            uuid_path = safe_path + ".uuid"
            if index_path_context.exists(uuid_path) and index_uuid:
                try:
                    with index_path_context.open_file(uuid_path, "r", encoding="utf-8") as uf:
                        on_disk_uuid = uf.read().strip()
                    if on_disk_uuid and on_disk_uuid != index_uuid:
                        uuid_mismatches.append({"path": path, "on_disk": on_disk_uuid, "expected": index_uuid})
                except Exception as exc:
                    uuid_mismatches.append({"path": path, "error": str(exc)})
            # Verify index-file integrity against the stored SHA-256
            # checksum.  Detects silent bit-rot before faiss touches the
            # data (a corrupted mmap can segfault inside the C++ layer).
            checksum_path = safe_path + ".checksum"
            if index_path_context.exists(checksum_path):
                try:
                    with index_path_context.open_file(checksum_path, "r", encoding="utf-8") as cf:
                        stored_checksum = cf.read().strip()
                    h = hashlib.sha256()
                    with index_path_context.open_file(safe_path, "rb") as f:
                        while True:
                            buf = f.read(1024 * 1024)
                            if not buf:
                                break
                            h.update(buf)
                    actual_checksum = h.hexdigest()
                    if stored_checksum != actual_checksum:
                        missing_files.append(
                            f"{safe_path} (checksum mismatch: "
                            f"stored={stored_checksum[:16]}… actual={actual_checksum[:16]}…)"
                        )
                        continue
                except Exception as exc:
                    missing_files.append(f"{safe_path} (checksum verification failed: {exc})")
                    continue
            try:
                index = faiss.read_index(safe_path, faiss.IO_FLAG_MMAP)
                faiss_totals.append(int(getattr(index, "ntotal", 0)))
            except Exception as exc:
                missing_files.append(f"{safe_path} ({exc})")
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
    await invalidate_search_cache_project(project_id)
    return f"✅ Rebuilt FAISS index for {project_id}"


@mcp.tool
async def list_jobs() -> Dict[str, Any]:
    """List active background jobs (index/update/dream)."""
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
            await search_engine.unload_project(project_id)
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
                for name in index_path_context.list_dir(cfg.index_dir):
                    if name.startswith(f"{project_id}.") and ".index.current" in name:
                        path = os.path.join(cfg.index_dir, name)
                        try:
                            index_path_context.unlink(path)
                        except Exception as exc:
                            failures.append(f"{path} ({exc})")
            except Exception as exc:
                failures.append(f"fallback cleanup ({exc})")
            if failures:
                # Roll back the deleting flag so the project is not stuck in
                # limbo and can be retried on the next call.
                await dbmod.set_project_deleting(cfg.db_path, project_id, False)
                return "❌ Failed to delete index artifacts: " + ", ".join(failures)
            await dbmod.delete_project(cfg.db_path, project_id)
            await invalidate_search_cache_project(project_id)
    await remove_project_lock(project_id)
    await remove_project_rwlock(project_id)
    return f"✅ Deleted {project_id}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    # Stdio transport by default
    main()
