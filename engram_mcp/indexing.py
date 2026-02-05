from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import math
import os
import re
import threading as _threading
import time
import uuid
import logging
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import EngramConfig
from .embeddings import EmbeddingService
from .parsing import (
    ParsedFile,
    iter_files,
    parse_file_to_chunks,
    hash_and_parse_file,
    SUPPORTED_DOC_EXTS,
    SUPPORTED_TEXT_EXTS,
    hash_file,
)

# Maximum chunks persisted (embedded + written) in a single batch.
# Keeps per-file peak memory bounded: 100 chunks × 384-dim float32 ≈ 150 KB
# of vectors regardless of file size.
_PERSIST_BATCH_SIZE = 100

# Upper bound on file-parsing / embedding concurrency.  On high-core-count
# servers (64 cores) defaulting to cpu_count causes massive I/O contention.
_CONCURRENCY_CAP = 8


def _capped_concurrency() -> int:
    """Return file-parsing concurrency capped at _CONCURRENCY_CAP."""
    return min(os.cpu_count() or 4, _CONCURRENCY_CAP)


from .locks import get_project_lock, get_project_rwlock
from .search import invalidate_faiss_cache, invalidate_search_cache_project, is_faiss_available
from .security import PathContext, ProjectID, validate_project_field
from . import db as dbmod

# Per-file parse timeout in seconds.
_PARSE_TIMEOUT_S = 30

# Shared executor for file-parsing tasks.  Sized at 3 × the concurrency cap
# so that a burst of timed-out-but-still-running threads (CPython cannot
# forcibly terminate threads) does not starve new submissions.
# _bounded_gather limits active parse coroutines to _CONCURRENCY_CAP, so at
# most that many threads are *actively used* at any instant; the extra
# headroom absorbs threads that are draining after their timeout expired.
_PARSE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=_CONCURRENCY_CAP * 3,
    thread_name_prefix="engram-parse",
)


class _CancellationToken:
    """Lightweight signal shared between the async caller and the worker thread.

    The token is passed into the wrapper closure submitted to the executor.
    When ``asyncio.wait_for`` raises *TimeoutError* the async side calls
    ``cancel()``; the worker thread checks ``is_cancelled()`` after *fn*
    returns.  Because CPython cannot forcibly terminate a thread the token
    does **not** interrupt a blocking call inside *fn* — it is a cooperative
    mechanism that:

    1. Lets the worker log and discard its result so stale data is never
       returned.
    2. Provides an observable hook for metrics / alerting on thread-leak
       duration.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = _threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


class _ParseTimeout(Exception):
    """Raised when a parse task exceeds its wall-clock budget."""


def _cancellable_wrapper(token: _CancellationToken, fn, *args):
    """Execute *fn* and raise _ParseTimeout if the token was cancelled before fn returned."""
    result = fn(*args)
    if token.is_cancelled():
        # The async side has already timed out and raised; discard the result
        # and log so operators can track how long the leaked thread ran.
        logging.warning(
            "Parse thread completed after its timeout was already raised; result discarded."
        )
        raise _ParseTimeout("discarded: token cancelled before return")
    return result


async def _run_in_thread_with_timeout(fn, *args, timeout: float = _PARSE_TIMEOUT_S):
    """Run *fn* in the shared parse executor with a wall-clock timeout.

    A ``_CancellationToken`` is shared between this coroutine and the
    worker thread.  On timeout the token is cancelled so the thread can
    detect — after *fn* returns naturally — that its result is no longer
    needed.  CPython provides no portable mechanism to kill a running
    thread, so truly stuck work (e.g. an infinite loop inside a C
    extension) persists until process exit; the shared pool is bounded at
    ``_CONCURRENCY_CAP * 3`` workers, capping the total number of leaked
    threads regardless of input size.
    """
    loop = asyncio.get_running_loop()
    token = _CancellationToken()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_PARSE_EXECUTOR, _cancellable_wrapper, token, fn, *args),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        token.cancel()
        raise asyncio.TimeoutError(f"Parse timed out after {timeout}s")


def _write_index_uuid(path_context: PathContext, index_path: str, uuid_str: str) -> None:
    """Write a companion .uuid file next to a FAISS index.

    On load (search.py) the UUID is compared against the project metadata
    row.  A mismatch means the DB and the index have diverged and stale
    vectors must not be served.
    """
    uuid_path = index_path + ".uuid"
    path_context.write_text_atomic(uuid_path, uuid_str, encoding="utf-8")


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "project"


def _pick_model_name(project_type: str, cfg: EngramConfig) -> str:
    pt = (project_type or "").lower()
    if "code" in pt:
        return cfg.model_name_code
    return cfg.model_name_text


def _content_hash(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


async def _bounded_gather(items: Iterable[Any], worker, concurrency: int):
    """Yield worker(item) results with at most *concurrency* live tasks.

    Unlike the previous TaskGroup-based design this never materialises more
    than *concurrency* asyncio.Task objects at once, so pointing it at a
    directory with 100 000 files does not allocate 100 000 coroutines up-front.
    """
    concurrency = max(1, int(concurrency))
    output: asyncio.Queue[Any] = asyncio.Queue(maxsize=concurrency)
    sentinel = object()
    sem = asyncio.Semaphore(concurrency)

    async def _run(item: Any) -> None:
        try:
            res = await worker(item)
            await output.put(res)
        except BaseException as exc:
            await output.put(exc)
        finally:
            sem.release()

    async def _producer() -> None:
        try:
            for item in items:
                # Block here until a worker slot opens; this is the key
                # back-pressure point — no new Task is created until one
                # finishes and releases the semaphore.
                await sem.acquire()
                asyncio.ensure_future(_run(item))
        except BaseException as exc:
            await output.put(exc)
        finally:
            await output.put(sentinel)

    producer = asyncio.create_task(_producer())
    try:
        while True:
            res = await output.get()
            if isinstance(res, BaseException):
                raise res
            if res is sentinel:
                break
            yield res
    finally:
        if not producer.done():
            producer.cancel()
        await producer


@dataclass
class Indexer:
    cfg: EngramConfig
    embedding_service: EmbeddingService
    project_path_context: PathContext
    index_path_context: PathContext
    faiss_available: bool = field(default_factory=is_faiss_available)

    async def _index_project_fts_only(
        self,
        *,
        directory: str,
        project_name: str,
        project_type: str,
    ) -> str:
        project_id = ProjectID(f"{_slug(project_name)}_{time.time_ns()}_{uuid.uuid4().hex[:8]}")
        rwlock = await get_project_rwlock(str(project_id))
        async with rwlock.write_lock():
            model_name = _pick_model_name(project_type, self.cfg)
            device = "cuda" if _cuda_available() else "cpu"
            project_initialized = False
            file_count = 0
            parse_failures = 0
            parse_attempts = 0
            total_chunks = 0

            def iter_indexable_files() -> Iterable[Any]:
                nonlocal file_count
                for p in iter_files(self.project_path_context, directory, self.cfg.ignore_patterns):
                    ext = p.suffix.lower()
                    if ext in SUPPORTED_TEXT_EXTS or ext in SUPPORTED_DOC_EXTS:
                        file_count += 1
                        yield p

            pdf_sem = asyncio.Semaphore(1)

            async def parse_one(path: Any) -> Optional[ParsedFile]:
                rel = str(path.relative_to(directory))
                try:
                    def _do_parse():
                        return hash_and_parse_file(
                            path_context=self.project_path_context,
                            path=path,
                            root=directory,
                            project_type=project_type,
                            chunk_size_tokens=self.cfg.chunk_size_tokens,
                            overlap_tokens=self.cfg.overlap_tokens,
                            max_file_size_mb=self.cfg.max_file_size_mb,
                        )

                    if str(path.suffix).lower() == ".pdf":
                        async with pdf_sem:
                            chunks, content_hash = await _run_in_thread_with_timeout(_do_parse)
                    else:
                        chunks, content_hash = await _run_in_thread_with_timeout(_do_parse)
                except Exception:
                    logging.warning("Failed to parse %s", rel, exc_info=True)
                    return None
                stat = self.project_path_context.stat(path)
                return ParsedFile(
                    file_path=str(path.relative_to(directory)),
                    mtime_ns=stat.st_mtime_ns,
                    size_bytes=stat.st_size,
                    content_hash=content_hash,
                    chunks=chunks,
                )

            try:
                async for parsed in _bounded_gather(
                    iter_indexable_files(),
                    parse_one,
                    _capped_concurrency(),
                ):
                    parse_attempts += 1
                    if parsed is None:
                        parse_failures += 1
                        if parse_failures >= max(5, int(math.ceil(parse_attempts * 0.1))):
                            raise RuntimeError("Too many files failed to parse; aborting indexing.")
                        continue

                    if not project_initialized:
                        await dbmod.upsert_project(
                            self.cfg.db_path,
                            project_id=str(project_id),
                            project_name=project_name,
                            project_type=project_type,
                            directory=directory,
                            directory_realpath=os.path.realpath(directory),
                            embedding_dim=0,
                            metadata={
                                "chunk_size_tokens": self.cfg.chunk_size_tokens,
                                "overlap_tokens": self.cfg.overlap_tokens,
                                "model_name": model_name,
                                "device": device,
                                "shard_count": 1,
                                "shard_size": int(self.cfg.shard_size),
                            },
                            faiss_index_uuid=None,
                            index_dirty=0,
                        )
                        project_initialized = True

                    file_row = dbmod.FileMetadataRow(
                        file_path=parsed.file_path,
                        mtime_ns=parsed.mtime_ns,
                        size_bytes=parsed.size_bytes,
                        content_hash=parsed.content_hash,
                        chunk_ids=[c.chunk_id for c in parsed.chunks],
                    )
                    if parsed.chunks:
                        internal_ids = await dbmod.reserve_internal_ids(
                            self.cfg.db_path, str(project_id), len(parsed.chunks)
                        )
                        rows = [
                            (c.chunk_id, int(iid), c.content, int(c.token_count), c.metadata)
                            for c, iid in zip(parsed.chunks, internal_ids)
                        ]
                        await dbmod.upsert_chunks(self.cfg.db_path, project_id=str(project_id), rows=rows)
                        total_chunks += len(parsed.chunks)
                    await dbmod.upsert_file_metadata(self.cfg.db_path, str(project_id), [file_row])

                if file_count == 0 or total_chunks == 0:
                    raise ValueError("No indexable content found.")
                if not project_initialized:
                    raise ValueError("No indexable content found.")

                await dbmod.refresh_project_stats(self.cfg.db_path, str(project_id))
                await dbmod.set_file_count(self.cfg.db_path, str(project_id), file_count)
                await invalidate_search_cache_project(str(project_id))
                return str(project_id)
            except BaseException:
                if project_initialized:
                    await dbmod.delete_project(self.cfg.db_path, str(project_id))
                raise

    async def _update_project_fts_only(
        self,
        *,
        project: ProjectID,
        proj: Dict[str, Any],
        metadata: Dict[str, Any],
        directory: str,
        current_realpath: str,
        current_files: Dict[str, Tuple[int, int, str, Any]],
        existing_meta: Dict[str, dbmod.FileMetadataRow],
    ) -> Dict[str, int]:
        project_type = proj["project_type"]
        model_name = metadata.get("model_name") or _pick_model_name(project_type, self.cfg)
        device = metadata.get("device") or ("cuda" if _cuda_available() else "cpu")

        file_count = len(current_files)
        deleted_files = set(existing_meta.keys()) - set(current_files.keys())
        changed_files = {
            rel
            for rel, (mtime_ns, size_bytes, content_hash, _) in current_files.items()
            if rel not in existing_meta
            or existing_meta[rel].mtime_ns != mtime_ns
            or existing_meta[rel].size_bytes != size_bytes
            or existing_meta[rel].content_hash != content_hash
        }

        to_delete_chunk_ids: List[str] = []
        paths_for_deletes = sorted({*deleted_files, *changed_files})
        if paths_for_deletes:
            chunk_map = await dbmod.fetch_file_chunk_ids(self.cfg.db_path, str(project), paths_for_deletes)
            for path in paths_for_deletes:
                to_delete_chunk_ids.extend(chunk_map.get(path, []))

        if not to_delete_chunk_ids and not changed_files:
            await dbmod.set_file_count(self.cfg.db_path, str(project), file_count)
            if not proj.get("directory_realpath"):
                await dbmod.upsert_project(
                    self.cfg.db_path,
                    project_id=str(project),
                    project_name=proj["project_name"],
                    project_type=project_type,
                    directory=directory,
                    directory_realpath=current_realpath,
                    embedding_dim=int(proj.get("embedding_dim") or 0),
                    metadata=metadata,
                    faiss_index_uuid=proj.get("faiss_index_uuid"),
                    index_dirty=proj.get("index_dirty") or 0,
                    deleting=proj.get("deleting") or 0,
                )
            return {"added": 0, "deleted": 0}

        added_chunk_ids: List[str] = []
        added_count = 0
        updated_file_rows: List[dbmod.FileMetadataRow] = []
        pdf_sem = asyncio.Semaphore(1)
        failed_changed_files: set = set()

        async def parse_one(rel_path: str) -> Optional[ParsedFile]:
            mtime_ns, size_bytes, content_hash, path = current_files[rel_path]
            try:
                def _do_parse():
                    return parse_file_to_chunks(
                        path_context=self.project_path_context,
                        path=path,
                        root=directory,
                        project_type=project_type,
                        chunk_size_tokens=self.cfg.chunk_size_tokens,
                        overlap_tokens=self.cfg.overlap_tokens,
                        max_file_size_mb=self.cfg.max_file_size_mb,
                    )

                if str(path.suffix).lower() == ".pdf":
                    async with pdf_sem:
                        chunks = await _run_in_thread_with_timeout(_do_parse)
                else:
                    chunks = await _run_in_thread_with_timeout(_do_parse)
            except Exception:
                logging.warning("Failed to parse %s", rel_path, exc_info=True)
                failed_changed_files.add(rel_path)
                return None
            return ParsedFile(
                file_path=rel_path,
                mtime_ns=mtime_ns,
                size_bytes=size_bytes,
                content_hash=content_hash,
                chunks=chunks,
            )

        try:
            async for parsed in _bounded_gather(
                sorted(changed_files), parse_one, _capped_concurrency()
            ):
                if parsed is None:
                    continue
                chunk_ids = [c.chunk_id for c in parsed.chunks]
                updated_file_rows.append(
                    dbmod.FileMetadataRow(
                        file_path=parsed.file_path,
                        mtime_ns=parsed.mtime_ns,
                        size_bytes=parsed.size_bytes,
                        content_hash=parsed.content_hash,
                        chunk_ids=chunk_ids,
                    )
                )
                if not parsed.chunks:
                    continue
                internal_ids = await dbmod.reserve_internal_ids(self.cfg.db_path, str(project), len(parsed.chunks))
                rows = [
                    (c.chunk_id, int(iid), c.content, int(c.token_count), c.metadata)
                    for c, iid in zip(parsed.chunks, internal_ids)
                ]
                await dbmod.upsert_chunks(self.cfg.db_path, project_id=str(project), rows=rows)
                added_chunk_ids.extend(chunk_ids)
                added_count += len(parsed.chunks)

            # Only delete old chunks for files that were *successfully*
            # re-parsed (or truly deleted from disk).  Chunks belonging to
            # files that failed re-parsing are preserved so that the previous
            # good index survives transient errors.
            safe_to_delete_paths = sorted(
                (deleted_files | changed_files) - failed_changed_files
            )
            if safe_to_delete_paths:
                safe_chunk_map = await dbmod.fetch_file_chunk_ids(
                    self.cfg.db_path, str(project), safe_to_delete_paths
                )
                safe_delete_ids = [
                    cid for path in safe_to_delete_paths for cid in safe_chunk_map.get(path, [])
                ]
                if safe_delete_ids:
                    await dbmod.delete_chunks(self.cfg.db_path, str(project), safe_delete_ids)
            else:
                safe_delete_ids = []

            if updated_file_rows:
                await dbmod.upsert_file_metadata(self.cfg.db_path, str(project), updated_file_rows)
            if deleted_files:
                await dbmod.delete_file_metadata(self.cfg.db_path, str(project), sorted(deleted_files))

            await dbmod.refresh_project_stats(self.cfg.db_path, str(project))
            await dbmod.set_file_count(self.cfg.db_path, str(project), file_count)
            await dbmod.upsert_project(
                self.cfg.db_path,
                project_id=str(project),
                project_name=proj["project_name"],
                project_type=project_type,
                directory=directory,
                directory_realpath=current_realpath,
                embedding_dim=int(proj.get("embedding_dim") or 0),
                metadata={
                    **metadata,
                    "model_name": model_name,
                    "device": device,
                },
                faiss_index_uuid=proj.get("faiss_index_uuid"),
                index_dirty=0,
                deleting=proj.get("deleting") or 0,
            )
            await invalidate_search_cache_project(str(project))
            return {"added": added_count, "deleted": len(safe_delete_ids)}
        except BaseException:
            if added_chunk_ids:
                await dbmod.delete_chunks(self.cfg.db_path, str(project), added_chunk_ids)
            raise

    async def _sync_project_artifacts(self, *, project_id: ProjectID, shard_count: int) -> None:
        artifacts = _index_artifacts(self.cfg.index_dir, project_id, shard_count)
        await dbmod.replace_project_artifacts(
            self.cfg.db_path,
            project_id=str(project_id),
            artifacts=artifacts,
        )

    async def _embed_chunks_cached(
        self,
        chunks: List[Any],
        *,
        model_name: str,
        device: str,
    ) -> np.ndarray:
        if not chunks:
            return np.zeros((0, 0), dtype=np.float32)

        texts = [c.content for c in chunks]
        hashes = [_content_hash(t) for t in texts]
        cached = await dbmod.fetch_embedding_cache(
            self.cfg.db_path,
            model_name=model_name,
            content_hashes=hashes,
        )

        vectors: List[Optional[np.ndarray]] = [None] * len(texts)
        to_embed_texts: List[str] = []
        to_embed_indices: List[int] = []

        for i, h in enumerate(hashes):
            if h in cached:
                vectors[i] = np.frombuffer(cached[h], dtype=np.float32)
            else:
                to_embed_texts.append(texts[i])
                to_embed_indices.append(i)

        new_cache_rows: List[Tuple[str, bytes]] = []
        batch = max(1, int(self.cfg.embedding_batch_size))
        idx = 0
        for i in range(0, len(to_embed_texts), batch):
            batch_texts = to_embed_texts[i:i + batch]
            vecs = await self.embedding_service.embed_texts(
                batch_texts,
                model_name=model_name,
                device=device,
            )
            for vec in vecs:
                target_idx = to_embed_indices[idx]
                vec_np = np.asarray(vec, dtype=np.float32)
                vectors[target_idx] = vec_np
                new_cache_rows.append((hashes[target_idx], vec_np.tobytes()))
                idx += 1

        if new_cache_rows:
            await dbmod.upsert_embedding_cache(
                self.cfg.db_path,
                model_name=model_name,
                rows=new_cache_rows,
                ttl_s=int(self.cfg.embedding_cache_ttl_s),
                max_rows=int(self.cfg.embedding_cache_max_rows),
            )

        if any(v is None for v in vectors):
            raise RuntimeError("Embedding cache did not resolve all vectors.")
        return np.vstack([v for v in vectors]).astype(np.float32)

    async def _resolve_vectors(
        self,
        batch: List[Dict[str, Any]],
        *,
        model_name: str,
        device: str,
    ) -> np.ndarray:
        """Return vectors for *batch*, using DB-persisted vectors where available.

        Chunks indexed before the vector-persistence migration have a NULL
        ``vector`` column; those are re-embedded on the fly.  All other chunks
        skip the (potentially expensive / paid-API) embedding call entirely.
        """
        persisted: Dict[int, np.ndarray] = {}
        needs_embed: List[Dict[str, Any]] = []
        for r in batch:
            vec_blob = r.get("vector")
            if vec_blob:
                # .copy() is required: np.frombuffer returns a read-only view
                # of the bytes object, which FAISS would reject.
                persisted[r["internal_id"]] = np.frombuffer(vec_blob, dtype=np.float32).copy()
            else:
                needs_embed.append(r)

        if needs_embed:
            newly_embedded = await self._embed_chunks_cached(
                [r["chunk"] for r in needs_embed],
                model_name=model_name,
                device=device,
            )
            for i, r in enumerate(needs_embed):
                persisted[r["internal_id"]] = newly_embedded[i]

        return np.vstack([persisted[r["internal_id"]] for r in batch]).astype(np.float32)

    async def _build_faiss_index(self, vectors: np.ndarray, *, total_vectors: Optional[int] = None) -> Any:
        if not self.faiss_available:
            raise RuntimeError("Vector search unavailable (faiss not installed).")
        import faiss

        d = int(vectors.shape[1])
        n_vectors = int(total_vectors or vectors.shape[0])

        if n_vectors < 10_000:
            index = faiss.IndexFlatIP(d)
            index = _ensure_id_map(index)
            return index

        nlist = max(1, int(self.cfg.faiss_nlist))
        nlist = min(nlist, max(1, n_vectors // 10))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(
            quantizer,
            d,
            nlist,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.nprobe = max(1, int(self.cfg.faiss_nprobe))
        index = _ensure_id_map(index)

        def _train():
            faiss.normalize_L2(vectors)
            if not index.is_trained:
                index.train(vectors)
            _enable_direct_map(index)
            return index

        return await asyncio.to_thread(_train)

    async def _add_vectors(self, index: Any, vectors: np.ndarray, ids: np.ndarray) -> None:
        if not self.faiss_available:
            raise RuntimeError("Vector search unavailable (faiss not installed).")
        import faiss

        def _add() -> None:
            faiss.normalize_L2(vectors)
            index.add_with_ids(vectors, ids)
            _enable_direct_map(index)

        await asyncio.to_thread(_add)

    async def _persist_index_batch(
        self,
        *,
        project_id: str,
        index: Any,
        vectors: np.ndarray,
        ids: np.ndarray,
        rows: List[Tuple[str, int, str, int, Dict[str, Any]]],
        file_rows: List[dbmod.FileMetadataRow],
        index_path: str,
        rwlock: Any,
        index_uuid: Optional[str] = None,
        persist_index: bool = True,
        mark_dirty: bool = True,
        clear_dirty: bool = True,
    ) -> None:
        chunk_ids = [row[0] for row in rows]
        file_paths = [row.file_path for row in file_rows]

        # --- Fix #2: WAL / dirty-flag protocol ---
        # Mark the project dirty *before* committing chunks to the DB.
        # If the process is killed after the DB commit but before the
        # FAISS file is written, the flag survives in SQLite and triggers
        # a full index rebuild on the next ``update_project`` call.
        if mark_dirty:
            await dbmod.mark_project_dirty(self.cfg.db_path, project_id)

        await dbmod.upsert_file_metadata(self.cfg.db_path, project_id, file_rows)
        await dbmod.upsert_chunks(self.cfg.db_path, project_id=project_id, rows=rows)

        # Persist vectors alongside chunks so that dirty-flag recovery can
        # skip re-embedding (which may hit a paid API or take minutes).
        vec_rows = [
            (vectors[i].tobytes(), project_id, int(ids[i]))
            for i in range(len(ids))
        ]
        await dbmod.upsert_chunk_vectors(self.cfg.db_path, vec_rows)

        vectors_added = False
        try:
            await self._add_vectors(index, vectors, ids)
            vectors_added = True
            if persist_index:
                async with rwlock.write_lock():
                    await invalidate_faiss_cache(index_path)
                    await asyncio.to_thread(_save_faiss, index, self.index_path_context, index_path)
                    # --- Fix #12: write UUID companion file ---
                    if index_uuid:
                        await asyncio.to_thread(
                            _write_index_uuid, self.index_path_context, index_path, index_uuid
                        )
                # FAISS + UUID both on disk — safe to clear dirty flag.
                if clear_dirty:
                    await dbmod.clear_project_dirty(self.cfg.db_path, project_id)
        except Exception:
            logging.error("Failed to persist index batch for project %s", project_id, exc_info=True)
            if vectors_added:
                try:
                    index.remove_ids(ids)
                except Exception:
                    logging.warning("Failed to rollback FAISS vectors for project %s", project_id, exc_info=True)
            await dbmod.delete_chunks(self.cfg.db_path, project_id, chunk_ids)
            if file_paths:
                await dbmod.delete_file_metadata(self.cfg.db_path, project_id, file_paths)
            raise

    async def index_project(
        self,
        *,
        directory: str,
        project_name: str,
        project_type: str,
        dedupe_by_directory: bool = True,
    ) -> str:
        await dbmod.init_db(self.cfg.db_path)
        directory = str(self.project_path_context.resolve_path(directory))
        project_name = validate_project_field(project_name, field_name="project_name")
        project_type = validate_project_field(project_type, field_name="project_type")
        directory_realpath = os.path.realpath(directory)
        if dedupe_by_directory:
            existing_by_dir = await dbmod.get_project_by_directory_realpath(
                self.cfg.db_path,
                directory_realpath=directory_realpath,
            )
            if existing_by_dir:
                if existing_by_dir.get("project_name") != project_name:
                    raise ValueError(
                        f"Directory already indexed as project '{existing_by_dir.get('project_name')}'. "
                        "Use that project_id or disable dedupe_by_directory."
                    )
                await self.update_project(project_id=str(existing_by_dir["project_id"]))
                return str(existing_by_dir["project_id"])
        existing = await dbmod.get_project_by_name(self.cfg.db_path, project_name=project_name)
        if existing:
            stored_realpath = existing.get("directory_realpath") or os.path.realpath(
                existing.get("directory") or ""
            )
            if stored_realpath and stored_realpath != directory_realpath:
                raise ValueError(
                    f"Project name '{project_name}' already exists for a different directory."
                )
            await self.update_project(project_id=str(existing["project_id"]))
            return str(existing["project_id"])

        if not self.faiss_available:
            logging.warning("FAISS not installed; indexing in FTS-only mode.")
            return await self._index_project_fts_only(
                directory=directory,
                project_name=project_name,
                project_type=project_type,
            )

        project_id = ProjectID(f"{_slug(project_name)}_{time.time_ns()}_{uuid.uuid4().hex[:8]}")
        rwlock = await get_project_rwlock(str(project_id))

        try:
            # Write lock is NOT held for the entire indexing loop.  Parsing
            # and embedding are CPU/IO-bound and can run while readers
            # (search) hold a read-lock.  The write-lock is acquired only
            # for the short critical section that mutates the FAISS index on
            # disk (inside _persist_index_batch and the final save below).
            model_name = _pick_model_name(project_type, self.cfg)
            device = "cuda" if _cuda_available() else "cpu"
            project_initialized = False
            index_path: Optional[str] = None
            index_current: Optional[str] = None
            file_count = 0

            def iter_indexable_files() -> Iterable[Any]:
                nonlocal file_count
                for p in iter_files(self.project_path_context, directory, self.cfg.ignore_patterns):
                    ext = p.suffix.lower()
                    if ext in SUPPORTED_TEXT_EXTS or ext in SUPPORTED_DOC_EXTS:
                        file_count += 1
                        yield p

            pdf_sem = asyncio.Semaphore(1)

            async def parse_one(path: Any) -> Optional[ParsedFile]:
                rel = str(path.relative_to(directory))
                try:
                    def _do_parse():
                        return hash_and_parse_file(
                            path_context=self.project_path_context,
                            path=path,
                            root=directory,
                            project_type=project_type,
                            chunk_size_tokens=self.cfg.chunk_size_tokens,
                            overlap_tokens=self.cfg.overlap_tokens,
                            max_file_size_mb=self.cfg.max_file_size_mb,
                        )

                    if str(path.suffix).lower() == ".pdf":
                        async with pdf_sem:
                            chunks, content_hash = await _run_in_thread_with_timeout(_do_parse)
                    else:
                        chunks, content_hash = await _run_in_thread_with_timeout(_do_parse)
                except Exception:
                    logging.warning("Failed to parse %s", rel, exc_info=True)
                    return None
                stat = self.project_path_context.stat(path)
                return ParsedFile(
                    file_path=str(path.relative_to(directory)),
                    mtime_ns=stat.st_mtime_ns,
                    size_bytes=stat.st_size,
                    content_hash=content_hash,
                    chunks=chunks,
                )

            embedding_dim: Optional[int] = None
            index: Optional[Any] = None
            pending_vectors: List[np.ndarray] = []
            pending_ids: List[np.ndarray] = []
            pending_rows: List[List[Tuple[str, int, str, int, Dict[str, Any]]]] = []
            pending_file_rows: List[List[dbmod.FileMetadataRow]] = []
            training_vectors: List[np.ndarray] = []
            total_chunks = 0
            train_target = max(1000, int(self.cfg.faiss_nlist) * 20)
            index_current, index_new = _index_paths(self.cfg.index_dir, project_id)
            index_path = index_new
            parse_failures = 0
            parse_attempts = 0
            dirty_marked = False
            index_uuid = uuid.uuid4().hex

            async for parsed in _bounded_gather(iter_indexable_files(), parse_one, _capped_concurrency()):
                parse_attempts += 1
                if parsed is None:
                    parse_failures += 1
                    if parse_failures >= max(5, int(math.ceil(parse_attempts * 0.1))):
                        raise RuntimeError("Too many files failed to parse; aborting indexing.")
                    continue

                all_chunk_ids = [c.chunk_id for c in parsed.chunks]
                file_row = dbmod.FileMetadataRow(
                    file_path=parsed.file_path,
                    mtime_ns=parsed.mtime_ns,
                    size_bytes=parsed.size_bytes,
                    content_hash=parsed.content_hash,
                    chunk_ids=all_chunk_ids,
                )
                if not parsed.chunks:
                    await dbmod.upsert_file_metadata(self.cfg.db_path, str(project_id), [file_row])
                    continue

                internal_ids = await dbmod.reserve_internal_ids(
                    self.cfg.db_path, str(project_id), len(parsed.chunks)
                )

                for batch_start in range(0, len(parsed.chunks), _PERSIST_BATCH_SIZE):
                    batch_end = min(batch_start + _PERSIST_BATCH_SIZE, len(parsed.chunks))
                    batch_chunks = parsed.chunks[batch_start:batch_end]
                    batch_iids = internal_ids[batch_start:batch_end]
                    batch_ids_arr = np.asarray(batch_iids, dtype=np.int64)

                    vectors = await self._embed_chunks_cached(batch_chunks, model_name=model_name, device=device)
                    if vectors.size == 0:
                        continue
                    if embedding_dim is None:
                        embedding_dim = int(vectors.shape[1])

                    if not project_initialized:
                        await dbmod.upsert_project(
                            self.cfg.db_path,
                            project_id=str(project_id),
                            project_name=project_name,
                            project_type=project_type,
                            directory=directory,
                            directory_realpath=directory_realpath,
                            embedding_dim=embedding_dim,
                            metadata={
                                "chunk_size_tokens": self.cfg.chunk_size_tokens,
                                "overlap_tokens": self.cfg.overlap_tokens,
                                "model_name": model_name,
                                "device": device,
                                "shard_count": 1,
                                "shard_size": int(self.cfg.shard_size),
                            },
                            faiss_index_uuid=index_uuid,
                        )
                        project_initialized = True

                    batch_rows = [
                        (c.chunk_id, int(iid), c.content, int(c.token_count), c.metadata)
                        for c, iid in zip(batch_chunks, batch_iids)
                    ]

                    if index is None:
                        training_vectors.append(vectors)
                        pending_vectors.append(vectors)
                        pending_ids.append(batch_ids_arr)
                        pending_rows.append(batch_rows)
                        pending_file_rows.append([])
                        if sum(v.shape[0] for v in training_vectors) >= train_target:
                            train_vecs = np.vstack(training_vectors)
                            pending_total = sum(v.shape[0] for v in pending_vectors)
                            index = await self._build_faiss_index(
                                train_vecs,
                                total_vectors=sum(v.shape[0] for v in training_vectors) + pending_total,
                            )
                            for vec_batch, id_batch, row_batch, file_row_batch in zip(
                                pending_vectors, pending_ids, pending_rows, pending_file_rows
                            ):
                                await self._persist_index_batch(
                                    project_id=str(project_id),
                                    index=index,
                                    vectors=vec_batch,
                                    ids=id_batch,
                                    rows=row_batch,
                                    file_rows=file_row_batch,
                                    index_path=index_path,
                                    rwlock=rwlock,
                                    index_uuid=index_uuid,
                                    persist_index=False,
                                    mark_dirty=not dirty_marked,
                                    clear_dirty=False,
                                )
                                dirty_marked = True
                            training_vectors.clear()
                            pending_vectors.clear()
                            pending_ids.clear()
                            pending_rows.clear()
                            pending_file_rows.clear()
                    else:
                        await self._persist_index_batch(
                            project_id=str(project_id),
                            index=index,
                            vectors=vectors,
                            ids=batch_ids_arr,
                            rows=batch_rows,
                            file_rows=[],
                            index_path=index_path,
                            rwlock=rwlock,
                            index_uuid=index_uuid,
                            persist_index=False,
                            mark_dirty=not dirty_marked,
                            clear_dirty=False,
                        )
                        dirty_marked = True

                    total_chunks += len(batch_chunks)

                # All batches for this file persisted; now write its metadata.
                await dbmod.upsert_file_metadata(self.cfg.db_path, str(project_id), [file_row])

            if file_count == 0:
                raise ValueError("No indexable content found.")

            if total_chunks == 0:
                raise ValueError("No indexable content found.")

            if embedding_dim is None:
                raise ValueError("Embedding generation produced no vectors.")

            if not project_initialized:
                await dbmod.upsert_project(
                    self.cfg.db_path,
                    project_id=str(project_id),
                    project_name=project_name,
                    project_type=project_type,
                    directory=directory,
                    directory_realpath=directory_realpath,
                    embedding_dim=embedding_dim,
                    metadata={
                        "chunk_size_tokens": self.cfg.chunk_size_tokens,
                        "overlap_tokens": self.cfg.overlap_tokens,
                        "model_name": model_name,
                        "device": device,
                        "shard_count": 1,
                        "shard_size": int(self.cfg.shard_size),
                    },
                    faiss_index_uuid=index_uuid,
                )

            if index is None:
                train_vecs = np.vstack(training_vectors) if training_vectors else np.vstack(pending_vectors)
                index = await self._build_faiss_index(train_vecs, total_vectors=total_chunks)
                for vec_batch, id_batch, row_batch, file_row_batch in zip(
                    pending_vectors, pending_ids, pending_rows, pending_file_rows
                ):
                    await self._persist_index_batch(
                        project_id=str(project_id),
                        index=index,
                        vectors=vec_batch,
                        ids=id_batch,
                        rows=row_batch,
                        file_rows=file_row_batch,
                        index_path=index_path,
                        rwlock=rwlock,
                        index_uuid=index_uuid,
                        persist_index=False,
                        mark_dirty=not dirty_marked,
                        clear_dirty=False,
                    )
                    dirty_marked = True
                pending_rows.clear()
                pending_file_rows.clear()

            if index_path and index is not None:
                async with rwlock.write_lock():
                    await invalidate_faiss_cache(index_path)
                    await asyncio.to_thread(_save_faiss, index, self.index_path_context, index_path)
                    if index_uuid:
                        await asyncio.to_thread(
                            _write_index_uuid, self.index_path_context, index_path, index_uuid
                        )
                if dirty_marked:
                    await dbmod.clear_project_dirty(self.cfg.db_path, str(project_id))

            await dbmod.refresh_project_stats(self.cfg.db_path, str(project_id))
            await dbmod.set_file_count(self.cfg.db_path, str(project_id), file_count)

            if index_path and index_current and self.index_path_context.exists(index_path):
                _replace_with_retries(self.index_path_context, index_path, index_current)
                await invalidate_faiss_cache(index_current)
                if index_uuid:
                    await asyncio.to_thread(
                        _write_index_uuid, self.index_path_context, index_current, index_uuid
                    )

            await self._maybe_shard_project(project_id=str(project_id))
            proj = await dbmod.get_project(self.cfg.db_path, str(project_id))
            shard_count = int((proj or {}).get("metadata", {}).get("shard_count") or 1)
            await self._sync_project_artifacts(project_id=project_id, shard_count=shard_count)
            await invalidate_search_cache_project(str(project_id))
            return str(project_id)
        except BaseException:
            if project_initialized:
                if index_path and self.index_path_context.exists(index_path):
                    self.index_path_context.unlink(index_path)
                if index_current and self.index_path_context.exists(index_current):
                    self.index_path_context.unlink(index_current)
                await dbmod.delete_project(self.cfg.db_path, str(project_id))
            raise

    async def update_project(self, *, project_id: str) -> Dict[str, int]:
        project = ProjectID(project_id)
        await dbmod.init_db(self.cfg.db_path)
        proj = await dbmod.get_project(self.cfg.db_path, str(project))
        if not proj:
            raise ValueError(f"Project not found: {project_id}")

        directory = proj.get("directory")
        if not directory:
            raise ValueError("Project has no stored directory; cannot update.")
        directory = str(self.project_path_context.resolve_path(directory))
        metadata = proj.get("metadata") or {}
        stored_realpath = proj.get("directory_realpath")
        current_realpath = os.path.realpath(directory)
        if stored_realpath and stored_realpath != current_realpath:
            raise ValueError(
                f"Project directory realpath changed from '{stored_realpath}' to '{current_realpath}'. "
                "Refusing to update to prevent symlink traversal."
            )

        project_type = proj["project_type"]
        stored_model = metadata.get("model_name")
        current_model = _pick_model_name(project_type, self.cfg)

        # Enforce model consistency to prevent vector space pollution
        if stored_model and stored_model != current_model:
            raise ValueError(
                f"Model mismatch: project was indexed with '{stored_model}' but config specifies '{current_model}'. "
                f"Delete and re-index the project, or update config to use the original model."
            )

        model_name = stored_model or current_model
        device = metadata.get("device") or ("cuda" if _cuda_available() else "cpu")

        # --- Fix #2: dirty-flag recovery ---
        # If a previous indexing run crashed after writing chunks to the
        # DB but before persisting the FAISS index, the dirty flag is set.
        # Rebuild the entire FAISS index from DB state before proceeding.
        if proj.get("index_dirty") and self.faiss_available:
            logging.warning(
                "Project %s has index_dirty set (incomplete previous write). "
                "Rebuilding FAISS index from database.",
                project_id,
            )
            await self._rebuild_index_from_db(project_id=str(project), metadata=metadata)
            await invalidate_search_cache_project(str(project))
            # Re-fetch metadata in case UUID was regenerated
            proj = await dbmod.get_project(self.cfg.db_path, str(project))
            metadata = proj.get("metadata") or {}
        elif proj.get("index_dirty") and not self.faiss_available:
            logging.warning(
                "Project %s has index_dirty set but FAISS is unavailable; "
                "continuing in FTS-only mode.",
                project_id,
            )

        lock = await get_project_lock(str(project))
        async with lock:
            existing_meta = await dbmod.fetch_file_metadata(self.cfg.db_path, str(project))
            rwlock = await get_project_rwlock(str(project))

            current_files: Dict[str, Tuple[int, int, str, Any]] = {}
            max_bytes = int(self.cfg.max_file_size_mb) * 1024 * 1024
            for p in iter_files(self.project_path_context, directory, self.cfg.ignore_patterns):
                ext = p.suffix.lower()
                if ext not in SUPPORTED_TEXT_EXTS and ext not in SUPPORTED_DOC_EXTS:
                    continue
                rel = str(p.relative_to(directory))
                stat = self.project_path_context.stat(p)
                existing = existing_meta.get(rel)
                # mtime + size both unchanged → file is unchanged; reuse
                # the stored hash and skip the full-file read entirely.
                if existing and existing.mtime_ns == stat.st_mtime_ns and existing.size_bytes == stat.st_size:
                    current_files[rel] = (stat.st_mtime_ns, stat.st_size, existing.content_hash, p)
                    continue
                # File is new or modified; hash it now.
                try:
                    content_hash = hash_file(self.project_path_context, p, Path(directory), max_bytes)
                except Exception:
                    logging.warning("Failed to hash %s", rel, exc_info=True)
                    continue
                current_files[rel] = (stat.st_mtime_ns, stat.st_size, content_hash, p)

            if not self.faiss_available:
                return await self._update_project_fts_only(
                    project=project,
                    proj=proj,
                    metadata=metadata,
                    directory=directory,
                    current_realpath=current_realpath,
                    current_files=current_files,
                    existing_meta=existing_meta,
                )

            file_count = len(current_files)
            deleted_files = set(existing_meta.keys()) - set(current_files.keys())
            changed_files = {
                rel
                for rel, (mtime_ns, size_bytes, content_hash, _) in current_files.items()
                if rel not in existing_meta
                or existing_meta[rel].mtime_ns != mtime_ns
                or existing_meta[rel].size_bytes != size_bytes
                or existing_meta[rel].content_hash != content_hash
            }

            to_delete_chunk_ids: List[str] = []
            paths_for_deletes = sorted({*deleted_files, *changed_files})
            if paths_for_deletes:
                chunk_map = await dbmod.fetch_file_chunk_ids(self.cfg.db_path, str(project), paths_for_deletes)
                for path in paths_for_deletes:
                    to_delete_chunk_ids.extend(chunk_map.get(path, []))

            if not to_delete_chunk_ids and not changed_files:
                await dbmod.set_file_count(self.cfg.db_path, str(project), file_count)
                if not stored_realpath:
                    await dbmod.upsert_project(
                        self.cfg.db_path,
                        project_id=str(project),
                        project_name=proj["project_name"],
                        project_type=project_type,
                        directory=directory,
                        directory_realpath=current_realpath,
                        embedding_dim=int(proj.get("embedding_dim") or 0),
                        metadata=metadata,
                        faiss_index_uuid=proj.get("faiss_index_uuid"),
                        index_dirty=proj.get("index_dirty") or 0,
                        deleting=proj.get("deleting") or 0,
                    )
                return {"added": 0, "deleted": 0}

            added_chunk_ids: List[str] = []
            try:
                internal_ids_to_delete = await dbmod.fetch_internal_ids_for_chunk_ids(
                    self.cfg.db_path, str(project), to_delete_chunk_ids
                )

                shard_count = int(metadata.get("shard_count") or 1)
                added_count = 0
                updated_file_rows: List[dbmod.FileMetadataRow] = []
                pdf_sem = asyncio.Semaphore(1)
                failed_changed_files: set = set()

                async def parse_one(rel_path: str) -> Optional[ParsedFile]:
                    mtime_ns, size_bytes, content_hash, path = current_files[rel_path]
                    try:
                        def _do_parse():
                            return parse_file_to_chunks(
                                path_context=self.project_path_context,
                                path=path,
                                root=directory,
                                project_type=project_type,
                                chunk_size_tokens=self.cfg.chunk_size_tokens,
                                overlap_tokens=self.cfg.overlap_tokens,
                                max_file_size_mb=self.cfg.max_file_size_mb,
                            )

                        if str(path.suffix).lower() == ".pdf":
                            async with pdf_sem:
                                chunks = await _run_in_thread_with_timeout(_do_parse)
                        else:
                            chunks = await _run_in_thread_with_timeout(_do_parse)
                    except Exception:
                        logging.warning("Failed to parse %s", rel_path, exc_info=True)
                        failed_changed_files.add(rel_path)
                        return None
                    return ParsedFile(
                        file_path=rel_path,
                        mtime_ns=mtime_ns,
                        size_bytes=size_bytes,
                        content_hash=content_hash,
                        chunks=chunks,
                    )

                # --- Phase 1: parse + embed outside the write lock so that
                # concurrent read-lock holders (search) are not blocked. ---
                pending_parsed: List[ParsedFile] = []
                async for parsed in _bounded_gather(
                    sorted(changed_files), parse_one, _capped_concurrency()
                ):
                    if parsed is None:
                        continue
                    chunk_ids = [c.chunk_id for c in parsed.chunks]
                    updated_file_rows.append(
                        dbmod.FileMetadataRow(
                            file_path=parsed.file_path,
                            mtime_ns=parsed.mtime_ns,
                            size_bytes=parsed.size_bytes,
                            content_hash=parsed.content_hash,
                            chunk_ids=chunk_ids,
                        )
                    )
                    if parsed.chunks:
                        pending_parsed.append(parsed)

                # Pre-embed all pending chunks while still outside the lock
                pending_embed: List[Tuple[ParsedFile, np.ndarray, List[int]]] = []
                for parsed in pending_parsed:
                    vectors = await self._embed_chunks_cached(
                        parsed.chunks, model_name=model_name, device=device,
                    )
                    internal_ids = await dbmod.reserve_internal_ids(
                        self.cfg.db_path, str(project), len(parsed.chunks)
                    )
                    pending_embed.append((parsed, vectors, internal_ids))

                # --- Phase 2: mutate FAISS indexes under write lock (short) ---
                if shard_count > 1:
                    shard_paths = [
                        _index_paths(self.cfg.index_dir, project, shard_id)
                        for shard_id in range(shard_count)
                    ]
                    shard_indexes: List[Any] = []
                    for current_path, _ in shard_paths:
                        if self.index_path_context.exists(current_path):
                            shard_indexes.append(
                                await asyncio.to_thread(
                                    _load_faiss, current_path, path_context=self.index_path_context
                                )
                            )
                        else:
                            shard_indexes.append(None)

                    async with rwlock.write_lock():
                        if internal_ids_to_delete:
                            for shard_id, index in enumerate(shard_indexes):
                                if index is None:
                                    continue
                                ids_for_shard = [
                                    iid for iid in internal_ids_to_delete if iid % shard_count == shard_id
                                ]
                                if ids_for_shard:
                                    index.remove_ids(np.asarray(ids_for_shard, dtype=np.int64))

                        for parsed, vectors, internal_ids in pending_embed:
                            rows = [
                                (c.chunk_id, int(iid), c.content, int(c.token_count), c.metadata)
                                for c, iid in zip(parsed.chunks, internal_ids)
                            ]
                            await dbmod.upsert_chunks(self.cfg.db_path, project_id=str(project), rows=rows)
                            added_chunk_ids.extend(c.chunk_id for c in parsed.chunks)
                            for shard_id in range(shard_count):
                                shard_mask = [i for i, iid in enumerate(internal_ids) if iid % shard_count == shard_id]
                                if not shard_mask:
                                    continue
                                shard_vectors = vectors[shard_mask]
                                shard_ids = np.asarray([internal_ids[i] for i in shard_mask], dtype=np.int64)
                                if shard_indexes[shard_id] is None:
                                    shard_indexes[shard_id] = await self._build_faiss_index(shard_vectors)
                                await self._add_vectors(shard_indexes[shard_id], shard_vectors, shard_ids)
                            added_count += len(parsed.chunks)

                        index_uuid = uuid.uuid4().hex
                        replacements: List[Tuple[str, str]] = []
                        for shard_id, index in enumerate(shard_indexes):
                            if index is None:
                                continue
                            current_path, new_path = shard_paths[shard_id]
                            await invalidate_faiss_cache(current_path)
                            await asyncio.to_thread(
                                _save_faiss, index, self.index_path_context, new_path,
                            )
                            await asyncio.to_thread(
                                _write_index_uuid, self.index_path_context, new_path, index_uuid
                            )
                            replacements.append((new_path, current_path))
                        if replacements:
                            _swap_index_files(self.index_path_context, replacements)
                            for new_path, current_path in replacements:
                                await asyncio.to_thread(
                                    _write_index_uuid, self.index_path_context, current_path, index_uuid
                                )
                        await dbmod.upsert_project(
                            self.cfg.db_path,
                            project_id=str(project),
                            project_name=proj["project_name"],
                            project_type=project_type,
                            directory=directory,
                            directory_realpath=current_realpath,
                            embedding_dim=int(proj.get("embedding_dim") or 0),
                            metadata=metadata,
                            faiss_index_uuid=index_uuid,
                            index_dirty=proj.get("index_dirty") or 0,
                            deleting=proj.get("deleting") or 0,
                        )
                else:
                    index_current, index_new = _index_paths(self.cfg.index_dir, project)
                    index_path = index_current
                    index = await asyncio.to_thread(
                        _load_faiss, index_path, path_context=self.index_path_context
                    )

                    async with rwlock.write_lock():
                        if internal_ids_to_delete:
                            index.remove_ids(np.asarray(internal_ids_to_delete, dtype=np.int64))

                        for parsed, vectors, internal_ids in pending_embed:
                            rows = [
                                (c.chunk_id, int(iid), c.content, int(c.token_count), c.metadata)
                                for c, iid in zip(parsed.chunks, internal_ids)
                            ]
                            await dbmod.upsert_chunks(self.cfg.db_path, project_id=str(project), rows=rows)
                            added_chunk_ids.extend(c.chunk_id for c in parsed.chunks)
                            await self._add_vectors(index, vectors, np.asarray(internal_ids, dtype=np.int64))
                            added_count += len(parsed.chunks)

                        await invalidate_faiss_cache(index_path)
                        await asyncio.to_thread(_save_faiss, index, self.index_path_context, index_new)
                        index_uuid = uuid.uuid4().hex
                        await asyncio.to_thread(
                            _write_index_uuid, self.index_path_context, index_new, index_uuid
                        )
                        _replace_with_retries(self.index_path_context, index_new, index_current)
                        await asyncio.to_thread(
                            _write_index_uuid, self.index_path_context, index_current, index_uuid
                        )
                        await dbmod.upsert_project(
                            self.cfg.db_path,
                            project_id=str(project),
                            project_name=proj["project_name"],
                            project_type=project_type,
                            directory=directory,
                            directory_realpath=current_realpath,
                            embedding_dim=int(proj.get("embedding_dim") or 0),
                            metadata=metadata,
                            faiss_index_uuid=index_uuid,
                            index_dirty=proj.get("index_dirty") or 0,
                            deleting=proj.get("deleting") or 0,
                        )

                # Only delete old chunks for files that were successfully
                # re-parsed (or removed from disk).  Preserve chunks for
                # files that failed to parse so the previous good data
                # survives transient errors.
                safe_to_delete_paths = sorted(
                    (deleted_files | changed_files) - failed_changed_files
                )
                if safe_to_delete_paths:
                    safe_chunk_map = await dbmod.fetch_file_chunk_ids(
                        self.cfg.db_path, str(project), safe_to_delete_paths
                    )
                    safe_delete_ids = [
                        cid for path in safe_to_delete_paths for cid in safe_chunk_map.get(path, [])
                    ]
                    if safe_delete_ids:
                        await dbmod.delete_chunks(self.cfg.db_path, str(project), safe_delete_ids)
                else:
                    safe_delete_ids = []

                if updated_file_rows:
                    await dbmod.upsert_file_metadata(self.cfg.db_path, str(project), updated_file_rows)
                if deleted_files:
                    await dbmod.delete_file_metadata(self.cfg.db_path, str(project), sorted(deleted_files))

                await dbmod.refresh_project_stats(self.cfg.db_path, str(project))
                await dbmod.set_file_count(self.cfg.db_path, str(project), file_count)
                await self._maybe_shard_project(project_id=str(project))
                proj = await dbmod.get_project(self.cfg.db_path, str(project))
                shard_count = int((proj or {}).get("metadata", {}).get("shard_count") or 1)
                await self._sync_project_artifacts(project_id=project, shard_count=shard_count)
                await invalidate_search_cache_project(str(project))
                return {"added": added_count, "deleted": len(safe_delete_ids)}
            except BaseException:
                if added_chunk_ids:
                    await dbmod.delete_chunks(self.cfg.db_path, str(project), added_chunk_ids)
                raise

    async def _update_sharded_indexes(
        self,
        *,
        project_id: str,
        shard_count: int,
        internal_ids_to_delete: List[int],
        to_add_chunks: List[Any],
        model_name: str,
        device: str,
    ) -> None:
        rwlock = await get_project_rwlock(project_id)
        project = ProjectID(project_id)
        added_chunk_ids: List[str] = []
        shard_paths = [
            _index_paths(self.cfg.index_dir, project, shard_id)
            for shard_id in range(shard_count)
        ]
        shard_indexes: List[Any] = []
        for current_path, _ in shard_paths:
            if self.index_path_context.exists(current_path):
                shard_indexes.append(
                    await asyncio.to_thread(
                        _load_faiss, current_path, path_context=self.index_path_context
                    )
                )
            else:
                shard_indexes.append(None)

        async with rwlock.write_lock():
            try:
                # --- dirty-flag protocol for crash recovery ---
                # Set the flag *before* any DB writes.  If the process is
                # killed after the DB commit but before all shards are on
                # disk, the flag survives in SQLite and triggers a full
                # rebuild via update_project → _rebuild_index_from_db.
                await dbmod.mark_project_dirty(self.cfg.db_path, project_id)

                if internal_ids_to_delete:
                    for shard_id, index in enumerate(shard_indexes):
                        if index is None:
                            continue
                        ids_for_shard = [iid for iid in internal_ids_to_delete if iid % shard_count == shard_id]
                        if ids_for_shard:
                            index.remove_ids(np.asarray(ids_for_shard, dtype=np.int64))

                if to_add_chunks:
                    vectors = await self._embed_chunks_cached(
                        to_add_chunks,
                        model_name=model_name,
                        device=device,
                    )
                    internal_ids = await dbmod.reserve_internal_ids(self.cfg.db_path, project_id, len(to_add_chunks))
                    rows = [
                        (c.chunk_id, int(iid), c.content, int(c.token_count), c.metadata)
                        for c, iid in zip(to_add_chunks, internal_ids)
                    ]
                    await dbmod.upsert_chunks(self.cfg.db_path, project_id=project_id, rows=rows)
                    added_chunk_ids = [c.chunk_id for c in to_add_chunks]

                    for shard_id in range(shard_count):
                        shard_mask = [i for i, iid in enumerate(internal_ids) if iid % shard_count == shard_id]
                        if not shard_mask:
                            continue
                        shard_vectors = vectors[shard_mask]
                        shard_ids = np.asarray([internal_ids[i] for i in shard_mask], dtype=np.int64)
                        if shard_indexes[shard_id] is None:
                            shard_indexes[shard_id] = await self._build_faiss_index(shard_vectors)
                        await self._add_vectors(shard_indexes[shard_id], shard_vectors, shard_ids)

                new_paths: List[Tuple[str, str]] = []
                for shard_id, index in enumerate(shard_indexes):
                    if index is None:
                        continue
                    current_path, new_path = shard_paths[shard_id]
                    await invalidate_faiss_cache(current_path)
                    await asyncio.to_thread(_save_faiss, index, self.index_path_context, new_path)
                    new_paths.append((new_path, current_path))
                if new_paths:
                    _swap_index_files(self.index_path_context, new_paths)

                # All shards safely on disk — clear the dirty flag.
                await dbmod.clear_project_dirty(self.cfg.db_path, project_id)
            except Exception:
                logging.error("Failed to update sharded indexes for project %s", project_id, exc_info=True)
                if added_chunk_ids:
                    await dbmod.delete_chunks(self.cfg.db_path, project_id, added_chunk_ids)
                raise

    async def _rebuild_index_from_db(self, *, project_id: str, metadata: Dict[str, Any]) -> None:
        """Rebuild the FAISS index entirely from chunks in SQLite.

        Called when the ``index_dirty`` flag indicates a previous crash
        between a DB commit and a FAISS disk write.  The flag is the
        only persistent crash-recovery signal; this method restores
        consistency and clears it.
        """
        if not self.faiss_available:
            raise RuntimeError("Vector search unavailable (faiss not installed).")
        model_name = metadata.get("model_name") or _pick_model_name("", self.cfg)
        device = metadata.get("device") or ("cuda" if _cuda_available() else "cpu")
        project = ProjectID(project_id)
        shard_count = int(metadata.get("shard_count") or 1)
        rwlock = await get_project_rwlock(project_id)
        batch_size = max(1, int(self.cfg.embedding_batch_size))
        train_target = max(1000, int(self.cfg.faiss_nlist) * 20)

        proj = await dbmod.get_project(self.cfg.db_path, project_id)
        total_vectors = int((proj or {}).get("chunk_count") or 0)

        index_uuid = uuid.uuid4().hex

        had_chunks = False

        if shard_count > 1:
            shard_indexes: List[Optional[Any]] = [None] * shard_count
            shard_training: List[List[np.ndarray]] = [[] for _ in range(shard_count)]
            shard_training_count: List[int] = [0 for _ in range(shard_count)]
            shard_pending_vecs: List[List[np.ndarray]] = [[] for _ in range(shard_count)]
            shard_pending_ids: List[List[np.ndarray]] = [[] for _ in range(shard_count)]

            async def _flush_pending(shard_id: int) -> None:
                if not shard_pending_vecs[shard_id]:
                    return
                vecs = np.vstack(shard_pending_vecs[shard_id])
                ids = np.concatenate(shard_pending_ids[shard_id])
                await self._add_vectors(shard_indexes[shard_id], vecs, ids)
                shard_pending_vecs[shard_id].clear()
                shard_pending_ids[shard_id].clear()

            def _add_training(shard_id: int, vecs: np.ndarray) -> None:
                if shard_training_count[shard_id] >= train_target:
                    return
                remaining = train_target - shard_training_count[shard_id]
                if remaining <= 0:
                    return
                if vecs.shape[0] > remaining:
                    vecs = vecs[:remaining]
                shard_training[shard_id].append(vecs)
                shard_training_count[shard_id] += int(vecs.shape[0])

            last_internal_id = -1
            while True:
                batch = await dbmod.fetch_chunk_batch(
                    self.cfg.db_path,
                    project_id,
                    last_internal_id=last_internal_id,
                    batch_size=batch_size,
                )
                if not batch:
                    break
                had_chunks = True
                last_internal_id = batch[-1]["internal_id"]
                vectors = await self._resolve_vectors(
                    batch, model_name=model_name, device=device,
                )
                ids = [r["internal_id"] for r in batch]

                shard_indices: List[List[int]] = [[] for _ in range(shard_count)]
                for i, iid in enumerate(ids):
                    shard_indices[iid % shard_count].append(i)

                for shard_id, positions in enumerate(shard_indices):
                    if not positions:
                        continue
                    shard_vecs = vectors[positions]
                    shard_ids = np.asarray([ids[i] for i in positions], dtype=np.int64)
                    if shard_indexes[shard_id] is None:
                        _add_training(shard_id, shard_vecs)
                        shard_pending_vecs[shard_id].append(shard_vecs)
                        shard_pending_ids[shard_id].append(shard_ids)
                        if shard_training_count[shard_id] >= train_target:
                            train_vecs = np.vstack(shard_training[shard_id])
                            estimated_total = max(
                                int(math.ceil(total_vectors / shard_count)),
                                int(train_vecs.shape[0]),
                            )
                            shard_indexes[shard_id] = await self._build_faiss_index(
                                train_vecs,
                                total_vectors=estimated_total,
                            )
                            await _flush_pending(shard_id)
                            shard_training[shard_id].clear()
                            shard_training_count[shard_id] = 0
                    else:
                        shard_pending_vecs[shard_id].append(shard_vecs)
                        shard_pending_ids[shard_id].append(shard_ids)
                        await _flush_pending(shard_id)

            for shard_id in range(shard_count):
                if shard_indexes[shard_id] is None:
                    if not shard_pending_vecs[shard_id]:
                        continue
                    train_vecs = np.vstack(shard_training[shard_id] or shard_pending_vecs[shard_id])
                    estimated_total = max(
                        int(math.ceil(total_vectors / shard_count)),
                        int(train_vecs.shape[0]),
                    )
                    shard_indexes[shard_id] = await self._build_faiss_index(
                        train_vecs,
                        total_vectors=estimated_total,
                    )
                await _flush_pending(shard_id)

            if not had_chunks:
                await dbmod.clear_project_dirty(self.cfg.db_path, project_id)
                return

            async with rwlock.write_lock():
                for shard_id, index in enumerate(shard_indexes):
                    if index is None:
                        continue
                    current_path, new_path = _index_paths(self.cfg.index_dir, project, shard_id)
                    await invalidate_faiss_cache(current_path)
                    await asyncio.to_thread(_save_faiss, index, self.index_path_context, new_path)
                    await asyncio.to_thread(
                        _write_index_uuid, self.index_path_context, new_path, index_uuid
                    )
                    _replace_with_retries(self.index_path_context, new_path, current_path)
                    await asyncio.to_thread(
                        _write_index_uuid, self.index_path_context, current_path, index_uuid
                    )
        else:
            index: Optional[Any] = None
            training_vectors: List[np.ndarray] = []
            pending_vectors: List[np.ndarray] = []
            pending_ids: List[np.ndarray] = []
            last_internal_id = -1

            while True:
                batch = await dbmod.fetch_chunk_batch(
                    self.cfg.db_path,
                    project_id,
                    last_internal_id=last_internal_id,
                    batch_size=batch_size,
                )
                if not batch:
                    break
                had_chunks = True
                last_internal_id = batch[-1]["internal_id"]
                vectors = await self._resolve_vectors(
                    batch, model_name=model_name, device=device,
                )
                ids = np.asarray([r["internal_id"] for r in batch], dtype=np.int64)
                if index is None:
                    training_vectors.append(vectors)
                    pending_vectors.append(vectors)
                    pending_ids.append(ids)
                    if sum(v.shape[0] for v in training_vectors) >= train_target:
                        train_vecs = np.vstack(training_vectors)
                        total = total_vectors or sum(v.shape[0] for v in pending_vectors)
                        index = await self._build_faiss_index(train_vecs, total_vectors=total)
                        for vecs, ids_arr in zip(pending_vectors, pending_ids):
                            await self._add_vectors(index, vecs, ids_arr)
                        training_vectors.clear()
                        pending_vectors.clear()
                        pending_ids.clear()
                else:
                    await self._add_vectors(index, vectors, ids)

            if index is None:
                if not pending_vectors:
                    await dbmod.clear_project_dirty(self.cfg.db_path, project_id)
                    return
                train_vecs = np.vstack(training_vectors or pending_vectors)
                total = total_vectors or sum(v.shape[0] for v in pending_vectors)
                index = await self._build_faiss_index(train_vecs, total_vectors=total)
                for vecs, ids_arr in zip(pending_vectors, pending_ids):
                    await self._add_vectors(index, vecs, ids_arr)

            async with rwlock.write_lock():
                current_path, new_path = _index_paths(self.cfg.index_dir, project)
                await invalidate_faiss_cache(current_path)
                await asyncio.to_thread(_save_faiss, index, self.index_path_context, new_path)
                await asyncio.to_thread(
                    _write_index_uuid, self.index_path_context, new_path, index_uuid
                )
                _replace_with_retries(self.index_path_context, new_path, current_path)
                await asyncio.to_thread(
                    _write_index_uuid, self.index_path_context, current_path, index_uuid
                )

        await dbmod.upsert_project(
            self.cfg.db_path,
            project_id=project_id,
            project_name=(proj or {}).get("project_name", ""),
            project_type=(proj or {}).get("project_type", ""),
            directory=(proj or {}).get("directory", ""),
            directory_realpath=(proj or {}).get("directory_realpath"),
            embedding_dim=int((proj or {}).get("embedding_dim") or 0),
            metadata=metadata,
            faiss_index_uuid=index_uuid,
            index_dirty=0,
            deleting=(proj or {}).get("deleting") or 0,
        )
        shard_count = int(metadata.get("shard_count") or 1)
        await self._sync_project_artifacts(project_id=ProjectID(project_id), shard_count=shard_count)
        logging.info("Rebuilt FAISS index for project %s; dirty flag cleared.", project_id)

    async def _maybe_shard_project(self, *, project_id: str) -> None:
        if not self.faiss_available:
            return
        proj = await dbmod.get_project(self.cfg.db_path, project_id)
        if not proj:
            return
        chunk_count = int(proj.get("chunk_count") or 0)
        shard_count = _compute_shard_count(chunk_count, self.cfg)
        metadata = proj.get("metadata") or {}
        current_shards = int(metadata.get("shard_count") or 1)
        if shard_count <= 1 or shard_count == current_shards:
            return

        model_name = metadata.get("model_name") or _pick_model_name(proj["project_type"], self.cfg)
        device = metadata.get("device") or "cpu"
        rwlock = await get_project_rwlock(project_id)
        project = ProjectID(project_id)
        index_uuid = proj.get("faiss_index_uuid") or uuid.uuid4().hex
        if not proj.get("faiss_index_uuid"):
            await dbmod.set_project_index_uuid(self.cfg.db_path, project_id, index_uuid)
        train_target = max(1000, int(self.cfg.faiss_nlist) * 20)
        shard_indexes: List[Optional[Any]] = [None] * shard_count
        shard_training: List[List[np.ndarray]] = [[] for _ in range(shard_count)]
        shard_training_count: List[int] = [0 for _ in range(shard_count)]
        shard_pending_vecs: List[List[np.ndarray]] = [[] for _ in range(shard_count)]
        shard_pending_ids: List[List[np.ndarray]] = [[] for _ in range(shard_count)]
        batch_size = max(1, int(self.cfg.embedding_batch_size))
        last_internal_id = -1

        async def _flush_pending(shard_id: int) -> None:
            if not shard_pending_vecs[shard_id]:
                return
            vecs = np.vstack(shard_pending_vecs[shard_id])
            ids = np.concatenate(shard_pending_ids[shard_id])
            await self._add_vectors(shard_indexes[shard_id], vecs, ids)
            shard_pending_vecs[shard_id].clear()
            shard_pending_ids[shard_id].clear()

        def _add_training(shard_id: int, vecs: np.ndarray) -> None:
            if shard_training_count[shard_id] >= train_target:
                return
            remaining = train_target - shard_training_count[shard_id]
            if remaining <= 0:
                return
            if vecs.shape[0] > remaining:
                vecs = vecs[:remaining]
            shard_training[shard_id].append(vecs)
            shard_training_count[shard_id] += int(vecs.shape[0])

        while True:
            batch = await dbmod.fetch_chunk_batch(
                self.cfg.db_path,
                project_id,
                last_internal_id=last_internal_id,
                batch_size=batch_size,
            )
            if not batch:
                break
            last_internal_id = batch[-1]["internal_id"]

            vectors = await self._embed_chunks_cached(
                [r["chunk"] for r in batch],
                model_name=model_name,
                device=device,
            )
            ids = [r["internal_id"] for r in batch]

            shard_indices: List[List[int]] = [[] for _ in range(shard_count)]
            for i, iid in enumerate(ids):
                shard_id = iid % shard_count
                shard_indices[shard_id].append(i)

            for shard_id, positions in enumerate(shard_indices):
                if not positions:
                    continue
                shard_vecs = vectors[positions]
                shard_ids = np.asarray([ids[i] for i in positions], dtype=np.int64)
                if shard_indexes[shard_id] is None:
                    _add_training(shard_id, shard_vecs)
                    shard_pending_vecs[shard_id].append(shard_vecs)
                    shard_pending_ids[shard_id].append(shard_ids)
                    if shard_training_count[shard_id] >= train_target:
                        train_vecs = np.vstack(shard_training[shard_id])
                        estimated_total = int(math.ceil(chunk_count / shard_count))
                        shard_indexes[shard_id] = await self._build_faiss_index(
                            train_vecs,
                            total_vectors=estimated_total,
                        )
                        await _flush_pending(shard_id)
                        shard_training[shard_id].clear()
                        shard_training_count[shard_id] = 0
                else:
                    shard_pending_vecs[shard_id].append(shard_vecs)
                    shard_pending_ids[shard_id].append(shard_ids)
                    await _flush_pending(shard_id)

        for shard_id in range(shard_count):
            if shard_indexes[shard_id] is None:
                if not shard_pending_vecs[shard_id]:
                    continue
                train_vecs = np.vstack(shard_training[shard_id] or shard_pending_vecs[shard_id])
                estimated_total = int(math.ceil(chunk_count / shard_count))
                shard_indexes[shard_id] = await self._build_faiss_index(
                    train_vecs,
                    total_vectors=estimated_total,
                )
            await _flush_pending(shard_id)

        async with rwlock.write_lock():
            shard_replacements: List[Tuple[str, str]] = []
            try:
                for shard_id, index in enumerate(shard_indexes):
                    if index is None:
                        continue
                    current_path, new_path = _index_paths(self.cfg.index_dir, project, shard_id)
                    await invalidate_faiss_cache(current_path)
                    await asyncio.to_thread(_save_faiss, index, self.index_path_context, new_path)
                    await asyncio.to_thread(
                        _write_index_uuid, self.index_path_context, new_path, index_uuid
                    )
                    shard_replacements.append((new_path, current_path))
                _swap_index_files(self.index_path_context, shard_replacements)
                for _, current_path in shard_replacements:
                    await asyncio.to_thread(
                        _write_index_uuid, self.index_path_context, current_path, index_uuid
                    )
            except Exception:
                logging.error("Failed to swap shard indexes for project %s", project_id, exc_info=True)
                for new_path, _ in shard_replacements:
                    if self.index_path_context.exists(new_path):
                        self.index_path_context.unlink(new_path)
                raise

        metadata["shard_count"] = shard_count
        metadata["shard_size"] = int(self.cfg.shard_size)
        await dbmod.upsert_project(
            self.cfg.db_path,
            project_id=project_id,
            project_name=proj["project_name"],
            project_type=proj["project_type"],
            directory=proj.get("directory"),
            directory_realpath=proj.get("directory_realpath"),
            embedding_dim=int(proj.get("embedding_dim") or 0),
            metadata=metadata,
            faiss_index_uuid=index_uuid,
            index_dirty=proj.get("index_dirty") or 0,
            deleting=proj.get("deleting") or 0,
        )
        await self._sync_project_artifacts(project_id=project, shard_count=shard_count)

        single_index_path, _ = _index_paths(self.cfg.index_dir, project)
        if self.index_path_context.exists(single_index_path):
            await invalidate_faiss_cache(single_index_path)
            self.index_path_context.unlink(single_index_path)
        single_uuid_path = single_index_path + ".uuid"
        if self.index_path_context.exists(single_uuid_path):
            self.index_path_context.unlink(single_uuid_path)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _compute_shard_count(chunk_count: int, cfg: EngramConfig) -> int:
    if chunk_count < int(cfg.shard_chunk_threshold):
        return 1
    return max(1, int(math.ceil(chunk_count / max(1, int(cfg.shard_size)))))


def _index_paths(index_dir: str, project_id: ProjectID, shard_id: Optional[int] = None) -> Tuple[str, str]:
    if shard_id is None:
        base = os.path.join(index_dir, str(project_id) + ".index")
    else:
        base = os.path.join(index_dir, str(project_id) + ".shard" + str(shard_id) + ".index")
    return base + ".current", base + ".new"


def _index_artifacts(index_dir: str, project_id: ProjectID, shard_count: int) -> List[Tuple[str, str]]:
    artifacts: List[Tuple[str, str]] = []
    if shard_count <= 1:
        current_path, _ = _index_paths(index_dir, project_id)
        artifacts.append((current_path, "faiss_index"))
        artifacts.append((current_path + ".uuid", "faiss_uuid"))
        artifacts.append((current_path + ".checksum", "faiss_checksum"))
        return artifacts
    for shard_id in range(shard_count):
        current_path, _ = _index_paths(index_dir, project_id, shard_id)
        artifacts.append((current_path, "faiss_index"))
        artifacts.append((current_path + ".uuid", "faiss_uuid"))
        artifacts.append((current_path + ".checksum", "faiss_checksum"))
    return artifacts


def _load_faiss(path: str, *, path_context: Optional[PathContext] = None) -> Any:
    import faiss

    if path_context is not None:
        safe_path = path_context.ensure_allowed_nofollow(path)
        st = path_context.lstat(safe_path)
        if stat.S_ISLNK(st.st_mode):
            raise RuntimeError(f"Refusing to load FAISS index symlink: {safe_path}")
        path = safe_path
    index = faiss.read_index(path, faiss.IO_FLAG_MMAP)
    index = _ensure_id_map(index)
    _enable_direct_map(index)
    return index


def _ensure_id_map(index: Any) -> Any:
    import faiss

    if isinstance(index, faiss.IndexIDMap2):
        return index
    return faiss.IndexIDMap2(index)


def _enable_direct_map(index: Any) -> None:
    if hasattr(index, "make_direct_map"):
        try:
            index.make_direct_map()
        except Exception:
            logging.debug("Failed to create FAISS direct map", exc_info=True)
        return
    inner = getattr(index, "index", None)
    if inner is not None and hasattr(inner, "make_direct_map"):
        try:
            inner.make_direct_map()
        except Exception:
            logging.debug("Failed to create FAISS inner direct map", exc_info=True)


def _compute_file_checksum(path: str) -> str:
    """SHA-256 of *path* in streaming fashion (no full-file memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _save_faiss(index: Any, path_context: PathContext, path: str) -> None:
    import faiss
    path_context.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    dir_path = os.path.dirname(os.path.abspath(path))
    tmp_path = path_context.create_temp_file(dir_path=dir_path, suffix=".index.tmp")
    try:
        faiss.write_index(index, tmp_path)
        # Compute the checksum *before* the atomic replace so the hash
        # covers exactly the bytes that end up at *path*.
        checksum = _compute_file_checksum(tmp_path)
        _replace_with_retries(path_context, tmp_path, path)
        # Persist checksum companion so project_health can detect bit-rot.
        path_context.write_text_atomic(path + ".checksum", checksum, encoding="utf-8")
        try:
            path_context.chmod(path, 0o600)
        except OSError:
            logging.debug("Failed to chmod index file %s", path, exc_info=True)
    except Exception:
        if path_context.exists(tmp_path):
            path_context.unlink(tmp_path)
        raise


def _replace_with_retries(
    path_context: PathContext,
    src: str,
    dest: str,
    *,
    retries: int = 5,
    delay_s: float = 0.1,
) -> None:
    last_exc: Optional[BaseException] = None
    for attempt in range(max(1, retries)):
        try:
            path_context.replace(src, dest)
            return
        except PermissionError as exc:
            last_exc = exc
            if os.name != "nt" or attempt >= retries - 1:
                raise
            time.sleep(delay_s * (attempt + 1))
    if last_exc:
        raise last_exc


def _swap_index_files(path_context: PathContext, replacements: List[Tuple[str, str]]) -> None:
    backups: List[Tuple[str, str]] = []
    try:
        for _, current_path in replacements:
            if path_context.exists(current_path):
                backup_path = current_path + ".bak"
                if path_context.exists(backup_path):
                    path_context.unlink(backup_path)
                _replace_with_retries(path_context, current_path, backup_path)
                backups.append((backup_path, current_path))
        for new_path, current_path in replacements:
            _replace_with_retries(path_context, new_path, current_path)
        for backup_path, _ in backups:
            if path_context.exists(backup_path):
                path_context.unlink(backup_path)
    except Exception:
        for backup_path, current_path in backups:
            if path_context.exists(backup_path):
                _replace_with_retries(path_context, backup_path, current_path)
        for new_path, _ in replacements:
            if path_context.exists(new_path):
                path_context.unlink(new_path)
        raise


async def ensure_index_permissions(path_context: PathContext, index_dir: str, *, db_path: str) -> None:
    """Best-effort chmod of index artifacts to owner-only permissions."""
    try:
        path_context.makedirs(index_dir, exist_ok=True)
    except Exception:
        logging.warning("Failed to create index directory %s", index_dir, exc_info=True)
        return
    try:
        os.chmod(index_dir, 0o700)
    except OSError:
        logging.debug("Failed to chmod index directory %s", index_dir, exc_info=True)
    try:
        await dbmod.init_db(db_path)
        artifacts = await dbmod.list_all_project_artifacts(db_path)
    except Exception:
        logging.warning("Failed to load index artifacts from DB", exc_info=True)
        artifacts = []
    for path in artifacts:
        try:
            path_context.chmod(path, 0o600)
        except OSError:
            logging.debug("Failed to chmod index artifact %s", path, exc_info=True)
    try:
        index_pattern = re.compile(r".+\\.index\\.(current|new|tmp|backup)(\\.uuid)?$")
        for name in path_context.list_dir(index_dir):
            if not index_pattern.match(name):
                continue
            candidate = os.path.join(index_dir, name)
            try:
                st = path_context.lstat(candidate)
            except Exception:
                continue
            if stat.S_ISLNK(st.st_mode):
                continue
            try:
                path_context.chmod(candidate, 0o600)
            except OSError:
                logging.debug("Failed to chmod index artifact %s", candidate, exc_info=True)
    except Exception:
        logging.debug("Failed to chmod stray index artifacts in %s", index_dir, exc_info=True)
