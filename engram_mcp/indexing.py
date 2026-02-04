from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import EngramConfig
from .embeddings import Embedder
from .parsing import (
    ParsedFile,
    parse_directory,
    iter_files,
    parse_file_to_chunks,
    SUPPORTED_DOC_EXTS,
    SUPPORTED_TEXT_EXTS,
    hash_file,
)
from .locks import get_project_lock, get_project_rwlock
from .search import invalidate_faiss_cache
from .security import enforce_allowed_roots
from . import db as dbmod


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "project"


def _sanitize_project_id(project_id: str) -> str:
    """Sanitize project_id to prevent directory traversal attacks."""
    # Use basename to strip any directory components
    sanitized = os.path.basename(project_id)
    # Additional validation: only allow alphanumeric, underscore, and hyphen
    if not re.match(r"^[a-zA-Z0-9_-]+$", sanitized):
        raise ValueError(f"Invalid project_id: {project_id}. Only alphanumeric characters, underscores, and hyphens are allowed.")
    return sanitized


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
    iterator = iter(items)
    pending: set[asyncio.Task] = set()
    for _ in range(concurrency):
        try:
            item = next(iterator)
        except StopIteration:
            break
        pending.add(asyncio.create_task(worker(item)))

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            yield task.result()
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.add(asyncio.create_task(worker(item)))


@dataclass
class Indexer:
    cfg: EngramConfig

    async def _parse(self, directory: str, project_type: str):
        return await parse_directory(
            root=directory,
            project_type=project_type,
            ignore_patterns=self.cfg.ignore_patterns,
            chunk_size_tokens=self.cfg.chunk_size_tokens,
            overlap_tokens=self.cfg.overlap_tokens,
            max_file_size_mb=self.cfg.max_file_size_mb,
        )

    async def _embed_chunks_cached(
        self,
        embedder: Embedder,
        chunks: List[Any],
        *,
        model_name: str,
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
        async for vecs in embedder.embed_texts_batched(to_embed_texts, batch_size=batch):
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
            )

        if any(v is None for v in vectors):
            raise RuntimeError("Embedding cache did not resolve all vectors.")
        return np.vstack([v for v in vectors]).astype(np.float32)

    async def _build_faiss_index(self, vectors: np.ndarray) -> Any:
        import faiss

        d = int(vectors.shape[1])
        nlist = max(1, int(self.cfg.faiss_nlist))
        m = max(1, int(self.cfg.faiss_m))
        nbits = max(1, int(self.cfg.faiss_nbits))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(
            quantizer,
            d,
            nlist,
            m,
            nbits,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.nprobe = max(1, int(self.cfg.faiss_nprobe))

        def _train():
            faiss.normalize_L2(vectors)
            if not index.is_trained:
                index.train(vectors)
            return index

        return await asyncio.to_thread(_train)

    async def _add_vectors(self, index: Any, vectors: np.ndarray, ids: np.ndarray) -> None:
        import faiss

        def _add() -> None:
            faiss.normalize_L2(vectors)
            index.add_with_ids(vectors, ids)

        await asyncio.to_thread(_add)

    async def _persist_index_batch(
        self,
        *,
        project_id: str,
        index: Any,
        vectors: np.ndarray,
        ids: np.ndarray,
        rows: List[Tuple[str, int, str, int, Dict[str, Any]]],
        index_path: str,
        rwlock: Any,
    ) -> None:
        chunk_ids = [row[0] for row in rows]
        await dbmod.upsert_chunks(self.cfg.db_path, project_id=project_id, rows=rows)
        try:
            await self._add_vectors(index, vectors, ids)
            async with rwlock.write_lock():
                await invalidate_faiss_cache(index_path)
                await asyncio.to_thread(_save_faiss, index, index_path)
        except Exception:
            await dbmod.delete_chunks(self.cfg.db_path, project_id, chunk_ids)
            raise

    async def index_project(self, *, directory: str, project_name: str, project_type: str) -> str:
        await dbmod.init_db(self.cfg.db_path)

        project_id = f"{_slug(project_name)}_{int(time.time())}"
        directory = os.path.abspath(directory)
        directory_realpath = os.path.realpath(directory)

        model_name = _pick_model_name(project_type, self.cfg)
        device = "cuda" if _cuda_available() else "cpu"
        embedder = Embedder(
            model_name=model_name,
            device=device,
            prefer_thread_for_cuda=self.cfg.prefer_thread_for_cuda,
        )

        project_initialized = False
        index_path: Optional[str] = None
        try:
            file_count = 0

            def iter_indexable_files() -> Iterable[Any]:
                nonlocal file_count
                for p in iter_files(directory, self.cfg.ignore_patterns):
                    ext = p.suffix.lower()
                    if ext in SUPPORTED_TEXT_EXTS or ext in SUPPORTED_DOC_EXTS:
                        file_count += 1
                        yield p

            sem = asyncio.Semaphore(os.cpu_count() or 4)
            max_bytes = int(self.cfg.max_file_size_mb) * 1024 * 1024

            async def parse_one(path: Any) -> Optional[ParsedFile]:
                rel = str(path.relative_to(directory))
                async with sem:
                    try:
                        chunks = await asyncio.to_thread(
                            parse_file_to_chunks,
                            path=path,
                            root=directory,
                            project_type=project_type,
                            chunk_size_tokens=self.cfg.chunk_size_tokens,
                            overlap_tokens=self.cfg.overlap_tokens,
                            max_file_size_mb=self.cfg.max_file_size_mb,
                        )
                        content_hash = await asyncio.to_thread(hash_file, path, Path(directory), max_bytes)
                    except Exception:
                        logging.warning("Failed to parse %s", rel, exc_info=True)
                        return None
                stat = path.stat()
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
            training_vectors: List[np.ndarray] = []
            total_chunks = 0
            file_metadata_rows: List[dbmod.FileMetadataRow] = []
            train_target = max(1000, int(self.cfg.faiss_nlist) * 20)
            safe_id = _sanitize_project_id(project_id)
            index_path = os.path.join(self.cfg.index_dir, f"{safe_id}.index")
            rwlock = await get_project_rwlock(project_id)

            async for parsed in _bounded_gather(iter_indexable_files(), parse_one, max(1, int(os.cpu_count() or 4))):
                if parsed is None:
                    continue

                chunk_ids = [c.chunk_id for c in parsed.chunks]
                file_metadata_rows.append(
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

                vectors = await self._embed_chunks_cached(embedder, parsed.chunks, model_name=model_name)
                if vectors.size == 0:
                    continue
                if embedding_dim is None:
                    embedding_dim = int(vectors.shape[1])

                if not project_initialized:
                    await dbmod.upsert_project(
                        self.cfg.db_path,
                        project_id=project_id,
                        project_name=project_name,
                        project_type=project_type,
                        directory=directory,
                        embedding_dim=embedding_dim,
                        metadata={
                            "chunk_size_tokens": self.cfg.chunk_size_tokens,
                            "overlap_tokens": self.cfg.overlap_tokens,
                            "model_name": model_name,
                            "device": device,
                            "shard_count": 1,
                            "shard_size": int(self.cfg.shard_size),
                            "directory_realpath": directory_realpath,
                        },
                    )
                    project_initialized = True

                internal_ids = await dbmod.reserve_internal_ids(self.cfg.db_path, project_id, len(parsed.chunks))
                ids = np.asarray(internal_ids, dtype=np.int64)

                rows = [
                    (c.chunk_id, int(iid), c.content, int(c.token_count), c.metadata)
                    for c, iid in zip(parsed.chunks, internal_ids)
                ]

                if index is None:
                    training_vectors.append(vectors)
                    pending_vectors.append(vectors)
                    pending_ids.append(ids)
                    pending_rows.append(rows)
                    if sum(v.shape[0] for v in training_vectors) >= train_target:
                        train_vecs = np.vstack(training_vectors)
                        index = await self._build_faiss_index(train_vecs)
                        for vec_batch, id_batch, row_batch in zip(pending_vectors, pending_ids, pending_rows):
                            await self._persist_index_batch(
                                project_id=project_id,
                                index=index,
                                vectors=vec_batch,
                                ids=id_batch,
                                rows=row_batch,
                                index_path=index_path,
                                rwlock=rwlock,
                            )
                        training_vectors.clear()
                        pending_vectors.clear()
                        pending_ids.clear()
                        pending_rows.clear()
                else:
                    await self._persist_index_batch(
                        project_id=project_id,
                        index=index,
                        vectors=vectors,
                        ids=ids,
                        rows=rows,
                        index_path=index_path,
                        rwlock=rwlock,
                    )

                total_chunks += len(parsed.chunks)

            if file_count == 0:
                raise ValueError("No indexable content found.")

            if total_chunks == 0:
                raise ValueError("No indexable content found.")

            if embedding_dim is None:
                raise ValueError("Embedding generation produced no vectors.")

            if not project_initialized:
                await dbmod.upsert_project(
                    self.cfg.db_path,
                    project_id=project_id,
                    project_name=project_name,
                    project_type=project_type,
                    directory=directory,
                    embedding_dim=embedding_dim,
                    metadata={
                        "chunk_size_tokens": self.cfg.chunk_size_tokens,
                        "overlap_tokens": self.cfg.overlap_tokens,
                        "model_name": model_name,
                        "device": device,
                        "shard_count": 1,
                        "shard_size": int(self.cfg.shard_size),
                        "directory_realpath": directory_realpath,
                    },
                )

            if index is None:
                train_vecs = np.vstack(training_vectors) if training_vectors else np.vstack(pending_vectors)
                index = await self._build_faiss_index(train_vecs)
                for vec_batch, id_batch, row_batch in zip(pending_vectors, pending_ids, pending_rows):
                    await self._persist_index_batch(
                        project_id=project_id,
                        index=index,
                        vectors=vec_batch,
                        ids=id_batch,
                        rows=row_batch,
                        index_path=index_path,
                        rwlock=rwlock,
                    )
                pending_rows.clear()

            await dbmod.upsert_file_metadata(self.cfg.db_path, project_id, file_metadata_rows)

            await dbmod.refresh_project_stats(self.cfg.db_path, project_id)
            await _set_file_count(self.cfg.db_path, project_id, file_count)

            await self._maybe_shard_project(project_id=project_id)
            return project_id
        except BaseException:
            if project_initialized:
                await dbmod.delete_project(self.cfg.db_path, project_id)
                if index_path and os.path.exists(index_path):
                    os.unlink(index_path)
            raise
        finally:
            embedder.close()

    async def update_project(self, *, project_id: str) -> Dict[str, int]:
        # Sanitize project_id to prevent path traversal
        project_id = _sanitize_project_id(project_id)
        await dbmod.init_db(self.cfg.db_path)
        proj = await dbmod.get_project(self.cfg.db_path, project_id)
        if not proj:
            raise ValueError(f"Project not found: {project_id}")

        directory = proj.get("directory")
        if not directory:
            raise ValueError("Project has no stored directory; cannot update.")
        directory = enforce_allowed_roots(directory, self.cfg.allowed_roots)
        metadata = proj.get("metadata") or {}
        stored_realpath = metadata.get("directory_realpath")
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

        lock = await get_project_lock(project_id)
        async with lock:
            existing_meta = await dbmod.fetch_file_metadata(self.cfg.db_path, project_id)
            rwlock = await get_project_rwlock(project_id)

            current_files: Dict[str, Tuple[int, int, str, Any]] = {}
            max_bytes = int(self.cfg.max_file_size_mb) * 1024 * 1024
            for p in iter_files(directory, self.cfg.ignore_patterns):
                ext = p.suffix.lower()
                if ext not in SUPPORTED_TEXT_EXTS and ext not in SUPPORTED_DOC_EXTS:
                    continue
                rel = str(p.relative_to(directory))
                stat = p.stat()
                content_hash = ""
                existing = existing_meta.get(rel)
                if existing and existing.mtime_ns == stat.st_mtime_ns and existing.size_bytes == stat.st_size:
                    try:
                        content_hash = hash_file(p, Path(directory), max_bytes)
                    except Exception:
                        logging.warning("Failed to hash %s", rel, exc_info=True)
                        continue
                    if existing.content_hash == content_hash:
                        current_files[rel] = (stat.st_mtime_ns, stat.st_size, content_hash, p)
                        continue
                if not content_hash:
                    try:
                        content_hash = hash_file(p, Path(directory), max_bytes)
                    except Exception:
                        logging.warning("Failed to hash %s", rel, exc_info=True)
                        continue
                current_files[rel] = (stat.st_mtime_ns, stat.st_size, content_hash, p)

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
                chunk_map = await dbmod.fetch_file_chunk_ids(self.cfg.db_path, project_id, paths_for_deletes)
                for path in paths_for_deletes:
                    to_delete_chunk_ids.extend(chunk_map.get(path, []))

            if not to_delete_chunk_ids and not changed_files:
                await _set_file_count(self.cfg.db_path, project_id, file_count)
                if not stored_realpath:
                    metadata["directory_realpath"] = current_realpath
                    await dbmod.upsert_project(
                        self.cfg.db_path,
                        project_id=project_id,
                        project_name=proj["project_name"],
                        project_type=project_type,
                        directory=directory,
                        embedding_dim=int(proj.get("embedding_dim") or 0),
                        metadata=metadata,
                    )
                return {"added": 0, "deleted": 0}

            embedder = Embedder(
                model_name=model_name,
                device=device,
                prefer_thread_for_cuda=self.cfg.prefer_thread_for_cuda,
            )

            added_chunk_ids: List[str] = []
            try:
                internal_ids_to_delete = await dbmod.fetch_internal_ids_for_chunk_ids(
                    self.cfg.db_path, project_id, to_delete_chunk_ids
                )

                shard_count = int(metadata.get("shard_count") or 1)
                added_count = 0
                updated_file_rows: List[dbmod.FileMetadataRow] = []
                sem = asyncio.Semaphore(os.cpu_count() or 4)

                async def parse_one(rel_path: str) -> Optional[ParsedFile]:
                    mtime_ns, size_bytes, content_hash, path = current_files[rel_path]
                    async with sem:
                        try:
                            chunks = await asyncio.to_thread(
                                parse_file_to_chunks,
                                path=path,
                                root=directory,
                                project_type=project_type,
                                chunk_size_tokens=self.cfg.chunk_size_tokens,
                                overlap_tokens=self.cfg.overlap_tokens,
                                max_file_size_mb=self.cfg.max_file_size_mb,
                            )
                        except Exception:
                            logging.warning("Failed to parse %s", rel_path, exc_info=True)
                            return None
                    return ParsedFile(
                        file_path=rel_path,
                        mtime_ns=mtime_ns,
                        size_bytes=size_bytes,
                        content_hash=content_hash,
                        chunks=chunks,
                    )

                safe_id = _sanitize_project_id(project_id)
                if shard_count > 1:
                    shard_paths = [
                        os.path.join(self.cfg.index_dir, f"{safe_id}.shard{shard_id}.index")
                        for shard_id in range(shard_count)
                    ]
                    shard_indexes: List[Any] = []
                    for path in shard_paths:
                        if os.path.exists(path):
                            shard_indexes.append(await asyncio.to_thread(_load_faiss, path))
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

                        async for parsed in _bounded_gather(
                            sorted(changed_files), parse_one, max(1, int(os.cpu_count() or 4))
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
                            vectors = await self._embed_chunks_cached(embedder, parsed.chunks, model_name=model_name)
                            internal_ids = await dbmod.reserve_internal_ids(
                                self.cfg.db_path, project_id, len(parsed.chunks)
                            )
                            rows = [
                                (c.chunk_id, int(iid), c.content, int(c.token_count), c.metadata)
                                for c, iid in zip(parsed.chunks, internal_ids)
                            ]
                            await dbmod.upsert_chunks(self.cfg.db_path, project_id=project_id, rows=rows)
                            added_chunk_ids.extend(chunk_ids)
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

                        for shard_id, index in enumerate(shard_indexes):
                            if index is None:
                                continue
                            await invalidate_faiss_cache(shard_paths[shard_id])
                            await asyncio.to_thread(_save_faiss, index, shard_paths[shard_id])
                else:
                    index_path = os.path.join(self.cfg.index_dir, f"{safe_id}.index")
                    async with rwlock.write_lock():
                        index = await asyncio.to_thread(_load_faiss, index_path)
                        if internal_ids_to_delete:
                            index.remove_ids(np.asarray(internal_ids_to_delete, dtype=np.int64))
                        async for parsed in _bounded_gather(
                            sorted(changed_files), parse_one, max(1, int(os.cpu_count() or 4))
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
                            vectors = await self._embed_chunks_cached(embedder, parsed.chunks, model_name=model_name)
                            internal_ids = await dbmod.reserve_internal_ids(
                                self.cfg.db_path, project_id, len(parsed.chunks)
                            )
                            rows = [
                                (c.chunk_id, int(iid), c.content, int(c.token_count), c.metadata)
                                for c, iid in zip(parsed.chunks, internal_ids)
                            ]
                            await dbmod.upsert_chunks(self.cfg.db_path, project_id=project_id, rows=rows)
                            added_chunk_ids.extend(chunk_ids)
                            await self._add_vectors(index, vectors, np.asarray(internal_ids, dtype=np.int64))
                            added_count += len(parsed.chunks)

                        await invalidate_faiss_cache(index_path)
                        await asyncio.to_thread(_save_faiss, index, index_path)

                if to_delete_chunk_ids:
                    await dbmod.delete_chunks(self.cfg.db_path, project_id, to_delete_chunk_ids)

                if updated_file_rows:
                    await dbmod.upsert_file_metadata(self.cfg.db_path, project_id, updated_file_rows)
                if deleted_files:
                    await dbmod.delete_file_metadata(self.cfg.db_path, project_id, sorted(deleted_files))

                metadata["directory_realpath"] = current_realpath
                await dbmod.upsert_project(
                    self.cfg.db_path,
                    project_id=project_id,
                    project_name=proj["project_name"],
                    project_type=project_type,
                    directory=directory,
                    embedding_dim=int(proj.get("embedding_dim") or 0),
                    metadata=metadata,
                )

                await dbmod.refresh_project_stats(self.cfg.db_path, project_id)
                await _set_file_count(self.cfg.db_path, project_id, file_count)
                await self._maybe_shard_project(project_id=project_id)
                return {"added": added_count, "deleted": len(to_delete_chunk_ids)}
            except BaseException:
                if added_chunk_ids:
                    await dbmod.delete_chunks(self.cfg.db_path, project_id, added_chunk_ids)
                raise
            finally:
                embedder.close()

    async def _update_sharded_indexes(
        self,
        *,
        project_id: str,
        safe_id: str,
        shard_count: int,
        internal_ids_to_delete: List[int],
        to_add_chunks: List[Any],
        embedder: Embedder,
        model_name: str,
    ) -> None:
        rwlock = await get_project_rwlock(project_id)
        shard_paths = [
            os.path.join(self.cfg.index_dir, f"{safe_id}.shard{shard_id}.index")
            for shard_id in range(shard_count)
        ]
        shard_indexes: List[Any] = []
        for path in shard_paths:
            if os.path.exists(path):
                shard_indexes.append(await asyncio.to_thread(_load_faiss, path))
            else:
                shard_indexes.append(None)

        async with rwlock.write_lock():
            if internal_ids_to_delete:
                for shard_id, index in enumerate(shard_indexes):
                    if index is None:
                        continue
                    ids_for_shard = [iid for iid in internal_ids_to_delete if iid % shard_count == shard_id]
                    if ids_for_shard:
                        index.remove_ids(np.asarray(ids_for_shard, dtype=np.int64))

            if to_add_chunks:
                vectors = await self._embed_chunks_cached(embedder, to_add_chunks, model_name=model_name)
                internal_ids = await dbmod.reserve_internal_ids(self.cfg.db_path, project_id, len(to_add_chunks))
                rows = [
                    (c.chunk_id, int(iid), c.content, int(c.token_count), c.metadata)
                    for c, iid in zip(to_add_chunks, internal_ids)
                ]
                await dbmod.upsert_chunks(self.cfg.db_path, project_id=project_id, rows=rows)

                for shard_id in range(shard_count):
                    shard_mask = [i for i, iid in enumerate(internal_ids) if iid % shard_count == shard_id]
                    if not shard_mask:
                        continue
                    shard_vectors = vectors[shard_mask]
                    shard_ids = np.asarray([internal_ids[i] for i in shard_mask], dtype=np.int64)
                    if shard_indexes[shard_id] is None:
                        shard_indexes[shard_id] = await self._build_faiss_index(shard_vectors)
                    await self._add_vectors(shard_indexes[shard_id], shard_vectors, shard_ids)

            for shard_id, index in enumerate(shard_indexes):
                if index is None:
                    continue
                await invalidate_faiss_cache(shard_paths[shard_id])
                await asyncio.to_thread(_save_faiss, index, shard_paths[shard_id])

    async def _maybe_shard_project(self, *, project_id: str) -> None:
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
        embedder = Embedder(
            model_name=model_name,
            device=metadata.get("device") or "cpu",
            prefer_thread_for_cuda=self.cfg.prefer_thread_for_cuda,
        )

        try:
            rwlock = await get_project_rwlock(project_id)
            all_rows = await _fetch_all_chunks_for_rebuild(self.cfg.db_path, project_id)
            vectors = await self._embed_chunks_cached(embedder, [r["chunk"] for r in all_rows], model_name=model_name)
            ids = [r["internal_id"] for r in all_rows]

            shards: List[List[int]] = [[] for _ in range(shard_count)]
            shard_vectors: List[List[int]] = [[] for _ in range(shard_count)]
            for i, iid in enumerate(ids):
                shard_id = iid % shard_count
                shards[shard_id].append(iid)
                shard_vectors[shard_id].append(i)

            safe_id = _sanitize_project_id(project_id)
            async with rwlock.write_lock():
                for shard_id in range(shard_count):
                    if not shard_vectors[shard_id]:
                        continue
                    vecs = vectors[shard_vectors[shard_id]]
                    index = await self._build_faiss_index(vecs)
                    await self._add_vectors(index, vecs, np.asarray(shards[shard_id], dtype=np.int64))
                    path = os.path.join(self.cfg.index_dir, f"{safe_id}.shard{shard_id}.index")
                    await invalidate_faiss_cache(path)
                    await asyncio.to_thread(_save_faiss, index, path)

            metadata["shard_count"] = shard_count
            metadata["shard_size"] = int(self.cfg.shard_size)
            await dbmod.upsert_project(
                self.cfg.db_path,
                project_id=project_id,
                project_name=proj["project_name"],
                project_type=proj["project_type"],
                directory=proj.get("directory"),
                embedding_dim=int(proj.get("embedding_dim") or 0),
                metadata=metadata,
            )

            single_index_path = os.path.join(self.cfg.index_dir, f"{safe_id}.index")
            if os.path.exists(single_index_path):
                await invalidate_faiss_cache(single_index_path)
                os.unlink(single_index_path)
        finally:
            embedder.close()


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


def _load_faiss(path: str) -> Any:
    import faiss

    return faiss.read_index(path)


def _save_faiss(index: Any, path: str) -> None:
    import faiss
    import tempfile

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    # Write to temporary file first for atomic update
    dir_path = os.path.dirname(os.path.abspath(path))
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=dir_path, suffix='.index.tmp') as tmp:
        tmp_path = tmp.name

    try:
        faiss.write_index(index, tmp_path)
        # Atomic rename (POSIX guarantees atomicity)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


async def _set_file_count(db_path: str, project_id: str, file_count: int) -> None:
    async with dbmod.get_connection(db_path) as db:
        await db.execute(
            "UPDATE projects SET file_count = ?, updated_at = datetime('now') WHERE project_id = ?",
            (int(file_count), project_id),
        )
        await db.commit()


async def _fetch_existing_chunk_ids(db_path: str, project_id: str) -> set[str]:
    async with dbmod.get_connection(db_path) as db:
        rows = await db.execute_fetchall(
            "SELECT chunk_id FROM chunks WHERE project_id = ?",
            (project_id,),
        )
    return {r[0] for r in rows}


async def _fetch_all_chunks_for_rebuild(db_path: str, project_id: str) -> List[Dict[str, Any]]:
    from .chunking import Chunk

    async with dbmod.get_connection(db_path) as db:
        rows = await db.execute_fetchall(
            "SELECT chunk_id, internal_id, content, token_count, metadata FROM chunks WHERE project_id = ? ORDER BY internal_id ASC",
            (project_id,),
        )

    out: List[Dict[str, Any]] = []
    for cid, iid, content, tcount, meta_json in rows:
        meta = {}  # already json in db
        try:
            import json

            meta = json.loads(meta_json or "{}")
        except Exception:
            meta = {}
        out.append(
            {
                "chunk": Chunk(chunk_id=cid, content=content, token_count=int(tcount), metadata=meta),
                "internal_id": int(iid),
            }
        )
    return out
