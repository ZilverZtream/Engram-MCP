from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import EngramConfig
from .embeddings import Embedder
from .parsing import parse_directory
from . import db as dbmod


# Global project lock manager to prevent race conditions
_project_locks: Dict[str, asyncio.Lock] = {}
_locks_lock = asyncio.Lock()


async def _get_project_lock(project_id: str) -> asyncio.Lock:
    """Get or create a lock for a specific project."""
    async with _locks_lock:
        if project_id not in _project_locks:
            _project_locks[project_id] = asyncio.Lock()
        return _project_locks[project_id]


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


@dataclass
class Indexer:
    cfg: EngramConfig

    async def _parse(self, directory: str, project_type: str):
        # Parsing is mostly disk IO; run in thread
        return await asyncio.to_thread(
            parse_directory,
            root=directory,
            project_type=project_type,
            ignore_patterns=self.cfg.ignore_patterns,
            chunk_size_tokens=self.cfg.chunk_size_tokens,
            overlap_tokens=self.cfg.overlap_tokens,
            max_file_size_mb=self.cfg.max_file_size_mb,
        )

    async def _embed_chunks(self, embedder: Embedder, chunks) -> np.ndarray:
        texts = [c.content for c in chunks]
        # Batched embedding for memory safety
        batch = max(1, int(self.cfg.embedding_batch_size))
        all_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), batch):
            vecs = await embedder.embed_texts(texts[i:i+batch])
            all_vecs.append(vecs)
        return np.vstack(all_vecs) if all_vecs else np.zeros((0, 0), dtype=np.float32)

    async def _build_faiss(self, vectors: np.ndarray, ids: np.ndarray) -> Any:
        import faiss

        d = int(vectors.shape[1])
        # Inner-product on L2-normalized vectors ~= cosine
        base = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efSearch = 256
        index = faiss.IndexIDMap2(base)

        def _add():
            faiss.normalize_L2(vectors)
            index.add_with_ids(vectors, ids)
            return index

        return await asyncio.to_thread(_add)

    async def index_project(self, *, directory: str, project_name: str, project_type: str) -> str:
        await dbmod.init_db(self.cfg.db_path)

        project_id = f"{_slug(project_name)}_{int(time.time())}"
        directory = os.path.abspath(directory)

        chunks, file_count = await self._parse(directory, project_type)
        if not chunks:
            raise ValueError("No indexable content found.")

        model_name = _pick_model_name(project_type, self.cfg)
        device = "cuda" if _cuda_available() else "cpu"
        embedder = Embedder(model_name=model_name, device=device, prefer_thread_for_cuda=self.cfg.prefer_thread_for_cuda)

        vectors = await self._embed_chunks(embedder, chunks)
        if vectors.size == 0:
            raise ValueError("Embedding generation produced no vectors.")

        embedding_dim = int(vectors.shape[1])

        # Reserve internal IDs
        internal_ids = await dbmod.reserve_internal_ids(self.cfg.db_path, project_id, len(chunks))
        ids = np.asarray(internal_ids, dtype=np.int64)

        # Store project + chunks
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
            },
        )

        rows = [(c.chunk_id, int(iid), c.content, int(c.token_count), c.metadata) for c, iid in zip(chunks, internal_ids)]
        await dbmod.upsert_chunks(self.cfg.db_path, project_id=project_id, rows=rows)

        # Build and save FAISS
        index = await self._build_faiss(vectors, ids)
        safe_id = _sanitize_project_id(project_id)
        index_path = os.path.join(self.cfg.index_dir, f"{safe_id}.index")

        # Acquire lock to prevent race condition with concurrent reads
        lock = await _get_project_lock(project_id)
        async with lock:
            await asyncio.to_thread(_save_faiss, index, index_path)

        await dbmod.refresh_project_stats(self.cfg.db_path, project_id)
        # Save file_count
        await _set_file_count(self.cfg.db_path, project_id, file_count)
        return project_id

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

        project_type = proj["project_type"]
        stored_model = (proj.get("metadata") or {}).get("model_name")
        current_model = _pick_model_name(project_type, self.cfg)

        # Enforce model consistency to prevent vector space pollution
        if stored_model and stored_model != current_model:
            raise ValueError(
                f"Model mismatch: project was indexed with '{stored_model}' but config specifies '{current_model}'. "
                f"Delete and re-index the project, or update config to use the original model."
            )

        model_name = stored_model or current_model
        device = (proj.get("metadata") or {}).get("device") or ("cuda" if _cuda_available() else "cpu")

        # Parse current directory
        chunks, file_count = await self._parse(directory, project_type)

        # Determine adds/deletes by chunk_id
        existing_ids = await _fetch_existing_chunk_ids(self.cfg.db_path, project_id)
        new_ids = {c.chunk_id for c in chunks}

        to_delete = sorted(existing_ids - new_ids)
        to_add = [c for c in chunks if c.chunk_id not in existing_ids]

        # Fast path: nothing changed
        if not to_delete and not to_add:
            await _set_file_count(self.cfg.db_path, project_id, file_count)
            return {"added": 0, "deleted": 0}

        # Rebuild full index for correctness (handles deletions cleanly)
        # Delete removed chunks
        await dbmod.delete_chunks(self.cfg.db_path, project_id, to_delete)

        # Assign internal IDs to new chunks and store
        internal_ids = await dbmod.reserve_internal_ids(self.cfg.db_path, project_id, len(to_add))
        rows = [(c.chunk_id, int(iid), c.content, int(c.token_count), c.metadata) for c, iid in zip(to_add, internal_ids)]
        if rows:
            await dbmod.upsert_chunks(self.cfg.db_path, project_id=project_id, rows=rows)

        # Fetch ALL chunks back in internal_id order to rebuild FAISS
        all_rows = await _fetch_all_chunks_for_rebuild(self.cfg.db_path, project_id)

        embedder = Embedder(model_name=model_name, device=device, prefer_thread_for_cuda=self.cfg.prefer_thread_for_cuda)
        vectors = await self._embed_chunks(embedder, [r["chunk"] for r in all_rows])
        ids = np.asarray([r["internal_id"] for r in all_rows], dtype=np.int64)

        index = await self._build_faiss(vectors, ids)
        safe_id = _sanitize_project_id(project_id)
        index_path = os.path.join(self.cfg.index_dir, f"{safe_id}.index")

        # Acquire lock to prevent race condition with concurrent reads
        lock = await _get_project_lock(project_id)
        async with lock:
            await asyncio.to_thread(_save_faiss, index, index_path)

        await dbmod.refresh_project_stats(self.cfg.db_path, project_id)
        await _set_file_count(self.cfg.db_path, project_id, file_count)
        return {"added": len(to_add), "deleted": len(to_delete)}


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


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
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE projects SET file_count = ?, updated_at = datetime('now') WHERE project_id = ?",
            (int(file_count), project_id),
        )
        await db.commit()


async def _fetch_existing_chunk_ids(db_path: str, project_id: str) -> set[str]:
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        rows = await db.execute_fetchall(
            "SELECT chunk_id FROM chunks WHERE project_id = ?",
            (project_id,),
        )
    return {r[0] for r in rows}


async def _fetch_all_chunks_for_rebuild(db_path: str, project_id: str) -> List[Dict[str, Any]]:
    import aiosqlite

    from .chunking import Chunk

    async with aiosqlite.connect(db_path) as db:
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
