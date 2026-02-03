from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numba import jit

from . import db as dbmod
from .locks import get_project_lock


def _sanitize_project_id(project_id: str) -> str:
    """Sanitize project_id to prevent directory traversal attacks."""
    # Use basename to strip any directory components
    sanitized = os.path.basename(project_id)
    # Additional validation: only allow alphanumeric, underscore, and hyphen
    if not re.match(r"^[a-zA-Z0-9_-]+$", sanitized):
        raise ValueError(f"Invalid project_id: {project_id}. Only alphanumeric characters, underscores, and hyphens are allowed.")
    return sanitized


@jit(nopython=True)
def _rrf_scores(bm25_indices: np.ndarray, vec_indices: np.ndarray, k: int = 60) -> np.ndarray:
    size = 0
    if bm25_indices.size:
        size = max(size, int(np.max(bm25_indices)) + 1)
    if vec_indices.size:
        size = max(size, int(np.max(vec_indices)) + 1)
    scores = np.zeros(size, dtype=np.float32)
    for rank in range(bm25_indices.shape[0]):
        idx = int(bm25_indices[rank])
        scores[idx] += 1.0 / (k + rank + 1)
    for rank in range(vec_indices.shape[0]):
        idx = int(vec_indices[rank])
        scores[idx] += 1.0 / (k + rank + 1)
    return scores


@jit(nopython=True)
def _mmr_select(E: np.ndarray, q: np.ndarray, k: int, mmr_lambda: float) -> np.ndarray:
    n = E.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, n)
    rel = E @ q
    selected = np.empty(k, dtype=np.int64)
    used = np.zeros(n, dtype=np.uint8)
    first = int(np.argmax(rel))
    selected[0] = first
    used[first] = 1
    selected_count = 1

    while selected_count < k:
        best = -1
        best_score = -1e9
        for i in range(n):
            if used[i]:
                continue
            max_sim = -1e9
            for j in range(selected_count):
                sim = np.dot(E[i], E[selected[j]])
                if sim > max_sim:
                    max_sim = sim
            score = mmr_lambda * rel[i] - (1.0 - mmr_lambda) * max_sim
            if score > best_score:
                best_score = score
                best = i
        if best == -1:
            break
        selected[selected_count] = best
        used[best] = 1
        selected_count += 1

    return selected[:selected_count]


def rrf_combine(bm25_results: List[Dict[str, Any]], vec_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion."""
    all_ids: List[str] = []
    all_results: Dict[str, Dict[str, Any]] = {}

    for rank, r in enumerate(bm25_results, 1):
        cid = r["id"]
        if cid not in all_results:
            all_results[cid] = r
            all_ids.append(cid)
        r["lex_rank"] = rank

    for rank, r in enumerate(vec_results, 1):
        cid = r["id"]
        if cid not in all_results:
            all_results[cid] = r
            all_ids.append(cid)
        r["vec_rank"] = rank

    id_to_index = {cid: i for i, cid in enumerate(all_ids)}
    bm25_idx = np.asarray([id_to_index[r["id"]] for r in bm25_results], dtype=np.int64)
    vec_idx = np.asarray([id_to_index[r["id"]] for r in vec_results], dtype=np.int64)
    scores = _rrf_scores(bm25_idx, vec_idx, k)
    order = np.argsort(-scores)

    combined = []
    for idx in order:
        cid = all_ids[int(idx)]
        score = float(scores[int(idx)])
        out = dict(all_results[cid])
        out["relevance_score"] = score
        combined.append(out)

    return combined


def apply_mmr(results: List[Dict[str, Any]], query_vec: np.ndarray, k: int, mmr_lambda: float = 0.7) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance re-ranking (requires embeddings in results)."""
    k = min(k, len(results))
    if k <= 1:
        return results[:k]

    emb = []
    valid = []
    for r in results:
        v = r.get("embedding")
        if v is None:
            continue
        emb.append(v)
        valid.append(r)

    if len(emb) < 2:
        return valid[:k]

    E = np.asarray(emb, dtype=np.float32)
    # Normalize
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)

    rel = E @ q

    selected = _mmr_select(E, q, k, float(mmr_lambda))
    return [valid[int(i)] for i in selected]


_search_cache: OrderedDict[str, Tuple[float, List[Dict[str, Any]]]] = OrderedDict()
_cache_ttl = 300.0
_cache_max_items = 512
_cache_lock = asyncio.Lock()
_faiss_cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
_faiss_cache_max_items = 8
_faiss_cache_lock = asyncio.Lock()


@dataclass
class SearchEngine:
    db_path: str
    index_dir: str

    async def _load_faiss(self, project_id: str):
        import faiss

        safe_id = _sanitize_project_id(project_id)
        index_path = os.path.join(self.index_dir, f"{safe_id}.index")
        return await self._load_faiss_path(project_id, index_path)

    async def _load_faiss_path(self, project_id: str, index_path: str):
        import faiss

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        index_mtime = os.path.getmtime(index_path)
        async with _faiss_cache_lock:
            cached = _faiss_cache.get(index_path)
            if cached and cached[0] == index_mtime:
                _faiss_cache.move_to_end(index_path)
                return cached[1]
            _faiss_cache.pop(index_path, None)

        # Acquire lock to prevent race condition with updates
        lock = await get_project_lock(project_id)
        async with lock:
            # Load in thread to avoid blocking event loop
            def _load():
                # Use IO_FLAG_READ_ONLY instead of MMAP to avoid SIGBUS issues
                return faiss.read_index(index_path)

            loop = asyncio.get_running_loop()
            index = await loop.run_in_executor(None, _load)

        async with _faiss_cache_lock:
            _faiss_cache[index_path] = (index_mtime, index)
            _faiss_cache.move_to_end(index_path)
            while len(_faiss_cache) > _faiss_cache_max_items:
                _faiss_cache.popitem(last=False)
        return index

    async def vector_search(
        self,
        *,
        project_id: str,
        query_vec: np.ndarray,
        top_k: int,
        index_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Vector search against FAISS index (runs in a thread to keep event loop responsive)."""
        import faiss

        loop = asyncio.get_running_loop()
        index = await self._load_faiss_path(project_id, index_path) if index_path else await self._load_faiss(project_id)

        def _search_sync() -> Tuple[List[int], List[float]]:
            q = np.asarray([query_vec], dtype=np.float32)
            faiss.normalize_L2(q)
            D, I = index.search(q, top_k)
            ids = [int(i) for i in I[0] if int(i) != -1]
            scores = [float(d) for d, i in zip(D[0], I[0]) if int(i) != -1]
            return ids, scores

        internal_ids, scores = await loop.run_in_executor(None, _search_sync)
        rows = await dbmod.fetch_chunks_by_internal_ids(self.db_path, project_id, internal_ids)

        # Map internal_id -> score rank (distance unavailable in this abstraction; rank is ok for RRF)
        id_to_rank = {iid: r for r, iid in enumerate(internal_ids, 1)}
        id_to_score = {iid: score for iid, score in zip(internal_ids, scores)}

        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "id": row.chunk_id,
                    "internal_id": row.internal_id,
                    "content": row.content,
                    "token_count": row.token_count,
                    "metadata": row.metadata,
                    "access_count": row.access_count,
                    "embedding": query_vec,  # only used for MMR; cheap to store query vec
                    "vec_rank": id_to_rank.get(row.internal_id, 10**9),
                    "vec_score": id_to_score.get(row.internal_id, 0.0),
                }
            )
        # Preserve FAISS rank order
        out.sort(key=lambda r: r["vec_rank"])
        return out

    async def _vector_search_sharded(
        self,
        *,
        project_id: str,
        query_vec: np.ndarray,
        top_k: int,
        shard_count: int,
    ) -> List[Dict[str, Any]]:
        safe_id = _sanitize_project_id(project_id)
        shard_paths = [
            os.path.join(self.index_dir, f"{safe_id}.shard{shard_id}.index")
            for shard_id in range(shard_count)
        ]
        tasks = [
            self.vector_search(project_id=project_id, query_vec=query_vec, top_k=top_k, index_path=path)
            for path in shard_paths
            if os.path.exists(path)
        ]
        if not tasks:
            return await self.vector_search(project_id=project_id, query_vec=query_vec, top_k=top_k)

        shard_results = await asyncio.gather(*tasks)
        merged = [item for sub in shard_results for item in sub]
        merged.sort(key=lambda r: r.get("vec_score", 0.0), reverse=True)
        return merged[:top_k]

    async def search(
        self,
        *,
        project_id: str,
        query: str,
        query_vec: np.ndarray,
        fts_top_k: int,
        vector_top_k: int,
        return_k: int,
        enable_mmr: bool,
        mmr_lambda: float,
    ) -> List[Dict[str, Any]]:
        params_hash = hashlib.sha256(
            f"{fts_top_k}:{vector_top_k}:{return_k}:{enable_mmr}:{mmr_lambda}".encode("utf-8")
        ).hexdigest()
        cache_key = f"{project_id}:{hashlib.sha256(query.encode('utf-8')).hexdigest()}:{params_hash}"
        async with _cache_lock:
            cached = _search_cache.get(cache_key)
            if cached and (time.time() - cached[0]) < _cache_ttl:
                _search_cache.move_to_end(cache_key)
                return cached[1]
            if cached:
                _search_cache.pop(cache_key, None)

        lex = await dbmod.fts_search(self.db_path, project_id=project_id, query=query, limit=fts_top_k)
        proj = await dbmod.get_project(self.db_path, project_id)
        shard_count = int((proj.get("metadata") or {}).get("shard_count") or 1) if proj else 1
        if shard_count > 1:
            vec = await self._vector_search_sharded(
                project_id=project_id, query_vec=query_vec, top_k=vector_top_k, shard_count=shard_count
            )
        else:
            vec = await self.vector_search(project_id=project_id, query_vec=query_vec, top_k=vector_top_k)

        # Offload RRF calculation to thread to prevent event loop blocking
        combined = await asyncio.to_thread(rrf_combine, lex, vec, 60)
        combined = combined[: max(return_k, 50)]

        if enable_mmr:
            # Offload MMR calculation to thread as well
            combined = await asyncio.to_thread(apply_mmr, combined, query_vec, return_k, mmr_lambda)
        else:
            combined = combined[:return_k]

        # bump access counts best-effort
        await dbmod.bump_access_counts(self.db_path, project_id, [r["id"] for r in combined])
        async with _cache_lock:
            _search_cache[cache_key] = (time.time(), combined)
            _search_cache.move_to_end(cache_key)
            while len(_search_cache) > _cache_max_items:
                _search_cache.popitem(last=False)
        return combined
