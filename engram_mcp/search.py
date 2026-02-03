from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from . import db as dbmod


def rrf_combine(bm25_results: List[Dict[str, Any]], vec_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion."""
    rank_scores: Dict[str, float] = {}
    all_results: Dict[str, Dict[str, Any]] = {}

    for rank, r in enumerate(bm25_results, 1):
        cid = r["id"]
        rank_scores[cid] = rank_scores.get(cid, 0.0) + (1.0 / (k + rank))
        all_results[cid] = r
        r["lex_rank"] = rank

    for rank, r in enumerate(vec_results, 1):
        cid = r["id"]
        rank_scores[cid] = rank_scores.get(cid, 0.0) + (1.0 / (k + rank))
        if cid not in all_results:
            all_results[cid] = r
        r["vec_rank"] = rank

    combined = []
    for cid, score in sorted(rank_scores.items(), key=lambda x: x[1], reverse=True):
        out = dict(all_results[cid])
        out["relevance_score"] = float(score)
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

    selected: List[int] = []
    remaining = list(range(len(valid)))

    first = int(np.argmax(rel))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        R = E[remaining]
        S = E[selected]
        sim = R @ S.T
        max_sim = sim.max(axis=1)
        mmr = mmr_lambda * rel[remaining] - (1 - mmr_lambda) * max_sim
        best_pos = int(np.argmax(mmr))
        best = remaining[best_pos]
        selected.append(best)
        remaining.remove(best)

    return [valid[i] for i in selected]


@dataclass
class SearchEngine:
    db_path: str
    index_dir: str

    async def _load_faiss(self, project_id: str):
        import faiss

        index_path = os.path.join(self.index_dir, f"{project_id}.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        # Memory-map for faster load on large indices
        return faiss.read_index(index_path, faiss.IO_FLAG_MMAP)

    async def vector_search(
        self,
        *,
        project_id: str,
        query_vec: np.ndarray,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Vector search against FAISS index (runs in a thread to keep event loop responsive)."""
        import faiss

        loop = asyncio.get_running_loop()
        index = await self._load_faiss(project_id)

        def _search_sync() -> List[int]:
            q = np.asarray([query_vec], dtype=np.float32)
            faiss.normalize_L2(q)
            D, I = index.search(q, top_k)
            return [int(i) for i in I[0] if int(i) != -1]

        internal_ids: List[int] = await loop.run_in_executor(None, _search_sync)
        rows = await dbmod.fetch_chunks_by_internal_ids(self.db_path, project_id, internal_ids)

        # Map internal_id -> score rank (distance unavailable in this abstraction; rank is ok for RRF)
        id_to_rank = {iid: r for r, iid in enumerate(internal_ids, 1)}

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
                }
            )
        # Preserve FAISS rank order
        out.sort(key=lambda r: r["vec_rank"])
        return out

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
        lex = await dbmod.fts_search(self.db_path, project_id=project_id, query=query, limit=fts_top_k)
        vec = await self.vector_search(project_id=project_id, query_vec=query_vec, top_k=vector_top_k)

        combined = rrf_combine(lex, vec, k=60)
        combined = combined[: max(return_k, 50)]

        if enable_mmr:
            combined = apply_mmr(combined, query_vec=query_vec, k=return_k, mmr_lambda=mmr_lambda)
        else:
            combined = combined[:return_k]

        # bump access counts best-effort
        await dbmod.bump_access_counts(self.db_path, project_id, [r["id"] for r in combined])
        return combined
