from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from typing import Dict, List, Tuple

from . import db as dbmod
from .config import load_config


_cfg = load_config()


async def identify_candidates(project_id: str) -> List[Tuple[str, str, int]]:
    """Identify co-occurring chunk pairs from recent search sessions.

    Returns a list of (chunk_id_a, chunk_id_b, count) tuples for pairs that:
    - originate from different files
    - are not already represented by an insight chunk
    - meet or exceed cfg.dream_threshold
    """
    async with dbmod.get_connection(_cfg.db_path) as db:
        rows = await db.execute_fetchall(
            """
            SELECT result_chunk_ids
            FROM search_sessions
            WHERE project_id = ?
            ORDER BY created_at DESC
            LIMIT 500
            """,
            (project_id,),
        )

    session_chunks: List[List[str]] = []
    all_chunk_ids: List[str] = []
    seen_all: set[str] = set()

    for (raw_ids,) in rows:
        if not raw_ids:
            continue
        try:
            chunk_ids = json.loads(raw_ids)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk_ids, list):
            continue
        unique: List[str] = []
        seen: set[str] = set()
        for cid in chunk_ids:
            if cid is None:
                continue
            cid_str = str(cid)
            if cid_str in seen:
                continue
            seen.add(cid_str)
            unique.append(cid_str)
            if cid_str not in seen_all:
                seen_all.add(cid_str)
                all_chunk_ids.append(cid_str)
        if len(unique) >= 2:
            session_chunks.append(unique)

    if not session_chunks:
        return []

    chunk_to_file: Dict[str, str] = {}
    batch_size = 900
    async with dbmod.get_connection(_cfg.db_path) as db:
        for i in range(0, len(all_chunk_ids), batch_size):
            batch = all_chunk_ids[i:i + batch_size]
            query = dbmod.build_in_query(
                """
                SELECT chunk_id, file_path
                FROM file_chunks
                WHERE project_id = ? AND chunk_id IN 
                """,
                batch,
            )
            rows = await db.execute_fetchall(query.text, (project_id, *query.params))
            for chunk_id, file_path in rows:
                chunk_to_file.setdefault(str(chunk_id), str(file_path))

        insight_rows = await db.execute_fetchall(
            """
            SELECT metadata
            FROM chunks
            WHERE project_id = ? AND metadata LIKE '%"type": "insight"%'
            """,
            (project_id,),
        )

    dreamed_pairs: set[Tuple[str, str]] = set()
    for (raw_meta,) in insight_rows:
        try:
            meta = json.loads(raw_meta or "{}")
        except json.JSONDecodeError:
            continue
        if meta.get("type") != "insight":
            continue
        source_chunks = meta.get("source_chunks")
        if not isinstance(source_chunks, list) or len(source_chunks) < 2:
            continue
        first = str(source_chunks[0])
        second = str(source_chunks[1])
        if not first or not second:
            continue
        dreamed_pairs.add(tuple(sorted((first, second))))

    counts: Counter[Tuple[str, str]] = Counter()
    for chunk_ids in session_chunks:
        for first, second in combinations(chunk_ids, 2):
            file_first = chunk_to_file.get(first)
            file_second = chunk_to_file.get(second)
            if not file_first or not file_second:
                continue
            if file_first == file_second:
                continue
            pair = tuple(sorted((first, second)))
            if pair in dreamed_pairs:
                continue
            counts[pair] += 1

    threshold = float(_cfg.dream_threshold)
    candidates = [
        (first, second, count)
        for (first, second), count in counts.items()
        if count >= threshold
    ]
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    return candidates
