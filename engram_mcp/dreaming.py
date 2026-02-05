from __future__ import annotations

import json
import logging
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


async def find_temporal_couplings(
    project_id: str,
    *,
    min_frequency: int = 5,
    limit: int = 100,
) -> List[Tuple[str, str, int, Dict[str, str]]]:
    """Find files that frequently change together in git history.

    This implements "Temporal Coupling Detection" - identifying files that
    statistically change in the same commits, even if there's no explicit
    import/call relationship between them.

    Args:
        project_id: The project to analyze
        min_frequency: Minimum number of co-changes to consider a coupling (default: 5)
        limit: Maximum number of couplings to return (default: 100)

    Returns:
        List of (file_a, file_b, frequency, metadata) tuples where:
        - file_a, file_b: Coupled file paths (file_a < file_b alphabetically)
        - frequency: Number of commits where both files changed
        - metadata: Additional context (recent commits, common tags, etc.)
    """
    async with dbmod.get_connection(_cfg.db_path) as db:
        # The Coupling Detector SQL
        # Finds pairs of files that changed in the same commit
        rows = await db.execute_fetchall(
            """
            SELECT
                a.file_path as file_a,
                b.file_path as file_b,
                COUNT(*) as pair_frequency
            FROM git_diffs a
            JOIN git_diffs b ON a.commit_hash = b.commit_hash
            WHERE
                a.project_id = ? AND
                b.project_id = ? AND
                a.file_path < b.file_path
            GROUP BY file_a, file_b
            HAVING pair_frequency >= ?
            ORDER BY pair_frequency DESC
            LIMIT ?
            """,
            (project_id, project_id, min_frequency, limit),
        )

    if not rows:
        logging.info("No temporal couplings found for project %s", project_id)
        return []

    couplings = []
    async with dbmod.get_connection(_cfg.db_path) as db:
        for file_a, file_b, frequency in rows:
            # Fetch metadata: find recent commits and common tags
            # Get commit hashes where both files changed
            commit_rows = await db.execute_fetchall(
                """
                SELECT DISTINCT a.commit_hash
                FROM git_diffs a
                JOIN git_diffs b ON a.commit_hash = b.commit_hash
                WHERE
                    a.project_id = ? AND
                    b.project_id = ? AND
                    a.file_path = ? AND
                    b.file_path = ?
                ORDER BY a.commit_hash DESC
                LIMIT 3
                """,
                (project_id, project_id, file_a, file_b),
            )

            recent_commits = [row[0] for row in commit_rows]

            # Get common tags for these commits
            tag_rows = await db.execute_fetchall(
                """
                SELECT tag, COUNT(*) as tag_count
                FROM git_tags
                WHERE commit_hash IN (
                    SELECT DISTINCT a.commit_hash
                    FROM git_diffs a
                    JOIN git_diffs b ON a.commit_hash = b.commit_hash
                    WHERE
                        a.project_id = ? AND
                        b.project_id = ? AND
                        a.file_path = ? AND
                        b.file_path = ?
                )
                GROUP BY tag
                ORDER BY tag_count DESC
                LIMIT 3
                """,
                (project_id, project_id, file_a, file_b),
            )

            common_tags = [tag for tag, _ in tag_rows]

            metadata = {
                "source": "temporal_history",
                "recent_commits": recent_commits,
                "common_tags": common_tags,
            }

            couplings.append((file_a, file_b, frequency, metadata))

    logging.info(
        "Found %d temporal couplings for project %s (min_frequency=%d)",
        len(couplings),
        project_id,
        min_frequency,
    )

    return couplings
