from __future__ import annotations

import json
import logging
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from . import db as dbmod
from .config import EngramConfig, load_config
from .generation import GenerationService

# Lazily loaded module-level config.  Access via _get_cfg() so that
# unit tests and callers that supply an explicit cfg= are not forced to
# have a config file present on disk.
_cfg: Optional[EngramConfig] = None


def _get_cfg() -> EngramConfig:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg

# Style Analysis Prompt Template
STYLE_ANALYSIS_PROMPT = """You are a code style analyzer. You will be given recent git diffs for a file.
Your task is to extract the coding patterns and conventions used in this file.

Recent changes to {file_path}:

{diffs}

Analyze the above changes and extract:
1. Naming conventions (e.g., snake_case, camelCase, PascalCase)
2. Common patterns (e.g., validation before DB calls, specific libraries/frameworks used)
3. Code organization patterns (e.g., class structure, function ordering)
4. Error handling approaches (e.g., try/except, return None, raise exceptions)
5. Import style (e.g., from X import Y vs import X.Y)
6. Documentation style (e.g., docstrings, inline comments)

Format your response as a concise style guide (3-5 bullet points) that could be prepended to an AI agent's context.
Focus on actionable patterns, not generic advice.

If there are insufficient changes to determine a clear pattern, respond with "INSUFFICIENT_DATA".

Style Guide:
"""


async def identify_candidates(
    project_id: str,
    *,
    cfg: Optional[EngramConfig] = None,
) -> List[Tuple[str, str, int]]:
    """Identify co-occurring chunk pairs from recent search sessions.

    Returns a list of (chunk_id_a, chunk_id_b, count) tuples for pairs that:
    - originate from different files
    - are not already represented by an insight chunk
    - meet or exceed cfg.dream_threshold
    """
    cfg = cfg or _get_cfg()
    async with dbmod.get_connection(cfg.db_path) as db:
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
    async with dbmod.get_connection(cfg.db_path) as db:
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

    threshold = float(cfg.dream_threshold)
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
    recent_commits_limit: int = 2000,
    cfg: Optional[EngramConfig] = None,
) -> List[Tuple[str, str, int, Dict[str, str]]]:
    """Find files that frequently change together in git history.

    This implements "Temporal Coupling Detection" - identifying files that
    statistically change in the same commits, even if there's no explicit
    import/call relationship between them.

    Args:
        project_id: The project to analyze
        min_frequency: Minimum number of co-changes to consider a coupling (default: 5)
        limit: Maximum number of couplings to return (default: 100)
        recent_commits_limit: Only analyze recent N commits to prevent full history scan (default: 2000)
        cfg: Optional EngramConfig instance; loaded lazily if not provided.

    Returns:
        List of (file_a, file_b, frequency, metadata) tuples where:
        - file_a, file_b: Coupled file paths (file_a < file_b alphabetically)
        - frequency: Number of commits where both files changed
        - metadata: Additional context (recent commits, common tags, etc.)
    """
    cfg = cfg or _get_cfg()
    async with dbmod.get_connection(cfg.db_path) as db:
        # The Coupling Detector SQL with optimization:
        # 1. Use CTE to limit to recent commits (prevents full history scan)
        # 2. Finds pairs of files that changed in the same commit
        rows = await db.execute_fetchall(
            """
            WITH recent_commits AS (
                SELECT commit_hash
                FROM git_commits
                WHERE project_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            )
            SELECT
                a.file_path as file_a,
                b.file_path as file_b,
                COUNT(*) as pair_frequency
            FROM git_diffs a
            JOIN git_diffs b ON a.commit_hash = b.commit_hash
            WHERE
                a.project_id = ? AND
                b.project_id = ? AND
                a.file_path < b.file_path AND
                a.commit_hash IN (SELECT commit_hash FROM recent_commits)
            GROUP BY file_a, file_b
            HAVING pair_frequency >= ?
            ORDER BY pair_frequency DESC
            LIMIT ?
            """,
            (project_id, recent_commits_limit, project_id, project_id, min_frequency, limit),
        )

    if not rows:
        logging.info("No temporal couplings found for project %s", project_id)
        return []

    # Batch metadata retrieval - fixes N+1 query pattern
    # Step 1: Collect all file pairs
    file_pairs = [(file_a, file_b, freq) for file_a, file_b, freq in rows]

    # Step 2: Use a single parameterised query with a JSON-array parameter
    # to fetch recent commits for ALL pairs at once.  This replaces the
    # previous UNION ALL approach whose query length grew O(pairs) and could
    # exceed SQLite's SQLITE_MAX_COMPOUND_SELECT limit.
    pair_commits: Dict[Tuple[str, str], List[str]] = {}

    async with dbmod.get_connection(cfg.db_path) as db:
        import json as _json

        # Encode all pairs as a JSON array: [[file_a, file_b], ...]
        pairs_json = _json.dumps([[fa, fb] for fa, fb, _ in file_pairs])

        commit_rows = await db.execute_fetchall(
            """
            WITH pair_list(file_a, file_b) AS (
                SELECT
                    value ->> 0,
                    value ->> 1
                FROM json_each(?)
            ),
            matched AS (
                SELECT
                    p.file_a,
                    p.file_b,
                    a.commit_hash,
                    c.timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.file_a, p.file_b
                        ORDER BY c.timestamp DESC
                    ) AS rn
                FROM pair_list p
                JOIN git_diffs a ON a.project_id = ? AND a.file_path = p.file_a
                JOIN git_diffs b ON b.commit_hash = a.commit_hash
                                AND b.project_id = ?
                                AND b.file_path = p.file_b
                JOIN git_commits c ON c.commit_hash = a.commit_hash
            )
            SELECT file_a, file_b, commit_hash, timestamp, rn
            FROM matched
            WHERE rn <= 3
            """,
            (pairs_json, project_id, project_id),
        )

        # Map commits to file pairs (keep only top 3 per pair)
        for file_a, file_b, commit_hash, _, rn in commit_rows:
            key = (file_a, file_b)
            if key not in pair_commits:
                pair_commits[key] = []
            pair_commits[key].append(commit_hash)

        # Step 3: Build batch query for tags
        # Collect all unique commit hashes
        all_commit_hashes = set()
        for commits in pair_commits.values():
            all_commit_hashes.update(commits)

        if all_commit_hashes:
            # Fetch all tags for all commits at once
            tag_query = dbmod.build_in_query(
                "SELECT commit_hash, tag FROM git_tags WHERE commit_hash IN ",
                list(all_commit_hashes),
            )
            tag_rows = await db.execute_fetchall(tag_query.text, tag_query.params)

            # Map tags to commits
            commit_tags: Dict[str, List[str]] = {}
            for commit_hash, tag in tag_rows:
                if commit_hash not in commit_tags:
                    commit_tags[commit_hash] = []
                commit_tags[commit_hash].append(tag)
        else:
            commit_tags = {}

    # Step 4: Build results by mapping metadata to each coupling
    couplings = []
    for file_a, file_b, frequency in file_pairs:
        key = (file_a, file_b)
        recent_commits = pair_commits.get(key, [])

        # Collect common tags from recent commits
        tag_counter: Counter[str] = Counter()
        for commit_hash in recent_commits:
            tags = commit_tags.get(commit_hash, [])
            tag_counter.update(tags)

        # Get top 3 most common tags
        common_tags = [tag for tag, _ in tag_counter.most_common(3)]

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


async def analyze_file_style(
    project_id: str,
    file_path: str,
    *,
    diff_limit: int = 10,
    max_tokens: int = 512,
    generation_service: Optional[GenerationService] = None,
    cfg: Optional[EngramConfig] = None,
) -> Dict[str, any]:
    """Analyze coding style and patterns from recent git diffs for a file.

    This implements "Style Mimicry" - extracting coding conventions from git history
    so that AI agents can generate code that matches the existing style.

    Args:
        project_id: The project to analyze
        file_path: Path to the file to analyze
        diff_limit: Number of recent diffs to analyze (default: 10)
        max_tokens: Maximum tokens for LLM generation (default: 512)
        generation_service: Optional GenerationService instance (creates one if not provided)
        cfg: Optional EngramConfig instance; loaded lazily if not provided.

    Returns:
        Dictionary containing:
        - style_guide: String with extracted style patterns (or None if insufficient data)
        - analyzed_commits: List of commit hashes that were analyzed
        - file_path: The analyzed file path
    """
    cfg = cfg or _get_cfg()
    async with dbmod.get_connection(cfg.db_path) as db:
        # Fetch recent diffs for this file
        diff_rows = await db.execute_fetchall(
            """
            SELECT d.commit_hash, d.diff_content, c.timestamp, c.message
            FROM git_diffs d
            JOIN git_commits c ON d.commit_hash = c.commit_hash
            WHERE d.project_id = ? AND d.file_path = ?
            ORDER BY c.timestamp DESC
            LIMIT ?
            """,
            (project_id, file_path, diff_limit),
        )

    if not diff_rows:
        logging.info("No diffs found for file %s in project %s", file_path, project_id)
        return {
            "style_guide": None,
            "analyzed_commits": [],
            "file_path": file_path,
            "error": "No git history found for this file",
        }

    # Format diffs into a readable format
    diffs_text = []
    analyzed_commits = []
    for commit_hash, diff_content, timestamp, message in diff_rows:
        analyzed_commits.append(commit_hash)
        # Truncate very long diffs
        truncated_diff = diff_content[:2000] if diff_content else ""
        if len(diff_content or "") > 2000:
            truncated_diff += "\n... (truncated)"

        diffs_text.append(
            f"Commit: {commit_hash[:8]}\n"
            f"Message: {message}\n"
            f"Diff:\n{truncated_diff}\n"
            f"{'-' * 40}\n"
        )

    # Combine all diffs
    combined_diffs = "\n".join(diffs_text)

    # Build the style analysis prompt
    prompt = STYLE_ANALYSIS_PROMPT.format(
        file_path=file_path,
        diffs=combined_diffs,
    )

    # Generate style guide using LLM
    owns_service = False
    if generation_service is None:
        generation_service = GenerationService()
        owns_service = True

    try:
        import os

        model_name = os.getenv("ENGRAM_GENERATION_MODEL", "gpt2")
        device = os.getenv("ENGRAM_GENERATION_DEVICE", "cpu")

        style_guide = await generation_service.generate(
            prompt,
            model_name=model_name,
            device=device,
            max_new_tokens=max_tokens,
            temperature=0.3,  # Lower temperature for more focused analysis
            top_p=0.9,
        )

        # Check if insufficient data
        if "INSUFFICIENT_DATA" in style_guide:
            logging.info("Insufficient data to analyze style for %s", file_path)
            return {
                "style_guide": None,
                "analyzed_commits": analyzed_commits,
                "file_path": file_path,
                "error": "Insufficient data to determine style patterns",
            }

        logging.info("Generated style guide for %s from %d commits", file_path, len(analyzed_commits))

        return {
            "style_guide": style_guide.strip(),
            "analyzed_commits": analyzed_commits,
            "file_path": file_path,
        }

    finally:
        if owns_service:
            await generation_service.close()


async def get_coding_style(
    project_id: str,
    file_path: str,
    *,
    generation_service: Optional[GenerationService] = None,
    cfg: Optional[EngramConfig] = None,
) -> Optional[str]:
    """Get a coding style guide for a file (simplified API).

    This is a convenience wrapper around analyze_file_style that just returns
    the style guide string (or None if unavailable).

    Args:
        project_id: The project to analyze
        file_path: Path to the file to analyze
        generation_service: Optional GenerationService instance

    Returns:
        String containing the style guide, or None if unavailable
    """
    result = await analyze_file_style(
        project_id,
        file_path,
        generation_service=generation_service,
        cfg=cfg,
    )
    return result.get("style_guide")
