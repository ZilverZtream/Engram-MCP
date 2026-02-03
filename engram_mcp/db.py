from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    project_type TEXT NOT NULL,
    directory TEXT,
    embedding_dim INTEGER NOT NULL,
    metadata TEXT,
    chunk_count INTEGER DEFAULT 0,
    file_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    internal_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    metadata TEXT,
    access_count INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_internal ON chunks(project_id, internal_id);
CREATE INDEX IF NOT EXISTS idx_chunks_project_access ON chunks(project_id, access_count DESC);

-- On-disk inverted index for lexical search
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id,
    project_id,
    content,
    tokenize = 'porter'
);

-- Keep FTS in sync (triggers)
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(chunk_id, project_id, content)
  VALUES (new.chunk_id, new.project_id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  DELETE FROM chunks_fts WHERE chunk_id = old.chunk_id;
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF content ON chunks BEGIN
  UPDATE chunks_fts SET content = new.content WHERE chunk_id = new.chunk_id;
END;
"""


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    internal_id: int
    content: str
    token_count: int
    metadata: Dict[str, Any]
    access_count: int


async def init_db(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA_SQL)
        await db.commit()


async def upsert_project(
    db_path: str,
    *,
    project_id: str,
    project_name: str,
    project_type: str,
    directory: Optional[str],
    embedding_dim: int,
    metadata: Dict[str, Any],
) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO projects(project_id, project_name, project_type, directory, embedding_dim, metadata, updated_at)
            VALUES(?,?,?,?,?,?, datetime('now'))
            ON CONFLICT(project_id) DO UPDATE SET
              project_name = excluded.project_name,
              project_type = excluded.project_type,
              directory = excluded.directory,
              embedding_dim = excluded.embedding_dim,
              metadata = excluded.metadata,
              updated_at = datetime('now')
            """,
            (project_id, project_name, project_type, directory, embedding_dim, json.dumps(metadata)),
        )
        await db.commit()


async def delete_project(db_path: str, project_id: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
        await db.commit()


async def list_projects(db_path: str) -> List[Dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            """
            SELECT project_id, project_name, project_type, directory, chunk_count, file_count, total_tokens, updated_at
            FROM projects
            ORDER BY updated_at DESC
            """
        )
        return [dict(r) for r in rows]


async def get_project(db_path: str, project_id: str) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        row = await db.execute_fetchone(
            """
            SELECT project_id, project_name, project_type, directory, embedding_dim, metadata, chunk_count, file_count, total_tokens, updated_at
            FROM projects
            WHERE project_id = ?
            """,
            (project_id,),
        )
        if not row:
            return None
        d = dict(row)
        d["metadata"] = json.loads(d.get("metadata") or "{}")
        return d


async def reserve_internal_ids(db_path: str, project_id: str, count: int) -> List[int]:
    """Reserve 'count' new internal IDs for a project.

    We store internal IDs explicitly so we can do stable add_with_ids() into FAISS.
    """
    if count <= 0:
        return []

    async with aiosqlite.connect(db_path) as db:
        row = await db.execute_fetchone(
            "SELECT COALESCE(MAX(internal_id), 0) FROM chunks WHERE project_id = ?",
            (project_id,),
        )
        start = int(row[0]) + 1
        return list(range(start, start + count))


async def upsert_chunks(
    db_path: str,
    *,
    project_id: str,
    rows: List[Tuple[str, int, str, int, Dict[str, Any]]],
) -> None:
    """Upsert chunks.

    rows entries: (chunk_id, internal_id, content, token_count, metadata)
    """
    async with aiosqlite.connect(db_path) as db:
        await db.executemany(
            """
            INSERT INTO chunks(chunk_id, project_id, internal_id, content, token_count, metadata, updated_at)
            VALUES(?,?,?,?,?,?, datetime('now'))
            ON CONFLICT(chunk_id) DO UPDATE SET
              content = excluded.content,
              token_count = excluded.token_count,
              metadata = excluded.metadata,
              updated_at = datetime('now')
            """,
            [
                (cid, project_id, iid, content, tcount, json.dumps(meta))
                for (cid, iid, content, tcount, meta) in rows
            ],
        )
        await db.commit()


async def delete_chunks(db_path: str, project_id: str, chunk_ids: List[str]) -> None:
    if not chunk_ids:
        return
    placeholders = ",".join(["?"] * len(chunk_ids))
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            f"DELETE FROM chunks WHERE project_id = ? AND chunk_id IN ({placeholders})",
            (project_id, *chunk_ids),
        )
        await db.commit()


async def fetch_chunks_by_internal_ids(
    db_path: str, project_id: str, internal_ids: List[int]
) -> List[ChunkRow]:
    if not internal_ids:
        return []
    placeholders = ",".join(["?"] * len(internal_ids))
    async with aiosqlite.connect(db_path) as db:
        rows = await db.execute_fetchall(
            f"""
            SELECT chunk_id, internal_id, content, token_count, metadata, access_count
            FROM chunks
            WHERE project_id = ? AND internal_id IN ({placeholders})
            """,
            (project_id, *internal_ids),
        )
    out: List[ChunkRow] = []
    for r in rows:
        meta = json.loads(r[4] or "{}")
        out.append(ChunkRow(r[0], int(r[1]), r[2], int(r[3]), meta, int(r[5] or 0)))
    return out


async def bump_access_counts(db_path: str, project_id: str, chunk_ids: List[str]) -> None:
    if not chunk_ids:
        return
    placeholders = ",".join(["?"] * len(chunk_ids))
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            f"UPDATE chunks SET access_count = access_count + 1 WHERE project_id = ? AND chunk_id IN ({placeholders})",
            (project_id, *chunk_ids),
        )
        await db.commit()


async def refresh_project_stats(db_path: str, project_id: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        row = await db.execute_fetchone(
            "SELECT COUNT(*), COALESCE(SUM(token_count), 0) FROM chunks WHERE project_id = ?",
            (project_id,),
        )
        chunk_count = int(row[0])
        total_tokens = int(row[1])
        await db.execute(
            """
            UPDATE projects
            SET chunk_count = ?, total_tokens = ?, updated_at = datetime('now')
            WHERE project_id = ?
            """,
            (chunk_count, total_tokens, project_id),
        )
        await db.commit()


async def fts_search(
    db_path: str,
    *,
    project_id: str,
    query: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """Lexical search using FTS5 + bm25 ranking."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            """
            SELECT c.chunk_id, c.internal_id, c.content, c.token_count, c.metadata, c.access_count,
                   bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts.project_id = ? AND chunks_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?
            """,
            (project_id, query, limit),
        )

    out: List[Dict[str, Any]] = []
    for r in rows:
        meta = json.loads(r[4] or "{}")
        out.append(
            {
                "id": r[0],
                "internal_id": int(r[1]),
                "content": r[2],
                "token_count": int(r[3]),
                "metadata": meta,
                "access_count": int(r[5] or 0),
                "lex_score": float(r[6]),
            }
        )
    return out
