from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    tokenize = 'unicode61'
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

CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model_name TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS file_metadata (
    project_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (project_id, file_path),
    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS file_chunks (
    project_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (project_id, file_path, chunk_id),
    FOREIGN KEY(project_id, file_path) REFERENCES file_metadata(project_id, file_path) ON DELETE CASCADE,
    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_file_chunks_project_file ON file_chunks(project_id, file_path);

CREATE TABLE IF NOT EXISTS internal_id_sequence (
    project_id TEXT PRIMARY KEY,
    next_internal_id INTEGER NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    error TEXT
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    name TEXT PRIMARY KEY,
    applied_at TEXT DEFAULT (datetime('now'))
);
"""


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    internal_id: int
    content: str
    token_count: int
    metadata: Dict[str, Any]
    access_count: int


@dataclass(frozen=True)
class FileMetadataRow:
    file_path: str
    mtime_ns: int
    size_bytes: int
    content_hash: str
    chunk_ids: List[str]


@dataclass(frozen=True)
class SQLQuery:
    text: str
    params: Tuple[Any, ...]


def build_in_query(prefix_sql: str, values: Sequence[Any], suffix_sql: str = "") -> SQLQuery:
    placeholders = ",".join(["?"] * len(values))
    sql = prefix_sql + "(" + placeholders + ")" + suffix_sql
    return SQLQuery(sql, tuple(values))


class _ConnectionPool:
    def __init__(self, db_path: str, maxsize: int = 10, timeout_s: float = 30.0) -> None:
        self.db_path = db_path
        self.maxsize = maxsize
        self.timeout_s = timeout_s
        self._queue: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize)
        self._created = 0
        self._lock = asyncio.Lock()
        self._all: set[aiosqlite.Connection] = set()
        self._semaphore = asyncio.Semaphore(maxsize)
        self._closing = False

    async def acquire(self) -> aiosqlite.Connection:
        if self._closing:
            raise RuntimeError("Connection pool is closing")
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.timeout_s)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("Timed out waiting for database connection") from exc

        try:
            conn = self._queue.get_nowait()
            return conn
        except asyncio.QueueEmpty:
            async with self._lock:
                if self._created < self.maxsize:
                    conn = await aiosqlite.connect(self.db_path)
                    await conn.execute("PRAGMA foreign_keys=ON;")
                    await conn.execute("PRAGMA journal_mode=WAL;")
                    self._created += 1
                    self._all.add(conn)
                    return conn
            conn = await self._queue.get()
            return conn

    async def release(self, conn: aiosqlite.Connection) -> None:
        await self._queue.put(conn)
        self._semaphore.release()

    async def close(self) -> None:
        self._closing = True
        for _ in range(self.maxsize):
            await self._semaphore.acquire()
        conns = list(self._all)
        self._all.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        for conn in conns:
            await conn.close()


_pool: Optional[_ConnectionPool] = None
_pool_lock = asyncio.Lock()


async def _get_pool(db_path: str) -> _ConnectionPool:
    global _pool
    async with _pool_lock:
        if _pool is None or _pool.db_path != db_path:
            _pool = _ConnectionPool(db_path=db_path, maxsize=10)
        return _pool


@asynccontextmanager
async def get_connection(db_path: str) -> Iterable[aiosqlite.Connection]:
    pool = await _get_pool(db_path)
    conn = await pool.acquire()
    try:
        yield conn
    finally:
        await pool.release(conn)


async def init_db(db_path: str) -> None:
    async with get_connection(db_path) as db:
        await db.executescript(SCHEMA_SQL)
        await _ensure_file_metadata_schema(db)
        await _ensure_file_chunks_schema(db)
        await _ensure_fts_tokenizer(db)
        await db.commit()


async def _ensure_file_metadata_schema(db: aiosqlite.Connection) -> None:
    rows = await db.execute_fetchall("PRAGMA table_info(file_metadata)")
    columns = {r[1] for r in rows}
    if "content_hash" not in columns:
        await db.execute("ALTER TABLE file_metadata ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''")


async def _ensure_file_chunks_schema(db: aiosqlite.Connection) -> None:
    row = await db.execute_fetchone(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'file_chunks'"
    )
    if row:
        return
    await db.executescript(
        """
        CREATE TABLE IF NOT EXISTS file_chunks (
            project_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            PRIMARY KEY (project_id, file_path, chunk_id),
            FOREIGN KEY(project_id, file_path) REFERENCES file_metadata(project_id, file_path) ON DELETE CASCADE,
            FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_file_chunks_project_file ON file_chunks(project_id, file_path);
        """
    )
    await _migrate_file_chunks_from_metadata(db)


async def _migrate_file_chunks_from_metadata(db: aiosqlite.Connection) -> None:
    row = await db.execute_fetchone(
        "SELECT 1 FROM schema_migrations WHERE name = ?",
        ("file_chunks_backfill",),
    )
    if row:
        return
    rows = await db.execute_fetchall("PRAGMA table_info(file_metadata)")
    columns = {r[1] for r in rows}
    if "chunk_ids" not in columns:
        return
    cursor = await db.execute("SELECT project_id, file_path, chunk_ids FROM file_metadata")
    batch: List[Tuple[str, str, str]] = []
    async for project_id, file_path, chunk_ids_json in cursor:
        try:
            chunk_ids = json.loads(chunk_ids_json or "[]")
        except Exception:
            chunk_ids = []
        for chunk_id in chunk_ids:
            batch.append((str(project_id), str(file_path), str(chunk_id)))
        if len(batch) >= 500:
            await db.executemany(
                "INSERT OR IGNORE INTO file_chunks(project_id, file_path, chunk_id) VALUES(?,?,?)",
                batch,
            )
            batch.clear()
    if batch:
        await db.executemany(
            "INSERT OR IGNORE INTO file_chunks(project_id, file_path, chunk_id) VALUES(?,?,?)",
            batch,
        )
    await db.execute(
        "INSERT OR IGNORE INTO schema_migrations(name) VALUES(?)",
        ("file_chunks_backfill",),
    )
    try:
        await db.execute("ALTER TABLE file_metadata DROP COLUMN chunk_ids")
    except aiosqlite.OperationalError:
        logging.info("SQLite does not support DROP COLUMN; leaving file_metadata.chunk_ids in place.")


async def _ensure_fts_tokenizer(db: aiosqlite.Connection) -> None:
    row = await db.execute_fetchone(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts'"
    )
    if not row or not row[0]:
        return
    sql = row[0].lower()
    if "tokenize = 'porter'" not in sql:
        return
    await db.executescript(
        """
        DROP TABLE IF EXISTS chunks_fts;
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id,
            project_id,
            content,
            tokenize = 'unicode61'
        );
        INSERT INTO chunks_fts(chunk_id, project_id, content)
        SELECT chunk_id, project_id, content FROM chunks;
        """
    )


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
    async with get_connection(db_path) as db:
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
    async with get_connection(db_path) as db:
        await db.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
        await db.commit()


async def list_projects(db_path: str) -> List[Dict[str, Any]]:
    async with get_connection(db_path) as db:
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
    async with get_connection(db_path) as db:
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

    async with get_connection(db_path) as db:
        await db.execute("BEGIN IMMEDIATE")
        row = await db.execute_fetchone(
            "SELECT next_internal_id FROM internal_id_sequence WHERE project_id = ?",
            (project_id,),
        )
        if row and row[0] is not None:
            start = int(row[0])
        else:
            row = await db.execute_fetchone(
                "SELECT COALESCE(MAX(internal_id), 0) FROM chunks WHERE project_id = ?",
                (project_id,),
            )
            start = int(row[0]) + 1
        next_id = start + count
        await db.execute(
            """
            INSERT INTO internal_id_sequence(project_id, next_internal_id)
            VALUES(?,?)
            ON CONFLICT(project_id) DO UPDATE SET
              next_internal_id = excluded.next_internal_id
            """,
            (project_id, next_id),
        )
        await db.commit()
        return list(range(start, next_id))


async def upsert_chunks(
    db_path: str,
    *,
    project_id: str,
    rows: List[Tuple[str, int, str, int, Dict[str, Any]]],
) -> None:
    """Upsert chunks.

    rows entries: (chunk_id, internal_id, content, token_count, metadata)
    """
    async with get_connection(db_path) as db:
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
    async with get_connection(db_path) as db:
        batch_size = 900
        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i:i + batch_size]
            query = build_in_query(
                "DELETE FROM chunks WHERE project_id = ? AND chunk_id IN ",
                batch,
            )
            await db.execute(query.text, (project_id, *query.params))
        await db.commit()


async def fetch_chunks_by_internal_ids(
    db_path: str, project_id: str, internal_ids: List[int]
) -> List[ChunkRow]:
    if not internal_ids:
        return []
    batch_size = 900
    row_map: Dict[int, ChunkRow] = {}
    async with get_connection(db_path) as db:
        for i in range(0, len(internal_ids), batch_size):
            batch = internal_ids[i:i + batch_size]
            query = build_in_query(
                """
                SELECT chunk_id, internal_id, content, token_count, metadata, access_count
                FROM chunks
                WHERE project_id = ? AND internal_id IN 
                """,
                batch,
            )
            rows = await db.execute_fetchall(query.text, (project_id, *query.params))
            for r in rows:
                meta = json.loads(r[4] or "{}")
                row_map[int(r[1])] = ChunkRow(r[0], int(r[1]), r[2], int(r[3]), meta, int(r[5] or 0))
    return [row_map[iid] for iid in internal_ids if iid in row_map]


async def bump_access_counts(db_path: str, project_id: str, chunk_ids: List[str]) -> None:
    if not chunk_ids:
        return
    query = build_in_query(
        "UPDATE chunks SET access_count = access_count + 1 WHERE project_id = ? AND chunk_id IN ",
        chunk_ids,
    )
    async with get_connection(db_path) as db:
        await db.execute(query.text, (project_id, *query.params))
        await db.commit()


async def refresh_project_stats(db_path: str, project_id: str) -> None:
    async with get_connection(db_path) as db:
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


def _sanitize_fts_query(query: str) -> str:
    if not query or not query.strip():
        return '""'
    escaped = query.replace('"', '""')
    return escaped


async def fts_search(
    db_path: str,
    *,
    project_id: str,
    query: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """Lexical search using FTS5 + bm25 ranking."""
    sanitized = _sanitize_fts_query(query)
    try:
        async with get_connection(db_path) as db:
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
                (project_id, sanitized, limit),
            )
    except aiosqlite.OperationalError:
        logging.warning("FTS query failed for %s", project_id, exc_info=True)
        return []

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


async def fetch_internal_ids_for_chunk_ids(
    db_path: str, project_id: str, chunk_ids: List[str]
) -> List[int]:
    if not chunk_ids:
        return []
    batch_size = 900
    out: List[int] = []
    async with get_connection(db_path) as db:
        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i:i + batch_size]
            query = build_in_query(
                """
                SELECT internal_id
                FROM chunks
                WHERE project_id = ? AND chunk_id IN 
                """,
                batch,
            )
            rows = await db.execute_fetchall(query.text, (project_id, *query.params))
            out.extend(int(r[0]) for r in rows)
    return out


async def fetch_file_metadata(db_path: str, project_id: str) -> Dict[str, FileMetadataRow]:
    async with get_connection(db_path) as db:
        rows = await db.execute_fetchall(
            """
            SELECT file_path, mtime_ns, size_bytes, content_hash
            FROM file_metadata
            WHERE project_id = ?
            """,
            (project_id,),
        )
    out: Dict[str, FileMetadataRow] = {}
    for path, mtime_ns, size_bytes, content_hash in rows:
        out[path] = FileMetadataRow(
            file_path=path,
            mtime_ns=int(mtime_ns),
            size_bytes=int(size_bytes),
            content_hash=str(content_hash or ""),
            chunk_ids=[],
        )
    return out


async def fetch_file_chunk_ids(
    db_path: str,
    project_id: str,
    file_paths: List[str],
) -> Dict[str, List[str]]:
    if not file_paths:
        return {}
    out: Dict[str, List[str]] = {path: [] for path in file_paths}
    batch_size = 900
    async with get_connection(db_path) as db:
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            query = build_in_query(
                """
                SELECT file_path, chunk_id
                FROM file_chunks
                WHERE project_id = ? AND file_path IN 
                """,
                batch,
            )
            rows = await db.execute_fetchall(query.text, (project_id, *query.params))
            for file_path, chunk_id in rows:
                out[str(file_path)].append(str(chunk_id))
    return out


async def upsert_file_metadata(
    db_path: str,
    project_id: str,
    rows: List[FileMetadataRow],
) -> None:
    if not rows:
        return
    async with get_connection(db_path) as db:
        columns = await db.execute_fetchall("PRAGMA table_info(file_metadata)")
        has_chunk_ids = any(col[1] == "chunk_ids" for col in columns)
        if has_chunk_ids:
            await db.executemany(
                """
                INSERT INTO file_metadata(project_id, file_path, mtime_ns, size_bytes, content_hash, chunk_ids)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(project_id, file_path) DO UPDATE SET
                  mtime_ns = excluded.mtime_ns,
                  size_bytes = excluded.size_bytes,
                  content_hash = excluded.content_hash,
                  chunk_ids = excluded.chunk_ids
                """,
                [
                    (
                        project_id,
                        r.file_path,
                        int(r.mtime_ns),
                        int(r.size_bytes),
                        r.content_hash,
                        "[]",
                    )
                    for r in rows
                ],
            )
        else:
            await db.executemany(
                """
                INSERT INTO file_metadata(project_id, file_path, mtime_ns, size_bytes, content_hash)
                VALUES(?,?,?,?,?)
                ON CONFLICT(project_id, file_path) DO UPDATE SET
                  mtime_ns = excluded.mtime_ns,
                  size_bytes = excluded.size_bytes,
                  content_hash = excluded.content_hash
                """,
                [
                    (
                        project_id,
                        r.file_path,
                        int(r.mtime_ns),
                        int(r.size_bytes),
                        r.content_hash,
                    )
                    for r in rows
                ],
            )
        file_paths = [r.file_path for r in rows]
        batch_size = 900
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            query = build_in_query(
                "DELETE FROM file_chunks WHERE project_id = ? AND file_path IN ",
                batch,
            )
            await db.execute(query.text, (project_id, *query.params))
        chunk_rows: List[Tuple[str, str, str]] = []
        for r in rows:
            for chunk_id in r.chunk_ids:
                chunk_rows.append((project_id, r.file_path, chunk_id))
        if chunk_rows:
            await db.executemany(
                "INSERT OR IGNORE INTO file_chunks(project_id, file_path, chunk_id) VALUES(?,?,?)",
                chunk_rows,
            )
        await db.commit()


async def delete_file_metadata(db_path: str, project_id: str, file_paths: List[str]) -> None:
    if not file_paths:
        return
    async with get_connection(db_path) as db:
        batch_size = 900
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            query = build_in_query(
                "DELETE FROM file_metadata WHERE project_id = ? AND file_path IN ",
                batch,
            )
            await db.execute(query.text, (project_id, *query.params))
        await db.commit()


async def fetch_embedding_cache(
    db_path: str,
    *,
    model_name: str,
    content_hashes: List[str],
) -> Dict[str, bytes]:
    if not content_hashes:
        return {}
    out: Dict[str, bytes] = {}
    batch_size = 900
    async with get_connection(db_path) as db:
        for i in range(0, len(content_hashes), batch_size):
            batch = content_hashes[i:i + batch_size]
            query = build_in_query(
                """
                    SELECT content_hash, embedding
                    FROM embedding_cache
                    WHERE model_name = ? AND content_hash IN 
                """,
                batch,
            )
            rows = await db.execute_fetchall(query.text, (model_name, *query.params))
            out.update({str(r[0]): bytes(r[1]) for r in rows})
    return out


async def mark_stale_jobs_failed(db_path: str, *, error: str) -> None:
    async with get_connection(db_path) as db:
        await db.execute(
            """
            UPDATE jobs
            SET status = 'FAILED',
                error = ?,
                updated_at = datetime('now')
            WHERE status IN ('QUEUED', 'PROCESSING')
            """,
            (error,),
        )
        await db.commit()


async def upsert_embedding_cache(
    db_path: str,
    *,
    model_name: str,
    rows: List[Tuple[str, bytes]],
    ttl_s: int = 0,
) -> None:
    if not rows:
        return
    async with get_connection(db_path) as db:
        await db.executemany(
            """
            INSERT INTO embedding_cache(content_hash, embedding, model_name, created_at)
            VALUES(?,?,?, datetime('now'))
            ON CONFLICT(content_hash) DO UPDATE SET
              embedding = excluded.embedding,
              model_name = excluded.model_name,
              created_at = datetime('now')
            """,
            [(content_hash, embedding, model_name) for content_hash, embedding in rows],
        )
        if int(ttl_s) > 0:
            await db.execute(
                """
                DELETE FROM embedding_cache
                WHERE model_name = ?
                  AND created_at < datetime('now', ?)
                """,
                (model_name, "-" + str(int(ttl_s)) + " seconds"),
            )
        await db.commit()


async def create_job(
    db_path: str,
    *,
    job_id: str,
    kind: str,
    status: str,
) -> None:
    async with get_connection(db_path) as db:
        await db.execute(
            """
            INSERT INTO jobs(job_id, kind, status, created_at, updated_at)
            VALUES(?,?,?, datetime('now'), datetime('now'))
            """,
            (job_id, kind, status),
        )
        await db.commit()


async def update_job_status(
    db_path: str,
    *,
    job_id: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    async with get_connection(db_path) as db:
        await db.execute(
            """
            UPDATE jobs
            SET status = ?, error = ?, updated_at = datetime('now')
            WHERE job_id = ?
            """,
            (status, error, job_id),
        )
        await db.commit()


async def list_jobs(db_path: str) -> List[Dict[str, Any]]:
    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            """
            SELECT job_id, kind, status, created_at, updated_at, error
            FROM jobs
            ORDER BY created_at DESC
            """
        )
    return [dict(row) for row in rows]


async def set_file_count(db_path: str, project_id: str, file_count: int) -> None:
    async with get_connection(db_path) as db:
        await db.execute(
            "UPDATE projects SET file_count = ?, updated_at = datetime('now') WHERE project_id = ?",
            (int(file_count), project_id),
        )
        await db.commit()


async def fetch_existing_chunk_ids(db_path: str, project_id: str) -> set[str]:
    async with get_connection(db_path) as db:
        rows = await db.execute_fetchall(
            "SELECT chunk_id FROM chunks WHERE project_id = ?",
            (project_id,),
        )
    return {r[0] for r in rows}


async def fetch_chunk_batch(
    db_path: str,
    project_id: str,
    *,
    last_internal_id: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    from .chunking import Chunk

    async with get_connection(db_path) as db:
        rows = await db.execute_fetchall(
            """
            SELECT chunk_id, internal_id, content, token_count, metadata
            FROM chunks
            WHERE project_id = ? AND internal_id > ?
            ORDER BY internal_id ASC
            LIMIT ?
            """,
            (project_id, int(last_internal_id), int(batch_size)),
        )

    out: List[Dict[str, Any]] = []
    for cid, iid, content, tcount, meta_json in rows:
        meta = {}
        try:
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
