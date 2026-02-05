from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import aiosqlite

# Polyfill: aiosqlite >= 0.22 removed execute_fetchone.  Re-add it so that
# the rest of the module can use the convenient one-liner without changing
# every call site.
if not hasattr(aiosqlite.Connection, "execute_fetchone"):

    async def _execute_fetchone(self, sql: str, parameters: tuple = ()) -> Any:  # type: ignore[override]
        async with self.execute(sql, parameters) as cursor:
            return await cursor.fetchone()

    aiosqlite.Connection.execute_fetchone = _execute_fetchone  # type: ignore[attr-defined]


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    project_type TEXT NOT NULL,
    directory TEXT,
    directory_realpath TEXT,
    embedding_dim INTEGER NOT NULL,
    metadata TEXT,
    index_dirty INTEGER NOT NULL DEFAULT 0,
    faiss_index_uuid TEXT,
    deleting INTEGER NOT NULL DEFAULT 0,
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
    vector BLOB,
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
  INSERT INTO chunks_fts(rowid, chunk_id, project_id, content)
  VALUES (new.rowid, new.chunk_id, new.project_id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  DELETE FROM chunks_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF content ON chunks BEGIN
  UPDATE chunks_fts SET content = new.content WHERE rowid = new.rowid;
END;

CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model_name TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_created_at ON embedding_cache(created_at);

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

CREATE TABLE IF NOT EXISTS project_artifacts (
    project_id TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (project_id, artifact_path),
    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS search_sessions (
    session_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    query TEXT NOT NULL,
    result_chunk_ids TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_search_sessions_project_created_at
ON search_sessions(project_id, created_at DESC);
"""

MAX_RESERVE_COUNT = 1_000_000


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


def _safe_load_metadata(raw: Optional[str], *, context: str) -> Dict[str, Any]:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        logging.warning("Failed to decode metadata for %s", context, exc_info=True)
        return {}


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

        # Everything after semaphore acquisition is wrapped so that any
        # failure releases the semaphore.  Without this guard a failed
        # aiosqlite.connect() (permission error, disk full, locked file)
        # permanently burns one semaphore slot; after maxsize failures the
        # pool deadlocks.
        try:
            try:
                conn = self._queue.get_nowait()
                return conn
            except asyncio.QueueEmpty:
                should_create = False
                async with self._lock:
                    if self._created < self.maxsize:
                        self._created += 1
                        should_create = True
                if should_create:
                    try:
                        conn = await aiosqlite.connect(self.db_path, isolation_level=None)
                        await conn.execute("PRAGMA foreign_keys=ON;")
                        await conn.execute("PRAGMA journal_mode=WAL;")
                        self._all.add(conn)
                        return conn
                    except Exception:
                        async with self._lock:
                            self._created -= 1
                        raise
                conn = await self._queue.get()
                return conn
        except BaseException:
            self._semaphore.release()
            raise

    async def release(self, conn: aiosqlite.Connection) -> None:
        # The semaphore is released unconditionally in `finally` so that a
        # failed rollback or a failed queue.put() cannot leak a slot.
        try:
            if conn.in_transaction:
                await conn.rollback()
            await self._queue.put(conn)
        except Exception:
            logging.warning("Failed to rollback or return pooled connection; closing.", exc_info=True)
            try:
                await conn.close()
            except Exception:
                logging.warning("Failed to close connection during release", exc_info=True)
            self._all.discard(conn)
            if self._created > 0:
                self._created -= 1
        finally:
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


_pools: Dict[str, _ConnectionPool] = {}
_pool_lock = asyncio.Lock()


async def _get_pool(db_path: str) -> _ConnectionPool:
    ensure_db_permissions(db_path)
    async with _pool_lock:
        pool = _pools.get(db_path)
        if pool is None:
            pool = _ConnectionPool(db_path=db_path, maxsize=10)
            _pools[db_path] = pool
        return pool


def ensure_db_permissions(db_path: str) -> None:
    db_path = os.path.abspath(db_path)
    db_dir = os.path.dirname(db_path)
    os.makedirs(db_dir, exist_ok=True)
    if not os.path.exists(db_path):
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(db_path, flags, 0o600)
            os.close(fd)
        except FileExistsError:
            pass
        except OSError:
            logging.warning("Failed to create database file %s securely.", db_path, exc_info=True)
    if os.name != "nt":
        try:
            os.chmod(db_path, 0o600)
        except OSError:
            logging.warning("Failed to chmod database file %s", db_path, exc_info=True)


async def close_db_pool(db_path: Optional[str] = None) -> None:
    async with _pool_lock:
        if db_path is None:
            pools = list(_pools.values())
            _pools.clear()
        else:
            pool = _pools.pop(db_path, None)
            pools = [pool] if pool else []
    for pool in pools:
        await pool.close()


@asynccontextmanager
async def get_connection(db_path: str) -> Iterable[aiosqlite.Connection]:
    pool = await _get_pool(db_path)
    conn = await pool.acquire()
    try:
        yield conn
    finally:
        await pool.release(conn)


async def init_db(db_path: str) -> None:
    ensure_db_permissions(db_path)
    async with get_connection(db_path) as db:
        await db.executescript(SCHEMA_SQL)
        await _ensure_projects_schema(db)
        await _ensure_file_metadata_schema(db)
        await _ensure_file_chunks_schema(db)
        await _ensure_chunks_vector_column(db)
        await _ensure_fts_tokenizer(db)
        await _ensure_fts_triggers(db)
        await _ensure_search_sessions_schema(db)
        await _ensure_graph_schema(db)
        await _ensure_repo_rules_schema(db)
        await db.commit()

async def _ensure_chunks_vector_column(db: aiosqlite.Connection) -> None:
    """Add the vector BLOB column used for crash-recovery without re-embedding."""
    rows = await db.execute_fetchall("PRAGMA table_info(chunks)")
    columns = {r[1] for r in rows}
    if "vector" not in columns:
        await db.execute("ALTER TABLE chunks ADD COLUMN vector BLOB")


async def _ensure_projects_schema(db: aiosqlite.Connection) -> None:
    rows = await db.execute_fetchall("PRAGMA table_info(projects)")
    columns = {r[1] for r in rows}
    if "directory_realpath" not in columns:
        await db.execute("ALTER TABLE projects ADD COLUMN directory_realpath TEXT")
    if "index_dirty" not in columns:
        await db.execute("ALTER TABLE projects ADD COLUMN index_dirty INTEGER NOT NULL DEFAULT 0")
    if "faiss_index_uuid" not in columns:
        await db.execute("ALTER TABLE projects ADD COLUMN faiss_index_uuid TEXT")
    if "deleting" not in columns:
        await db.execute("ALTER TABLE projects ADD COLUMN deleting INTEGER NOT NULL DEFAULT 0")


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


def _iter_json_string_array(json_text: str) -> "Iterable[str]":
    """Yield string elements from a JSON array without materialising the full list.

    Avoids allocating a Python list when the array may contain hundreds of
    thousands of elements (e.g. chunk-id lists for large files).  Falls back
    to the standard decoder for each element so escape sequences are handled
    correctly.

    Robustness: the opening ``[`` is located by stripping leading whitespace
    rather than scanning for the first ``[`` in the raw text.  The previous
    scanner could mis-identify a ``[`` that appeared inside a leading string
    or comment; ``json.loads`` would reject such input, so we raise
    ``ValueError`` early to surface data-corruption rather than silently
    yielding wrong results.
    """
    decoder = json.JSONDecoder()
    stripped = json_text.lstrip()
    if not stripped or stripped[0] != '[':
        # Not a valid JSON array — surface the problem so callers can detect
        # data corruption rather than silently returning an empty sequence.
        raise ValueError(
            f"Expected JSON array; got {stripped[:32]!r}… "
            f"(first non-whitespace char is {stripped[0]!r})" if stripped
            else "Expected JSON array; got empty string"
        )
    # idx into the *original* string, pointing just past '['
    idx = len(json_text) - len(stripped) + 1
    length = len(json_text)
    while idx < length:
        # Skip whitespace and commas.
        while idx < length and json_text[idx] in ' \t\n\r,':
            idx += 1
        if idx >= length or json_text[idx] == ']':
            break
        try:
            obj, end = decoder.raw_decode(json_text, idx)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON array near index {idx}: {json_text[idx:idx + 32]!r}…"
            ) from exc
        yield str(obj)
        idx = end


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
    migration_failed = False
    async for project_id, file_path, chunk_ids_json in cursor:
        # Stream the JSON array element-by-element so that a file with
        # 100 000+ chunk IDs does not materialise a 100 000-element Python
        # list all at once.  Flush to the DB every 500 rows to keep peak
        # memory bounded regardless of per-file chunk count.
        try:
            for chunk_id in _iter_json_string_array(chunk_ids_json or "[]"):
                batch.append((str(project_id), str(file_path), chunk_id))
                if len(batch) >= 500:
                    await db.executemany(
                        "INSERT OR IGNORE INTO file_chunks(project_id, file_path, chunk_id) VALUES(?,?,?)",
                        batch,
                    )
                    batch.clear()
        except ValueError as exc:
            logging.error(
                "Aborting file_chunks_backfill: malformed chunk_ids JSON for %s %s",
                project_id,
                file_path,
            )
            logging.error("Migration error details: %s", exc)
            migration_failed = True
            break
    if migration_failed:
        return
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
        INSERT INTO chunks_fts(rowid, chunk_id, project_id, content)
        SELECT rowid, chunk_id, project_id, content FROM chunks;
        """
    )


async def _ensure_fts_triggers(db: aiosqlite.Connection) -> None:
    await db.executescript(
        """
        DROP TRIGGER IF EXISTS chunks_ai;
        DROP TRIGGER IF EXISTS chunks_ad;
        DROP TRIGGER IF EXISTS chunks_au;
        CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
          INSERT INTO chunks_fts(rowid, chunk_id, project_id, content)
          VALUES (new.rowid, new.chunk_id, new.project_id, new.content);
        END;
        CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
          DELETE FROM chunks_fts WHERE rowid = old.rowid;
        END;
        CREATE TRIGGER chunks_au AFTER UPDATE OF content ON chunks BEGIN
          UPDATE chunks_fts SET content = new.content WHERE rowid = new.rowid;
        END;
        """
    )
    await db.execute("DELETE FROM chunks_fts;")
    await db.execute(
        "INSERT INTO chunks_fts(rowid, chunk_id, project_id, content) "
        "SELECT rowid, chunk_id, project_id, content FROM chunks;"
    )


async def _ensure_search_sessions_schema(db: aiosqlite.Connection) -> None:
    row = await db.execute_fetchone(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'search_sessions'"
    )
    if not row:
        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS search_sessions (
                session_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                query TEXT NOT NULL,
                result_chunk_ids TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_search_sessions_project_created_at
            ON search_sessions(project_id, created_at DESC);
            """
        )
        return
    await db.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_search_sessions_project_created_at
        ON search_sessions(project_id, created_at DESC)
        """
    )
    columns = await db.execute_fetchall("PRAGMA table_info(search_sessions)")
    column_names = {str(row[1]) for row in columns}
    if "result_chunk_ids" not in column_names:
        await db.execute("ALTER TABLE search_sessions ADD COLUMN result_chunk_ids TEXT")


async def _ensure_graph_schema(db: aiosqlite.Connection) -> None:
    """Migration graph_v1: create graph_nodes and graph_edges tables.

    node_id is TEXT (not an integer) so that it can carry fully-qualified
    C++ signatures such as ``MyClass::method`` without loss.
    """
    row = await db.execute_fetchone(
        "SELECT 1 FROM schema_migrations WHERE name = 'graph_v1'"
    )
    if row:
        return

    await db.executescript(
        """
        CREATE TABLE IF NOT EXISTS graph_nodes (
            node_id   TEXT    NOT NULL,
            project_id TEXT   NOT NULL,
            node_type TEXT    NOT NULL,
            name      TEXT    NOT NULL,
            file_path TEXT    NOT NULL,
            start_line INTEGER NOT NULL,
            end_line   INTEGER NOT NULL,
            metadata  TEXT,
            PRIMARY KEY (project_id, node_id),
            FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_graph_nodes_project_file
            ON graph_nodes(project_id, file_path);
        CREATE INDEX IF NOT EXISTS idx_graph_nodes_project_name
            ON graph_nodes(project_id, name);

        CREATE TABLE IF NOT EXISTS graph_edges (
            source_id  TEXT NOT NULL,
            target_id  TEXT NOT NULL,
            project_id TEXT NOT NULL,
            edge_type  TEXT NOT NULL,
            PRIMARY KEY (project_id, source_id, target_id),
            FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_graph_edges_project_source
            ON graph_edges(project_id, source_id);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_project_target
            ON graph_edges(project_id, target_id);
        """
    )
    await db.execute(
        "INSERT INTO schema_migrations(name) VALUES(?)", ("graph_v1",)
    )


async def upsert_project(
    db_path: str,
    *,
    project_id: str,
    project_name: str,
    project_type: str,
    directory: Optional[str],
    directory_realpath: Optional[str],
    embedding_dim: int,
    metadata: Dict[str, Any],
    index_dirty: int = 0,
    faiss_index_uuid: Optional[str] = None,
    deleting: int = 0,
) -> None:
    async with get_connection(db_path) as db:
        await db.execute(
            """
            INSERT INTO projects(
                project_id,
                project_name,
                project_type,
                directory,
                directory_realpath,
                embedding_dim,
                metadata,
                index_dirty,
                faiss_index_uuid,
                deleting,
                updated_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?, datetime('now'))
            ON CONFLICT(project_id) DO UPDATE SET
              project_name = excluded.project_name,
              project_type = excluded.project_type,
              directory = excluded.directory,
              directory_realpath = excluded.directory_realpath,
              embedding_dim = excluded.embedding_dim,
              metadata = excluded.metadata,
              index_dirty = excluded.index_dirty,
              faiss_index_uuid = excluded.faiss_index_uuid,
              deleting = excluded.deleting,
              updated_at = datetime('now')
            """,
            (
                project_id,
                project_name,
                project_type,
                directory,
                directory_realpath,
                embedding_dim,
                json.dumps(metadata),
                int(index_dirty),
                faiss_index_uuid,
                int(deleting),
            ),
        )
        await db.commit()


async def delete_project(db_path: str, project_id: str) -> None:
    async with get_connection(db_path) as db:
        await db.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
        await db.commit()


async def replace_project_artifacts(
    db_path: str,
    *,
    project_id: str,
    artifacts: Sequence[Tuple[str, str]],
) -> None:
    async with get_connection(db_path) as db:
        await db.execute("DELETE FROM project_artifacts WHERE project_id = ?", (project_id,))
        if artifacts:
            await db.executemany(
                """
                INSERT OR REPLACE INTO project_artifacts(project_id, artifact_path, kind)
                VALUES(?,?,?)
                """,
                [(project_id, path, kind) for path, kind in artifacts],
            )
        await db.commit()


async def list_project_artifacts(db_path: str, project_id: str) -> List[Dict[str, str]]:
    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            """
            SELECT artifact_path, kind
            FROM project_artifacts
            WHERE project_id = ?
            ORDER BY created_at DESC
            """,
            (project_id,),
        )
        return [dict(r) for r in rows]


async def list_all_project_artifacts(db_path: str) -> List[str]:
    async with get_connection(db_path) as db:
        rows = await db.execute_fetchall(
            "SELECT artifact_path FROM project_artifacts ORDER BY created_at DESC"
        )
        return [str(r[0]) for r in rows]


async def insert_search_session(
    db_path: str,
    *,
    project_id: str,
    query_text: str,
    result_chunk_ids: Sequence[str],
) -> None:
    async with get_connection(db_path) as db:
        session_id = uuid.uuid4().hex
        result_json = json.dumps(list(result_chunk_ids))
        await db.execute(
            """
            INSERT OR REPLACE INTO search_sessions(
                session_id,
                project_id,
                query,
                result_chunk_ids,
                created_at
            )
            VALUES(?,?,?,?, datetime('now'))
            """,
            (session_id, project_id, query_text, result_json),
        )
        await db.commit()


async def prune_search_sessions(
    db_path: str,
    *,
    project_id: Optional[str] = None,
    max_age_s: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> int:
    deleted = 0
    async with get_connection(db_path) as db:
        if max_age_s is not None:
            age_clause = "datetime('now', ?)"
            age_param = f"-{int(max_age_s)} seconds"
            if project_id:
                cursor = await db.execute(
                    "DELETE FROM search_sessions WHERE project_id = ? AND created_at < " + age_clause,
                    (project_id, age_param),
                )
            else:
                cursor = await db.execute(
                    "DELETE FROM search_sessions WHERE created_at < " + age_clause,
                    (age_param,),
                )
            deleted += cursor.rowcount or 0
        if max_rows is not None:
            max_rows = int(max_rows)
            if max_rows < 0:
                max_rows = 0
            project_clause = "WHERE project_id = ?" if project_id else ""
            params: Tuple[Any, ...] = (project_id,) if project_id else ()
            cursor = await db.execute(
                f"""
                DELETE FROM search_sessions
                WHERE session_id IN (
                    SELECT session_id
                    FROM search_sessions
                    {project_clause}
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (*params, max_rows),
            )
            deleted += cursor.rowcount or 0
        if deleted:
            await db.commit()
    return deleted


async def list_projects(db_path: str) -> List[Dict[str, Any]]:
    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            """
            SELECT project_id, project_name, project_type, directory, directory_realpath,
                   chunk_count, file_count, total_tokens, updated_at, deleting
            FROM projects
            WHERE deleting = 0
            ORDER BY updated_at DESC
            """
        )
        return [dict(r) for r in rows]


async def get_project(db_path: str, project_id: str) -> Optional[Dict[str, Any]]:
    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        row = await db.execute_fetchone(
            """
            SELECT project_id, project_name, project_type, directory, directory_realpath, embedding_dim, metadata,
                   index_dirty, faiss_index_uuid, deleting, chunk_count, file_count, total_tokens, created_at, updated_at
            FROM projects
            WHERE project_id = ?
            """,
            (project_id,),
        )
        if not row:
            return None
        d = dict(row)
        d["metadata"] = _safe_load_metadata(d.get("metadata"), context=f"project:{project_id}")
        d["index_dirty"] = int(d.get("index_dirty") or 0)
        return d


async def get_project_by_name(
    db_path: str,
    *,
    project_name: str,
) -> Optional[Dict[str, Any]]:
    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        row = await db.execute_fetchone(
            """
            SELECT project_id, project_name, project_type, directory, directory_realpath, embedding_dim, metadata,
                   index_dirty, faiss_index_uuid, deleting, chunk_count, file_count, total_tokens, updated_at
            FROM projects
            WHERE project_name = ?
            """,
            (project_name,),
        )
        if not row:
            return None
        d = dict(row)
        d["metadata"] = _safe_load_metadata(d.get("metadata"), context=f"project_name:{project_name}")
        d["index_dirty"] = int(d.get("index_dirty") or 0)
        return d


async def get_project_by_directory_realpath(
    db_path: str,
    *,
    directory_realpath: str,
) -> Optional[Dict[str, Any]]:
    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        row = await db.execute_fetchone(
            """
            SELECT project_id, project_name, project_type, directory, directory_realpath, embedding_dim, metadata,
                   index_dirty, faiss_index_uuid, deleting, chunk_count, file_count, total_tokens, updated_at
            FROM projects
            WHERE directory_realpath = ?
            """,
            (directory_realpath,),
        )
        if not row:
            return None
        d = dict(row)
        d["metadata"] = _safe_load_metadata(
            d.get("metadata"), context=f"project_directory:{directory_realpath}"
        )
        d["index_dirty"] = int(d.get("index_dirty") or 0)
        return d


async def set_project_index_uuid(db_path: str, project_id: str, index_uuid: str) -> None:
    async with get_connection(db_path) as db:
        await db.execute(
            """
            UPDATE projects
            SET faiss_index_uuid = ?, updated_at = datetime('now')
            WHERE project_id = ?
            """,
            (index_uuid, project_id),
        )
        await db.commit()


async def set_project_deleting(db_path: str, project_id: str, deleting: bool) -> None:
    async with get_connection(db_path) as db:
        await db.execute(
            """
            UPDATE projects
            SET deleting = ?, updated_at = datetime('now')
            WHERE project_id = ?
            """,
            (1 if deleting else 0, project_id),
        )
        await db.commit()


async def reserve_internal_ids(db_path: str, project_id: str, count: int) -> List[int]:
    """Reserve 'count' new internal IDs for a project.

    We store internal IDs explicitly so we can do stable add_with_ids() into FAISS.

    The manual BEGIN IMMEDIATE / COMMIT pair is wrapped in try/except so that
    any failure (timeout, constraint violation, etc.) triggers an explicit
    rollback *before* the connection is returned to the pool.  Without this
    guard a failed mid-transaction error would leave the connection in an
    open-transaction state; because connections are pooled and reused the
    zombie transaction causes spurious SQLITE_BUSY errors on unrelated
    requests later.
    """
    if count > MAX_RESERVE_COUNT:
        raise ValueError(f"count cannot exceed {MAX_RESERVE_COUNT}")
    if count <= 0:
        return []

    async with get_connection(db_path) as db:
        await db.execute("BEGIN IMMEDIATE")
        try:
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
        except BaseException:
            await db.rollback()
            raise


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


async def upsert_chunk_vectors(
    db_path: str,
    rows: List[Tuple[bytes, str, int]],
) -> None:
    """Persist embedding vectors for crash-recovery without re-embedding.

    Each entry in *rows* is (vector_bytes, project_id, internal_id).
    The UPDATE targets chunks that were just inserted by upsert_chunks in
    the same logical batch, so all rows are guaranteed to exist.
    """
    if not rows:
        return
    async with get_connection(db_path) as db:
        await db.executemany(
            "UPDATE chunks SET vector = ? WHERE project_id = ? AND internal_id = ?",
            rows,
        )
        await db.commit()


async def upsert_graph_nodes(
    db_path: str,
    *,
    project_id: str,
    nodes: List[Tuple[str, str, str, str, int, int, Dict[str, Any]]],
) -> None:
    """Upsert graph nodes.

    Each tuple in *nodes*:
        (node_id, node_type, name, file_path, start_line, end_line, metadata)
    """
    if not nodes:
        return
    async with get_connection(db_path) as db:
        await db.executemany(
            """
            INSERT INTO graph_nodes(
                project_id, node_id, node_type, name, file_path,
                start_line, end_line, metadata
            )
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(project_id, node_id) DO UPDATE SET
              node_type  = excluded.node_type,
              name       = excluded.name,
              file_path  = excluded.file_path,
              start_line = excluded.start_line,
              end_line   = excluded.end_line,
              metadata   = excluded.metadata
            """,
            [
                (project_id, nid, ntype, name, fp, sl, el, json.dumps(meta))
                for (nid, ntype, name, fp, sl, el, meta) in nodes
            ],
        )
        await db.commit()


async def upsert_graph_edges(
    db_path: str,
    *,
    project_id: str,
    edges: List[Tuple[str, str, str]],
) -> None:
    """Upsert graph edges.

    Each tuple in *edges*: (source_id, target_id, edge_type)
    """
    if not edges:
        return
    async with get_connection(db_path) as db:
        await db.executemany(
            """
            INSERT INTO graph_edges(project_id, source_id, target_id, edge_type)
            VALUES(?,?,?,?)
            ON CONFLICT(project_id, source_id, target_id) DO UPDATE SET
              edge_type = excluded.edge_type
            """,
            [(project_id, src, tgt, etype) for (src, tgt, etype) in edges],
        )
        await db.commit()


async def delete_graph_nodes_for_files(
    db_path: str,
    *,
    project_id: str,
    file_paths: List[str],
) -> None:
    """Remove all graph nodes that belong to the given files.

    Called before re-parsing changed files so that stale nodes (e.g. a
    renamed function) do not linger.  Deleted files' nodes are cleaned up
    by the same call.
    """
    if not file_paths:
        return
    batch_size = 900
    async with get_connection(db_path) as db:
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i : i + batch_size]
            query = build_in_query(
                "DELETE FROM graph_nodes WHERE project_id = ? AND file_path IN ",
                batch,
            )
            await db.execute(query.text, (project_id, *query.params))
        await db.commit()


async def query_graph_nodes(
    db_path: str,
    *,
    project_id: str,
    node_type: Optional[str] = None,
    name_pattern: Optional[str] = None,
    file_path: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Query graph_nodes with optional filters.

    *name_pattern* is a SQL LIKE pattern (e.g. ``%parse%``).
    """
    clauses = ["project_id = ?"]
    params: List[Any] = [project_id]
    if node_type:
        clauses.append("node_type = ?")
        params.append(node_type)
    if name_pattern:
        clauses.append("name LIKE ?")
        params.append(name_pattern)
    if file_path:
        clauses.append("file_path = ?")
        params.append(file_path)
    params.append(int(limit))
    where = " AND ".join(clauses)
    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            f"""
            SELECT node_id, node_type, name, file_path, start_line, end_line, metadata
            FROM graph_nodes
            WHERE {where}
            ORDER BY file_path, start_line
            LIMIT ?
            """,
            tuple(params),
        )
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "node_id": r["node_id"],
                "node_type": r["node_type"],
                "name": r["name"],
                "file_path": r["file_path"],
                "start_line": int(r["start_line"]),
                "end_line": int(r["end_line"]),
                "metadata": _safe_load_metadata(r["metadata"], context=f"graph_node:{r['node_id']}"),
            }
        )
    return out


async def fetch_graph_edges(
    db_path: str,
    *,
    project_id: str,
    node_id: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return edges where *node_id* is the source or the target.

    The result dict has two keys:
      ``outgoing`` – edges where *node_id* is the source (calls, contains …)
      ``incoming`` – edges where *node_id* is the target (called_by …)
    """
    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        out_rows = await db.execute_fetchall(
            """
            SELECT target_id, edge_type
            FROM graph_edges
            WHERE project_id = ? AND source_id = ?
            """,
            (project_id, node_id),
        )
        in_rows = await db.execute_fetchall(
            """
            SELECT source_id, edge_type
            FROM graph_edges
            WHERE project_id = ? AND target_id = ?
            """,
            (project_id, node_id),
        )
    return {
        "outgoing": [{"target_id": r["target_id"], "edge_type": r["edge_type"]} for r in out_rows],
        "incoming": [{"source_id": r["source_id"], "edge_type": r["edge_type"]} for r in in_rows],
    }


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


_FTS5_KEYWORDS: set[str] = {"not", "and", "or", "near"}


def _sanitize_fts_query(query: str) -> str:
    """Sanitise *query* for FTS5 MATCH while preserving code punctuation.

    The previous implementation stripped everything except [A-Za-z0-9_],
    which destroyed code-search precision:  ``std::vector`` became two
    independent AND terms instead of a phrase match, ``@component``
    lost its decorator prefix, and ``C++`` collapsed to ``C``.

    FTS5 only needs a handful of structural operators removed to avoid
    query-parse errors (*  "  ^  +  (  )  {  }).  Angle brackets and
    hyphens are replaced with spaces: angle brackets so that
    ``vector<int>`` tokenises the same way at query time as at index
    time, and hyphens so that ``pre-trained`` is not mis-parsed as
    ``pre NOT trained`` by the FTS5 query engine.  Each
    whitespace-delimited token is then quoted so FTS5 treats it as a
    phrase; the ``unicode61`` tokeniser splits on the same boundaries
    used during indexing, making ``"std::vector"`` match the
    adjacent-token pair (std, vector).  Tokens that are bare FTS5
    operator keywords (NOT / AND / OR / NEAR) are dropped entirely to
    prevent unintended query-syntax injection.
    """
    if not query or not query.strip():
        return '""'
    # Replace FTS5 syntax operators, angle-brackets, and hyphens with spaces.
    # Note: '+' is deliberately NOT stripped — it is not an FTS5 operator
    # and stripping it collapses "C++" to "C".
    cleaned = re.sub(r'[*^(){}<>\-]', ' ', query)
    tokens = [t for t in cleaned.split() if t and t.lower() not in _FTS5_KEYWORDS]
    if not tokens:
        return '""'
    if len(tokens) > 50:
        raise ValueError("FTS strict query exceeds the 50-token maximum.")
    tokens = [t[:64] for t in tokens]
    return " AND ".join(f'"{t}"' for t in tokens)


def build_fts_query(query: str, *, mode: str = "strict") -> str:
    """Build a sanitized FTS5 query string supporting AND/OR modes and phrases."""
    if not query or not query.strip():
        return '""'
    # Replace FTS5 syntax operators, angle-brackets, and hyphens with spaces.
    # '+' is not an FTS5 operator; keeping it preserves "C++" precision.
    cleaned = re.sub(r'[*^(){}<>\-]', ' ', query)
    mode_lower = str(mode).lower()
    if mode_lower == "phrase":
        cleaned_phrase = re.sub(r'\s+', ' ', cleaned).strip()
        return f'"{cleaned_phrase[:256]}"' if cleaned_phrase else '""'
    parts = re.findall(r'"([^"]+)"|(\S+)', cleaned)
    tokens: List[str] = []
    for quoted, plain in parts:
        token = quoted or plain
        if not token:
            continue
        token = token.strip()
        if not token:
            continue
        token = re.sub(r'\s+', ' ', token)
        if token.lower() in _FTS5_KEYWORDS:
            continue
        tokens.append(token)
    if not tokens:
        return '""'
    if len(tokens) > 50:
        raise ValueError("FTS query exceeds the 50-token maximum.")
    tokens = [t[:64] for t in tokens]
    joiner = " OR " if mode_lower == "any" else " AND "
    return joiner.join(f'"{t}"' for t in tokens)


async def fts_search(
    db_path: str,
    *,
    project_id: str,
    query: str,
    limit: int,
    mode: str = "strict",
) -> List[Dict[str, Any]]:
    """Lexical search using FTS5 + bm25 ranking."""
    sanitized = build_fts_query(query, mode=mode)
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
        meta = _safe_load_metadata(r[4], context=f"chunk:{r[0]}")
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


async def mark_jobs_shutdown(db_path: str, *, error: str) -> None:
    async with get_connection(db_path) as db:
        await db.execute(
            """
            UPDATE jobs
            SET error = ?,
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
    max_rows: int = 0,
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
        await db.commit()
    await prune_embedding_cache(db_path, ttl_s=ttl_s, max_rows=max_rows)


async def prune_embedding_cache(
    db_path: str,
    *,
    ttl_s: int = 0,
    max_rows: int = 0,
    model_name: Optional[str] = None,
) -> int:
    ttl_s = int(ttl_s)
    max_rows = int(max_rows)
    if ttl_s <= 0 and max_rows <= 0:
        return 0
    deleted = 0
    async with get_connection(db_path) as db:
        if ttl_s > 0:
            if model_name:
                result = await db.execute(
                    """
                    DELETE FROM embedding_cache
                    WHERE model_name = ?
                      AND created_at < datetime('now', ?)
                    """,
                    (model_name, "-" + str(ttl_s) + " seconds"),
                )
            else:
                result = await db.execute(
                    """
                    DELETE FROM embedding_cache
                    WHERE created_at < datetime('now', ?)
                    """,
                    ("-" + str(ttl_s) + " seconds",),
                )
            deleted += result.rowcount if result.rowcount is not None else 0
        if max_rows > 0:
            if model_name:
                row = await db.execute_fetchone(
                    "SELECT COUNT(*) FROM embedding_cache WHERE model_name = ?",
                    (model_name,),
                )
            else:
                row = await db.execute_fetchone("SELECT COUNT(*) FROM embedding_cache")
            count = int(row[0]) if row else 0
            if count > max_rows:
                to_delete = count - max_rows
                if model_name:
                    result = await db.execute(
                        """
                        DELETE FROM embedding_cache
                        WHERE content_hash IN (
                            SELECT content_hash FROM embedding_cache
                            WHERE model_name = ?
                            ORDER BY created_at ASC
                            LIMIT ?
                        )
                        """,
                        (model_name, to_delete),
                    )
                else:
                    result = await db.execute(
                        """
                        DELETE FROM embedding_cache
                        WHERE content_hash IN (
                            SELECT content_hash FROM embedding_cache
                            ORDER BY created_at ASC
                            LIMIT ?
                        )
                        """,
                        (to_delete,),
                    )
                deleted += result.rowcount if result.rowcount is not None else 0
        await db.commit()
    return deleted


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


async def prune_jobs(db_path: str, *, max_age_days: int, max_rows: int) -> None:
    max_age_days = max(1, int(max_age_days))
    max_rows = max(1, int(max_rows))
    async with get_connection(db_path) as db:
        await db.execute(
            """
            DELETE FROM jobs
            WHERE created_at < datetime('now', ?)
            """,
            (f"-{max_age_days} days",),
        )
        await db.execute(
            """
            DELETE FROM jobs
            WHERE job_id NOT IN (
                SELECT job_id
                FROM jobs
                ORDER BY created_at DESC
                LIMIT ?
            )
            """,
            (max_rows,),
        )
        await db.commit()


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
            SELECT chunk_id, internal_id, content, token_count, metadata, vector
            FROM chunks
            WHERE project_id = ? AND internal_id > ?
            ORDER BY internal_id ASC
            LIMIT ?
            """,
            (project_id, int(last_internal_id), int(batch_size)),
        )

    out: List[Dict[str, Any]] = []
    for cid, iid, content, tcount, meta_json, vector in rows:
        meta = {}
        try:
            meta = json.loads(meta_json or "{}")
        except json.JSONDecodeError:
            logging.warning("Failed to decode chunk metadata for %s", cid, exc_info=True)
            meta = {}
        out.append(
            {
                "chunk": Chunk(chunk_id=cid, content=content, token_count=int(tcount), metadata=meta),
                "internal_id": int(iid),
                "vector": vector,
            }
        )
    return out


async def fetch_chunk_by_id(db_path: str, *, project_id: str, chunk_id: str) -> Optional[Dict[str, Any]]:
    async with get_connection(db_path) as db:
        row = await db.execute_fetchone(
            """
            SELECT chunk_id, internal_id, content, token_count, metadata, access_count
            FROM chunks
            WHERE project_id = ? AND chunk_id = ?
            """,
            (project_id, chunk_id),
        )
    if not row:
        return None
    meta = {}
    try:
        meta = json.loads(row[4] or "{}")
    except json.JSONDecodeError:
        logging.warning("Failed to decode chunk metadata for %s", row[0], exc_info=True)
        meta = {}
    return {
        "id": row[0],
        "internal_id": int(row[1]),
        "content": row[2],
        "token_count": int(row[3]),
        "metadata": meta,
        "access_count": int(row[5] or 0),
    }


async def fetch_chunks_by_ids(
    db_path: str,
    *,
    project_id: str,
    chunk_ids: List[str],
) -> List[Dict[str, Any]]:
    if not chunk_ids:
        return []
    batch_size = 900
    row_map: Dict[str, Dict[str, Any]] = {}
    async with get_connection(db_path) as db:
        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i:i + batch_size]
            query = build_in_query(
                """
                SELECT chunk_id, internal_id, content, token_count, metadata, access_count
                FROM chunks
                WHERE project_id = ? AND chunk_id IN 
                """,
                batch,
            )
            rows = await db.execute_fetchall(query.text, (project_id, *query.params))
            for cid, iid, content, tcount, meta_json, access_count in rows:
                meta = {}
                try:
                    meta = json.loads(meta_json or "{}")
                except json.JSONDecodeError:
                    logging.warning("Failed to decode chunk metadata for %s", cid, exc_info=True)
                    meta = {}
                row_map[str(cid)] = {
                    "id": cid,
                    "internal_id": int(iid),
                    "content": content,
                    "token_count": int(tcount),
                    "metadata": meta,
                    "access_count": int(access_count or 0),
                }
    return [row_map[cid] for cid in chunk_ids if cid in row_map]


async def mark_project_dirty(db_path: str, project_id: str) -> None:
    """Set index_dirty flag before writing chunks to the DB.

    If the process crashes after the DB commit but before the FAISS index
    is persisted to disk, the flag survives and ``update_project`` will
    trigger a full index rebuild on next startup.
    """
    async with get_connection(db_path) as db:
        await db.execute(
            "UPDATE projects SET index_dirty = 1, updated_at = datetime('now') WHERE project_id = ?",
            (project_id,),
        )
        await db.commit()


async def clear_project_dirty(db_path: str, project_id: str) -> None:
    """Clear index_dirty after a successful FAISS disk write."""
    async with get_connection(db_path) as db:
        await db.execute(
            "UPDATE projects SET index_dirty = 0, updated_at = datetime('now') WHERE project_id = ?",
            (project_id,),
        )
        await db.commit()


# ============================================================================
# Agent-Native Tool Suite: Deterministic Queries
# ============================================================================


async def get_codebase_statistics(
    db_path: str,
    *,
    project_id: str,
) -> Dict[str, Any]:
    """Return high-level codebase overview for agent orientation.

    Returns:
        - total_files: Total file count
        - total_chunks: Total chunk count
        - top_modified_files: Top 5 most recently modified files
        - top_directories: Top-level directories in the project
        - language_breakdown: Count of files by extension
    """
    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row

        # Get total files and chunks
        stats_row = await db.execute_fetchone(
            """
            SELECT
                COUNT(DISTINCT fm.file_path) as file_count,
                COUNT(DISTINCT c.chunk_id) as chunk_count
            FROM file_metadata fm
            LEFT JOIN file_chunks fc ON fm.project_id = fc.project_id AND fm.file_path = fc.file_path
            LEFT JOIN chunks c ON fc.chunk_id = c.chunk_id
            WHERE fm.project_id = ?
            """,
            (project_id,),
        )

        # Get top 5 most recently modified files
        top_files_rows = await db.execute_fetchall(
            """
            SELECT file_path, mtime_ns, size_bytes
            FROM file_metadata
            WHERE project_id = ?
            ORDER BY mtime_ns DESC
            LIMIT 5
            """,
            (project_id,),
        )

        # Get all file paths for directory and extension analysis
        all_files_rows = await db.execute_fetchall(
            "SELECT file_path FROM file_metadata WHERE project_id = ?",
            (project_id,),
        )

    # Analyze directories
    top_dirs: set[str] = set()
    ext_counts: Dict[str, int] = {}

    for row in all_files_rows:
        path = str(row["file_path"])
        # Extract top-level directory
        parts = path.split("/")
        if len(parts) > 1:
            top_dirs.add(parts[0])
        # Extract extension
        if "." in path:
            ext = path.rsplit(".", 1)[-1]
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    # Sort extensions by count
    sorted_exts = sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_files": int(stats_row["file_count"]) if stats_row else 0,
        "total_chunks": int(stats_row["chunk_count"]) if stats_row else 0,
        "top_modified_files": [
            {
                "path": r["file_path"],
                "size_bytes": int(r["size_bytes"]),
            }
            for r in top_files_rows
        ],
        "top_directories": sorted(top_dirs),
        "language_breakdown": [{"extension": ext, "count": count} for ext, count in sorted_exts],
    }


async def find_symbol_with_references(
    db_path: str,
    *,
    project_id: str,
    symbol_name: str,
) -> Dict[str, Any]:
    """Deterministic symbol lookup with full reference graph.

    Args:
        symbol_name: Exact or partial name (uses LIKE %symbol%)

    Returns:
        Dictionary with:
        - definitions: List of locations where symbol is defined
        - references: List of files/functions that reference the symbol
        - total_definitions: Count of definitions found
        - total_references: Count of references found
    """
    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row

        # Find all matching definitions (exact and partial)
        definition_rows = await db.execute_fetchall(
            """
            SELECT node_id, node_type, name, file_path, start_line, end_line, metadata
            FROM graph_nodes
            WHERE project_id = ? AND (name = ? OR name LIKE ?)
            ORDER BY file_path, start_line
            LIMIT 50
            """,
            (project_id, symbol_name, f"%{symbol_name}%"),
        )

        definitions = []
        all_node_ids = []

        for r in definition_rows:
            node_id = r["node_id"]
            all_node_ids.append(node_id)
            definitions.append(
                {
                    "node_id": node_id,
                    "node_type": r["node_type"],
                    "name": r["name"],
                    "file_path": r["file_path"],
                    "start_line": int(r["start_line"]),
                    "end_line": int(r["end_line"]),
                    "metadata": _safe_load_metadata(r["metadata"], context=f"symbol:{node_id}"),
                }
            )

        # Find all references (incoming edges to these symbols)
        references = []
        if all_node_ids:
            # Find all nodes that reference our symbols
            for node_id in all_node_ids:
                ref_rows = await db.execute_fetchall(
                    """
                    SELECT DISTINCT gn.node_id, gn.node_type, gn.name, gn.file_path,
                           gn.start_line, gn.end_line, ge.edge_type
                    FROM graph_edges ge
                    JOIN graph_nodes gn ON ge.source_id = gn.node_id AND ge.project_id = gn.project_id
                    WHERE ge.project_id = ? AND ge.target_id = ?
                    ORDER BY gn.file_path, gn.start_line
                    LIMIT 100
                    """,
                    (project_id, node_id),
                )

                for ref in ref_rows:
                    references.append(
                        {
                            "source_node_id": ref["node_id"],
                            "source_type": ref["node_type"],
                            "source_name": ref["name"],
                            "source_file": ref["file_path"],
                            "source_line": int(ref["start_line"]),
                            "edge_type": ref["edge_type"],
                            "target_symbol": symbol_name,
                        }
                    )

    return {
        "definitions": definitions,
        "references": references,
        "total_definitions": len(definitions),
        "total_references": len(references),
    }


async def fetch_chunks_by_file_lines(
    db_path: str,
    *,
    project_id: str,
    file_path: str,
    start_line: int,
    end_line: int,
) -> List[Dict[str, Any]]:
    """Retrieve chunks that cover a specific line range in a file.

    This is optimized for stack trace analysis where we need to fetch
    the exact code around a specific line number.

    Args:
        file_path: Relative path within the project
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed)

    Returns:
        List of chunk dictionaries containing the relevant lines
    """
    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row

        # Get all chunks for this file
        rows = await db.execute_fetchall(
            """
            SELECT c.chunk_id, c.internal_id, c.content, c.token_count,
                   c.metadata, c.access_count
            FROM chunks c
            JOIN file_chunks fc ON c.chunk_id = fc.chunk_id
            WHERE fc.project_id = ? AND fc.file_path = ?
            ORDER BY c.internal_id
            """,
            (project_id, file_path),
        )

    results = []
    for r in rows:
        meta = _safe_load_metadata(r["metadata"], context=f"chunk:{r['chunk_id']}")

        # Check if chunk covers the requested line range
        chunk_start = meta.get("start_line", 0)
        chunk_end = meta.get("end_line", 999999)

        # Include chunk if there's any overlap with requested range
        if chunk_start <= end_line and chunk_end >= start_line:
            results.append(
                {
                    "id": r["chunk_id"],
                    "internal_id": int(r["internal_id"]),
                    "content": r["content"],
                    "token_count": int(r["token_count"]),
                    "metadata": meta,
                    "access_count": int(r["access_count"] or 0),
                    "file_path": file_path,
                }
            )

    return results


async def _ensure_repo_rules_schema(db: aiosqlite.Connection) -> None:
    """Migration repo_rules_v1: create repo_rules table for Shadow Documentation."""
    row = await db.execute_fetchone(
        "SELECT 1 FROM schema_migrations WHERE name = 'repo_rules_v1'"
    )
    if row:
        return

    await db.executescript(
        """
        CREATE TABLE IF NOT EXISTS repo_rules (
            rule_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            file_pattern TEXT NOT NULL,
            rule_text TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_repo_rules_project
            ON repo_rules(project_id, priority DESC);
        """
    )
    await db.execute(
        "INSERT INTO schema_migrations(name) VALUES(?)", ("repo_rules_v1",)
    )


async def upsert_repo_rule(
    db_path: str,
    *,
    project_id: str,
    rule_id: str,
    file_pattern: str,
    rule_text: str,
    priority: int = 0,
) -> None:
    """Add or update a repository rule for Shadow Documentation.

    Args:
        rule_id: Unique identifier for the rule
        file_pattern: Glob pattern to match files (e.g., "*.py", "src/**/*.ts")
        rule_text: The constraint or guideline text to inject
        priority: Higher priority rules are applied first (default 0)
    """
    async with get_connection(db_path) as db:
        await _ensure_repo_rules_schema(db)
        await db.execute(
            """
            INSERT INTO repo_rules(rule_id, project_id, file_pattern, rule_text, priority)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(rule_id) DO UPDATE SET
                project_id = excluded.project_id,
                file_pattern = excluded.file_pattern,
                rule_text = excluded.rule_text,
                priority = excluded.priority
            """,
            (rule_id, project_id, file_pattern, rule_text, priority),
        )
        await db.commit()


async def fetch_repo_rules(
    db_path: str,
    *,
    project_id: str,
) -> List[Dict[str, Any]]:
    """Fetch all repository rules for a project, ordered by priority."""
    async with get_connection(db_path) as db:
        await _ensure_repo_rules_schema(db)
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            """
            SELECT rule_id, file_pattern, rule_text, priority
            FROM repo_rules
            WHERE project_id = ?
            ORDER BY priority DESC
            """,
            (project_id,),
        )

    return [
        {
            "rule_id": r["rule_id"],
            "file_pattern": r["file_pattern"],
            "rule_text": r["rule_text"],
            "priority": int(r["priority"]),
        }
        for r in rows
    ]


async def delete_repo_rule(db_path: str, rule_id: str) -> None:
    """Delete a repository rule by ID."""
    async with get_connection(db_path) as db:
        await db.execute("DELETE FROM repo_rules WHERE rule_id = ?", (rule_id,))
        await db.commit()
