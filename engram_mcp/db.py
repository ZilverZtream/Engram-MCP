from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import aiosqlite
import numpy as np

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
    error TEXT,
    progress TEXT
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

-- Git history tables for Episodic Diff Memory
CREATE TABLE IF NOT EXISTS git_commits (
    commit_hash TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    author TEXT,
    timestamp INTEGER,
    message TEXT,
    embedding_uuid TEXT,
    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_git_commits_project ON git_commits(project_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_git_commits_embedding ON git_commits(embedding_uuid);

-- FTS5 index on commit messages for fast keyword search
CREATE VIRTUAL TABLE IF NOT EXISTS git_commits_fts USING fts5(
    commit_hash,
    message,
    tokenize = 'unicode61'
);

CREATE TRIGGER IF NOT EXISTS git_commits_fts_ai AFTER INSERT ON git_commits BEGIN
  INSERT INTO git_commits_fts(rowid, commit_hash, message)
  VALUES (new.rowid, new.commit_hash, new.message);
END;

CREATE TRIGGER IF NOT EXISTS git_commits_fts_ad AFTER DELETE ON git_commits BEGIN
  DELETE FROM git_commits_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS git_commits_fts_au AFTER UPDATE OF message ON git_commits BEGIN
  UPDATE git_commits_fts SET message = new.message WHERE rowid = new.rowid;
END;

CREATE TABLE IF NOT EXISTS git_diffs (
    id TEXT PRIMARY KEY,
    commit_hash TEXT NOT NULL,
    project_id TEXT NOT NULL DEFAULT '',
    file_path TEXT NOT NULL,
    change_type TEXT,
    diff_content TEXT,
    FOREIGN KEY(commit_hash) REFERENCES git_commits(commit_hash) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_git_diffs_commit ON git_diffs(commit_hash);
CREATE INDEX IF NOT EXISTS idx_git_diffs_file ON git_diffs(file_path);
CREATE INDEX IF NOT EXISTS idx_git_diffs_project_file ON git_diffs(project_id, file_path);

CREATE TABLE IF NOT EXISTS git_tags (
    commit_hash TEXT,
    tag TEXT,
    PRIMARY KEY(commit_hash, tag),
    FOREIGN KEY(commit_hash) REFERENCES git_commits(commit_hash) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_git_tags_tag ON git_tags(tag);
"""

MAX_RESERVE_COUNT = 1_000_000
_MAX_CHUNK_BATCH_CONTENT_BYTES = 16 * 1024 * 1024


def _is_duplicate_column_error(exc: Exception, column_name: str) -> bool:
    msg = str(exc).lower()
    return f"duplicate column name: {column_name.lower()}" in msg


async def _add_column_if_missing(db: aiosqlite.Connection, table: str, column_def: str) -> None:
    column_name = column_def.split()[0]
    try:
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
    except aiosqlite.OperationalError as exc:
        if _is_duplicate_column_error(exc, column_name):
            return
        raise


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
        obj = json.loads(raw or "{}")
    except json.JSONDecodeError:
        logging.warning("Failed to decode metadata for %s", context, exc_info=True)
        # Fail-closed: corrupted metadata is treated as restricted content.
        return {"_metadata_decode_error": True, "private": True}
    if not isinstance(obj, dict):
        logging.warning("Metadata payload for %s is not a JSON object", context)
        return {"_metadata_decode_error": True, "private": True}
    return obj


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
                        conn = await asyncio.wait_for(aiosqlite.connect(self.db_path, isolation_level=None), timeout=self.timeout_s)
                        await conn.execute("PRAGMA foreign_keys=ON;")
                        await conn.execute("PRAGMA journal_mode=WAL;")
                        self._all.add(conn)
                        return conn
                    except Exception:
                        async with self._lock:
                            self._created -= 1
                        raise
                conn = await asyncio.wait_for(self._queue.get(), timeout=self.timeout_s)
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
            async with self._lock:
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
        await _ensure_git_commits_fts_schema(db)
        await _ensure_git_diffs_project_id(db)
        await _ensure_jobs_progress_column(db)
        await db.commit()

async def _ensure_chunks_vector_column(db: aiosqlite.Connection) -> None:
    """Add the vector BLOB column used for crash-recovery without re-embedding."""
    rows = await db.execute_fetchall("PRAGMA table_info(chunks)")
    columns = {r[1] for r in rows}
    if "vector" not in columns:
        await _add_column_if_missing(db, "chunks", "vector BLOB")


async def _ensure_projects_schema(db: aiosqlite.Connection) -> None:
    rows = await db.execute_fetchall("PRAGMA table_info(projects)")
    columns = {r[1] for r in rows}
    if "directory_realpath" not in columns:
        await _add_column_if_missing(db, "projects", "directory_realpath TEXT")
    if "index_dirty" not in columns:
        await _add_column_if_missing(db, "projects", "index_dirty INTEGER NOT NULL DEFAULT 0")
    if "faiss_index_uuid" not in columns:
        await _add_column_if_missing(db, "projects", "faiss_index_uuid TEXT")
    if "deleting" not in columns:
        await _add_column_if_missing(db, "projects", "deleting INTEGER NOT NULL DEFAULT 0")


async def _ensure_file_metadata_schema(db: aiosqlite.Connection) -> None:
    rows = await db.execute_fetchall("PRAGMA table_info(file_metadata)")
    columns = {r[1] for r in rows}
    if "content_hash" not in columns:
        await _add_column_if_missing(db, "file_metadata", "content_hash TEXT NOT NULL DEFAULT ''")


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
    """Yield string entries from a JSON array with C-level parsing.

    This intentionally delegates parsing to ``json.loads`` to avoid Python
    bytecode hot loops over attacker-controlled whitespace/comma runs.
    """
    try:
        decoded = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed JSON array payload") from exc
    if not isinstance(decoded, list):
        raise ValueError("Expected JSON array payload")
    for obj in decoded:
        yield str(obj)


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
        await _add_column_if_missing(db, "search_sessions", "result_chunk_ids TEXT")


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
    """Upsert chunks with defensive internal-id collision handling.

    rows entries: (chunk_id, internal_id, content, token_count, metadata)
    """
    if not rows:
        return

    duplicate_counts = Counter(iid for _, iid, _, _, _ in rows)
    duplicate_iids = {iid for iid, count in duplicate_counts.items() if count > 1}
    if duplicate_iids:
        raise ValueError(f"Duplicate internal_id values in batch: {sorted(duplicate_iids)[:10]}")

    async with get_connection(db_path) as db:
        normalized_rows = list(rows)
        if normalized_rows:
            incoming_iids = [iid for _, iid, _, _, _ in normalized_rows]
            query = build_in_query(
                "SELECT internal_id, chunk_id FROM chunks WHERE project_id = ? AND internal_id IN ",
                incoming_iids,
            )
            existing = await db.execute_fetchall(query.text, (project_id, *query.params))
            occupied = {int(iid): str(cid) for iid, cid in existing}
            collisions = [
                idx for idx, (cid, iid, _, _, _) in enumerate(normalized_rows)
                if iid in occupied and occupied[iid] != cid
            ]
            if collisions:
                seq_row = await db.execute_fetchone(
                    "SELECT next_internal_id FROM internal_id_sequence WHERE project_id = ?",
                    (project_id,),
                )
                if seq_row and seq_row[0] is not None:
                    next_id = int(seq_row[0])
                else:
                    max_row = await db.execute_fetchone(
                        "SELECT COALESCE(MAX(internal_id), 0) FROM chunks WHERE project_id = ?",
                        (project_id,),
                    )
                    next_id = int(max_row[0]) + 1
                for row_idx in collisions:
                    cid, _iid, content, tcount, meta = normalized_rows[row_idx]
                    normalized_rows[row_idx] = (cid, next_id, content, tcount, meta)
                    next_id += 1
                await db.execute(
                    """
                    INSERT INTO internal_id_sequence(project_id, next_internal_id)
                    VALUES(?,?)
                    ON CONFLICT(project_id) DO UPDATE SET
                      next_internal_id = excluded.next_internal_id
                    """,
                    (project_id, next_id),
                )

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
                for (cid, iid, content, tcount, meta) in normalized_rows
            ],
        )
        await db.commit()


async def upsert_chunk_vectors(
    db_path: str,
    rows: List[Tuple[bytes, str, int]],
) -> None:
    """Persist embedding vectors with dimensionality validation."""
    if not rows:
        return

    project_ids = {project_id for _vector, project_id, _internal_id in rows}
    async with get_connection(db_path) as db:
        dims: Dict[str, int] = {}
        if project_ids:
            query = build_in_query(
                "SELECT project_id, embedding_dim FROM projects WHERE project_id IN ",
                sorted(project_ids),
            )
            dim_rows = await db.execute_fetchall(query.text, query.params)
            dims = {str(project_id): int(dim) for project_id, dim in dim_rows}

        for vector_bytes, project_id, internal_id in rows:
            dim = dims.get(project_id)
            if dim is None:
                raise ValueError(f"Unknown project_id for vector upsert: {project_id}")
            if len(vector_bytes) % 4 != 0:
                raise ValueError(
                    f"Vector byte length for {project_id}:{internal_id} is not float32-aligned: {len(vector_bytes)}"
                )
            expected_len = dim * 4
            if len(vector_bytes) != expected_len:
                raise ValueError(
                    f"Vector length mismatch for {project_id}:{internal_id}. "
                    f"Expected {expected_len} bytes ({dim} dims), got {len(vector_bytes)}"
                )

        await db.executemany(
            "UPDATE chunks SET vector = ? WHERE project_id = ? AND internal_id = ?",
            rows,
        )
        await db.commit()


async def upsert_chunks_with_embeddings(
    db_path: str,
    *,
    project_id: str,
    chunks: List[Tuple[str, str, str, int, str, np.ndarray]],
) -> None:
    """Upsert chunks with embeddings in a single transaction.

    This is used for git commit chunks where we have embeddings but no internal_id yet.
    Each tuple: (chunk_id, project_id, content, token_count, metadata_json, embedding)

    Args:
        db_path: Path to the database
        project_id: The project ID
        chunks: List of (chunk_id, project_id, content, token_count, metadata_json, embedding)
    """
    if not chunks:
        return

    async with get_connection(db_path) as db:
        # First, get the next available internal_id for this project
        row = await db.execute_fetchone(
            "SELECT COALESCE(MAX(internal_id), -1) FROM chunks WHERE project_id = ?",
            (project_id,),
        )
        next_internal_id = int(row[0]) + 1 if row else 0

        # Prepare chunk rows with sequential internal_ids
        chunk_rows = []
        for idx, (chunk_id, proj_id, content, token_count, metadata_json, embedding) in enumerate(chunks):
            internal_id = next_internal_id + idx
            vector_bytes = embedding.astype(np.float32).tobytes()
            chunk_rows.append((
                chunk_id,
                proj_id,
                internal_id,
                content,
                token_count,
                metadata_json,
                vector_bytes,
            ))

        # Insert/update chunks with vectors
        await db.executemany(
            """
            INSERT INTO chunks(chunk_id, project_id, internal_id, content, token_count, metadata, vector, updated_at)
            VALUES(?,?,?,?,?,?,?, datetime('now'))
            ON CONFLICT(chunk_id) DO UPDATE SET
              content = excluded.content,
              token_count = excluded.token_count,
              metadata = excluded.metadata,
              vector = excluded.vector,
              updated_at = datetime('now')
            """,
            chunk_rows,
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
_FTS5_OPERATORS = re.compile(r'[*^(){}<>]')


def _normalize_fts_term(term: str, *, max_len: int = 128) -> str:
    cleaned = _FTS5_OPERATORS.sub(" ", term)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned[:max_len]


def _tokenize_fts_expression(query: str) -> List[str]:
    return re.findall(r'"[^"]+"|\(|\)|\S+', query)


def _sanitize_fts_query(query: str) -> str:
    """Sanitize strict lexical query while preserving code punctuation."""
    return build_fts_query(query, mode="strict")


def build_fts_query(query: str, *, mode: str = "strict") -> str:
    """Build a sanitized FTS5 query string with optional boolean operators."""
    if not query or not query.strip():
        return '""'
    mode_lower = str(mode).lower()
    if mode_lower == "phrase":
        phrase = _normalize_fts_term(query, max_len=256)
        return f'"{phrase}"' if phrase else '""'

    raw_tokens = _tokenize_fts_expression(query)
    if not raw_tokens:
        return '""'

    out_tokens: List[str] = []
    pending_operator: Optional[str] = None
    terms_seen = 0

    for raw in raw_tokens:
        token = raw.strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in _FTS5_KEYWORDS:
            pending_operator = lowered.upper()
            continue
        if token in {"(", ")"}:
            continue

        if token.startswith('"') and token.endswith('"') and len(token) >= 2:
            normalized = _normalize_fts_term(token[1:-1], max_len=256)
        else:
            normalized = _normalize_fts_term(token)
        if not normalized:
            continue

        if terms_seen > 0:
            if mode_lower == "any":
                out_tokens.append("OR")
            else:
                out_tokens.append(pending_operator or "AND")
        out_tokens.append(f'"{normalized}"')
        terms_seen += 1
        pending_operator = None

        if terms_seen > 50:
            raise ValueError("FTS query exceeds the 50-token maximum.")

    return " ".join(out_tokens) if out_tokens else '""'


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


async def fetch_virtual_memory_files(
    db_path: str,
    project_id: str,
    prefix: str = "vfs://memory/",
) -> Dict[str, str]:
    """
    Fetch virtual memory bank files for a project.

    Returns a dictionary mapping section names to their content.
    For example: {"activeContext": "...", "productContext": "..."}
    """
    async with get_connection(db_path) as db:
        # Get all file paths that match the memory bank prefix
        rows = await db.execute_fetchall(
            """
            SELECT file_path
            FROM file_metadata
            WHERE project_id = ? AND file_path LIKE ?
            """,
            (project_id, f"{prefix}%"),
        )

        if not rows:
            return {}

        file_paths = [str(row[0]) for row in rows]

        # Fetch chunk IDs for these files
        chunk_map = await fetch_file_chunk_ids(db_path, project_id, file_paths)

        # Fetch the actual chunk content
        all_chunk_ids = [cid for cids in chunk_map.values() for cid in cids]
        if not all_chunk_ids:
            return {}

        # Fetch chunks in batches
        chunks_by_id: Dict[str, str] = {}
        batch_size = 900
        for i in range(0, len(all_chunk_ids), batch_size):
            batch = all_chunk_ids[i:i + batch_size]
            query = build_in_query(
                """
                SELECT chunk_id, content
                FROM chunks
                WHERE project_id = ? AND chunk_id IN
                """,
                batch,
            )
            chunk_rows = await db.execute_fetchall(query.text, (project_id, *query.params))
            for chunk_id, content in chunk_rows:
                chunks_by_id[str(chunk_id)] = str(content)

        # Assemble content by section
        result: Dict[str, str] = {}
        for file_path, chunk_ids in chunk_map.items():
            # Extract section name from path: vfs://memory/{section}.md -> section
            section_name = file_path.replace(prefix, "").replace(".md", "")
            # Concatenate all chunks for this file
            content = "".join(chunks_by_id.get(cid, "") for cid in chunk_ids)
            if content:
                result[section_name] = content

        return result


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


async def update_job_progress(
    db_path: str,
    *,
    job_id: str,
    progress: str,
) -> None:
    """Update the progress field for a running job.

    *progress* is a JSON-encoded string describing the current state, e.g.
    ``'{"files_parsed": 120, "files_total": 500, "phase": "embedding"}'``.
    Callers should update this at most once every ~100 files to avoid
    excessive write amplification.
    """
    async with get_connection(db_path) as db:
        await db.execute(
            """
            UPDATE jobs
            SET progress = ?, updated_at = datetime('now')
            WHERE job_id = ?
            """,
            (progress, job_id),
        )
        await db.commit()


async def list_jobs(db_path: str) -> List[Dict[str, Any]]:
    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            """
            SELECT job_id, kind, status, created_at, updated_at, error, progress
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
    max_content_bytes: int = _MAX_CHUNK_BATCH_CONTENT_BYTES,
) -> List[Dict[str, Any]]:
    from .chunking import Chunk

    capped_batch_size = max(1, int(batch_size))
    byte_budget = max(1024, int(max_content_bytes))

    out: List[Dict[str, Any]] = []
    consumed_bytes = 0
    async with get_connection(db_path) as db:
        cursor = await db.execute(
            """
            SELECT chunk_id, internal_id, content, token_count, metadata, vector
            FROM chunks
            WHERE project_id = ? AND internal_id > ?
            ORDER BY internal_id ASC
            LIMIT ?
            """,
            (project_id, int(last_internal_id), capped_batch_size),
        )
        try:
            while True:
                row = await cursor.fetchone()
                if row is None:
                    break
                cid, iid, content, tcount, meta_json, vector = row
                consumed_bytes += len((content or "").encode("utf-8", errors="ignore"))
                if consumed_bytes > byte_budget and out:
                    break
                meta = _safe_load_metadata(meta_json, context=f"chunk:{cid}")
                out.append(
                    {
                        "chunk": Chunk(chunk_id=cid, content=content, token_count=int(tcount), metadata=meta),
                        "internal_id": int(iid),
                        "vector": vector,
                    }
                )
                if consumed_bytes > byte_budget:
                    break
        finally:
            await cursor.close()
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
    """Return high-level codebase overview for agent orientation."""
    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
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

        top_dirs_rows = await db.execute_fetchall(
            """
            SELECT
                CASE
                    WHEN instr(file_path, '/') > 0 THEN substr(file_path, 1, instr(file_path, '/') - 1)
                    ELSE ''
                END AS top_dir,
                COUNT(*) AS cnt
            FROM file_metadata
            WHERE project_id = ?
            GROUP BY top_dir
            HAVING top_dir != ''
            ORDER BY cnt DESC, top_dir ASC
            LIMIT 25
            """,
            (project_id,),
        )

        ext_rows = await db.execute_fetchall(
            """
            SELECT
                lower(
                    CASE
                        WHEN instr(file_path, '.') = 0 THEN ''
                        ELSE substr(file_path, instr(file_path, '.') + 1)
                    END
                ) AS extension,
                COUNT(*) AS cnt
            FROM file_metadata
            WHERE project_id = ?
            GROUP BY extension
            HAVING extension != ''
            ORDER BY cnt DESC, extension ASC
            LIMIT 10
            """,
            (project_id,),
        )

    return {
        "total_files": int(stats_row["file_count"]) if stats_row else 0,
        "total_chunks": int(stats_row["chunk_count"]) if stats_row else 0,
        "top_modified_files": [
            {"path": r["file_path"], "size_bytes": int(r["size_bytes"])}
            for r in top_files_rows
        ],
        "top_directories": [str(r["top_dir"]) for r in top_dirs_rows],
        "language_breakdown": [
            {"extension": str(r["extension"]), "count": int(r["cnt"])} for r in ext_rows
        ],
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

        # Find temporally coupled files for the files containing these symbols
        coupled_files = []
        file_paths_in_definitions = list({d["file_path"] for d in definitions})

        for file_path in file_paths_in_definitions:
            # Create file node ID
            file_node_id = f"{project_id}:{file_path}:FILE:{file_path}"

            # Find coupled_with edges
            coupling_rows = await db.execute_fetchall(
                """
                SELECT ge.target_id, ge.edge_type
                FROM graph_edges ge
                WHERE ge.project_id = ? AND ge.source_id = ? AND ge.edge_type = 'coupled_with'
                LIMIT 20
                """,
                (project_id, file_node_id),
            )

            for row in coupling_rows:
                target_id = row["target_id"]
                # Extract file path from node_id (format: project_id:file_path:FILE:file_path)
                parts = target_id.split(":", 3)
                if len(parts) >= 2:
                    coupled_file_path = parts[1]
                    coupled_files.append({
                        "source_file": file_path,
                        "coupled_file": coupled_file_path,
                        "edge_type": "coupled_with",
                    })

    return {
        "definitions": definitions,
        "references": references,
        "coupled_files": coupled_files,
        "total_definitions": len(definitions),
        "total_references": len(references),
        "total_coupled_files": len(coupled_files),
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


async def _ensure_git_commits_fts_schema(db: aiosqlite.Connection) -> None:
    """Migration git_commits_fts_v1: add FTS5 virtual table for commit message search."""
    row = await db.execute_fetchone(
        "SELECT 1 FROM schema_migrations WHERE name = 'git_commits_fts_v1'"
    )
    if row:
        return

    await db.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS git_commits_fts USING fts5(
            commit_hash,
            message,
            tokenize = 'unicode61'
        );

        DROP TRIGGER IF EXISTS git_commits_fts_ai;
        DROP TRIGGER IF EXISTS git_commits_fts_ad;
        DROP TRIGGER IF EXISTS git_commits_fts_au;

        CREATE TRIGGER git_commits_fts_ai AFTER INSERT ON git_commits BEGIN
          INSERT INTO git_commits_fts(rowid, commit_hash, message)
          VALUES (new.rowid, new.commit_hash, new.message);
        END;

        CREATE TRIGGER git_commits_fts_ad AFTER DELETE ON git_commits BEGIN
          DELETE FROM git_commits_fts WHERE rowid = old.rowid;
        END;

        CREATE TRIGGER git_commits_fts_au AFTER UPDATE OF message ON git_commits BEGIN
          UPDATE git_commits_fts SET message = new.message WHERE rowid = new.rowid;
        END;
        """
    )
    # Backfill any pre-existing commits into the new FTS index
    await db.execute("DELETE FROM git_commits_fts;")
    await db.execute(
        "INSERT INTO git_commits_fts(rowid, commit_hash, message) "
        "SELECT rowid, commit_hash, message FROM git_commits;"
    )
    await db.execute(
        "INSERT INTO schema_migrations(name) VALUES(?)", ("git_commits_fts_v1",)
    )


async def _ensure_jobs_progress_column(db: aiosqlite.Connection) -> None:
    """Migration jobs_progress_v1: add progress TEXT column to the jobs table."""
    await _add_column_if_missing(db, "jobs", "progress TEXT")


async def _ensure_git_diffs_project_id(db: aiosqlite.Connection) -> None:
    """Migration git_diffs_project_id_v1: add project_id column to git_diffs.

    Without this column, diffs from different projects share the same table
    rows and temporal-coupling queries that filter on project_id would fail
    with OperationalError. Backfills from git_commits so existing rows get
    their project_id populated correctly.
    """
    row = await db.execute_fetchone(
        "SELECT 1 FROM schema_migrations WHERE name = 'git_diffs_project_id_v1'"
    )
    if row:
        return

    # Add column if missing (idempotent)
    await _add_column_if_missing(db, "git_diffs", "project_id TEXT NOT NULL DEFAULT ''")

    # Backfill project_id from the parent git_commits row
    await db.execute(
        """
        UPDATE git_diffs
        SET project_id = (
            SELECT project_id FROM git_commits
            WHERE git_commits.commit_hash = git_diffs.commit_hash
        )
        WHERE project_id = ''
        """
    )

    # Add the index now that the column is populated
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_git_diffs_project_file ON git_diffs(project_id, file_path)"
    )

    await db.execute(
        "INSERT INTO schema_migrations(name) VALUES(?)", ("git_diffs_project_id_v1",)
    )


# ============================================================================
# Episodic Diff Memory: Git History API
# ============================================================================


async def upsert_git_commits(
    db_path: str,
    *,
    commits: List[Tuple[str, str, str, int, str, Optional[str]]],
) -> None:
    """Bulk insert or update git commit records.

    Each tuple: (commit_hash, project_id, author, timestamp, message, embedding_uuid)
    """
    if not commits:
        return
    async with get_connection(db_path) as db:
        await db.executemany(
            """
            INSERT INTO git_commits(commit_hash, project_id, author, timestamp, message, embedding_uuid)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(commit_hash) DO UPDATE SET
                author = excluded.author,
                timestamp = excluded.timestamp,
                message = excluded.message,
                embedding_uuid = excluded.embedding_uuid
            """,
            commits,
        )
        await db.commit()


async def upsert_git_diffs(
    db_path: str,
    *,
    diffs: List[Tuple[str, str, str, str, str, str]],
) -> None:
    """Bulk insert or update git diff records.

    Each tuple: (id, commit_hash, project_id, file_path, change_type, diff_content)
    """
    if not diffs:
        return
    async with get_connection(db_path) as db:
        await db.executemany(
            """
            INSERT INTO git_diffs(id, commit_hash, project_id, file_path, change_type, diff_content)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                project_id = excluded.project_id,
                change_type = excluded.change_type,
                diff_content = excluded.diff_content
            """,
            diffs,
        )
        await db.commit()


async def upsert_git_tags(
    db_path: str,
    *,
    tags: List[Tuple[str, str]],
) -> None:
    """Bulk insert or update git tags.

    Each tuple: (commit_hash, tag)
    """
    if not tags:
        return
    async with get_connection(db_path) as db:
        await db.executemany(
            """
            INSERT OR IGNORE INTO git_tags(commit_hash, tag)
            VALUES(?, ?)
            """,
            tags,
        )
        await db.commit()


async def fetch_git_commits(
    db_path: str,
    *,
    project_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Fetch git commits for a project, ordered by timestamp descending."""
    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            """
            SELECT commit_hash, author, timestamp, message, embedding_uuid
            FROM git_commits
            WHERE project_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (project_id, limit),
        )
    return [dict(r) for r in rows]


async def fetch_commits_by_uuids(
    db_path: str,
    *,
    project_id: str,
    uuids: List[str],
) -> List[Dict[str, Any]]:
    """Fetch git commits by their embedding UUIDs (for vector search).

    Args:
        db_path: Path to the database
        project_id: The project ID to filter by
        uuids: List of embedding UUIDs to fetch

    Returns:
        List of commit dictionaries with standard fields
    """
    if not uuids:
        return []

    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        query = build_in_query(
            """
            SELECT commit_hash, author, timestamp, message, embedding_uuid
            FROM git_commits
            WHERE project_id = ? AND embedding_uuid IN
            """,
            uuids,
        )
        rows = await db.execute_fetchall(query.text, (project_id, *query.params))
    return [dict(r) for r in rows]


async def fetch_git_diffs_for_commit(
    db_path: str,
    *,
    commit_hash: str,
) -> List[Dict[str, Any]]:
    """Fetch all diffs for a specific commit."""
    async with get_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute_fetchall(
            """
            SELECT id, file_path, change_type, diff_content
            FROM git_diffs
            WHERE commit_hash = ?
            """,
            (commit_hash,),
        )
    return [dict(r) for r in rows]


async def search_git_commits_by_message(
    db_path: str,
    *,
    project_id: str,
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Search commit messages using FTS5 full-text search.

    Falls back to a plain LIKE query when the FTS index is not yet available
    (e.g. on a database that has not been migrated yet).
    """
    sanitized = build_fts_query(query)
    try:
        async with get_connection(db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(
                """
                SELECT c.commit_hash, c.author, c.timestamp, c.message, c.embedding_uuid
                FROM git_commits_fts AS fts
                JOIN git_commits c ON c.commit_hash = fts.commit_hash
                WHERE c.project_id = ? AND fts MATCH ?
                ORDER BY bm25(fts) ASC, c.timestamp DESC
                LIMIT ?
                """,
                (project_id, sanitized, limit),
            )
    except aiosqlite.OperationalError:
        logging.warning("git_commits_fts not available, falling back to LIKE", exc_info=True)
        async with get_connection(db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(
                """
                SELECT commit_hash, author, timestamp, message, embedding_uuid
                FROM git_commits
                WHERE project_id = ? AND message LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (project_id, f"%{query}%", limit),
            )
    return [dict(r) for r in rows]


async def fetch_git_tags_for_commit(
    db_path: str,
    *,
    commit_hash: str,
) -> List[str]:
    """Fetch all tags associated with a commit."""
    async with get_connection(db_path) as db:
        rows = await db.execute_fetchall(
            "SELECT tag FROM git_tags WHERE commit_hash = ?",
            (commit_hash,),
        )
    return [str(r[0]) for r in rows]


async def count_git_commits(
    db_path: str,
    *,
    project_id: str,
) -> int:
    """Count total git commits for a project."""
    async with get_connection(db_path) as db:
        row = await db.execute_fetchone(
            "SELECT COUNT(*) FROM git_commits WHERE project_id = ?",
            (project_id,),
        )
    return int(row[0]) if row else 0
