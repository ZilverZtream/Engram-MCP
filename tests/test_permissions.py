import asyncio
import os
import pytest

pytest.importorskip("aiosqlite")

from engram_mcp import db as dbmod
from engram_mcp import indexing
from engram_mcp.security import PathContext


@pytest.mark.skipif(os.name == "nt", reason="Windows permissions differ")
def test_db_permissions(tmp_path):
    db_path = tmp_path / "memory.db"
    dbmod.ensure_db_permissions(str(db_path))
    mode = db_path.stat().st_mode & 0o777
    assert mode == 0o600


@pytest.mark.skipif(os.name == "nt", reason="Windows permissions differ")
def test_index_permissions(tmp_path):
    faiss = pytest.importorskip("faiss")
    path_context = PathContext([str(tmp_path)])
    index = faiss.IndexFlatIP(2)
    index_path = tmp_path / "proj.index.current"
    indexing._save_faiss(index, path_context, str(index_path))
    indexing._write_index_uuid(path_context, str(index_path), "abc123")
    db_path = tmp_path / "memory.db"
    asyncio.run(indexing.ensure_index_permissions(path_context, str(tmp_path), db_path=str(db_path)))
    index_mode = index_path.stat().st_mode & 0o777
    uuid_mode = (tmp_path / "proj.index.current.uuid").stat().st_mode & 0o777
    assert index_mode == 0o600
    assert uuid_mode == 0o600
