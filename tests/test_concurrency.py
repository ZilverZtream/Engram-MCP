import asyncio
import os

import pytest

pytest.importorskip("aiosqlite")

from engram_mcp import db as dbmod
from engram_mcp.indexing import _save_faiss, _write_index_uuid
from engram_mcp.search import SearchEngine
from engram_mcp.security import PathContext
from engram_mcp.locks import get_project_rwlock


@pytest.mark.asyncio
async def test_concurrent_search_delete(tmp_path):
    faiss = pytest.importorskip("faiss")
    np = pytest.importorskip("numpy")

    db_path = tmp_path / "memory.db"
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    path_context = PathContext([str(index_dir)])

    await dbmod.init_db(str(db_path))
    project_id = "proj1"
    await dbmod.upsert_project(
        str(db_path),
        project_id=project_id,
        project_name="Project 1",
        project_type="code",
        directory=None,
        embedding_dim=2,
        metadata={"faiss_index_uuid": "uuid1", "shard_count": 1, "shard_size": 100},
    )
    rows = [("chunk1", 1, "hello world", 2, {})]
    await dbmod.upsert_chunks(str(db_path), project_id=project_id, rows=rows)

    index = faiss.IndexFlatIP(2)
    index.add_with_ids(np.array([[1.0, 0.0]], dtype=np.float32), np.array([1], dtype=np.int64))
    index_path = index_dir / f"{project_id}.index.current"
    _save_faiss(index, path_context, str(index_path))
    _write_index_uuid(path_context, str(index_path), "uuid1")

    search_engine = SearchEngine(db_path=str(db_path), index_dir=str(index_dir), index_path_context=path_context)

    started = asyncio.Event()

    async def do_search():
        started.set()
        return await search_engine.search(
            project_id=project_id,
            query="hello",
            query_vec=np.array([1.0, 0.0], dtype=np.float32),
            fts_top_k=10,
            vector_top_k=10,
            return_k=5,
            enable_mmr=False,
            mmr_lambda=0.7,
            fts_mode="strict",
        )

    async def do_delete():
        await started.wait()
        rwlock = await get_project_rwlock(project_id)
        async with rwlock.write_lock():
            await dbmod.delete_project(str(db_path), project_id)
            if os.path.exists(index_path):
                os.unlink(index_path)
            uuid_path = str(index_path) + ".uuid"
            if os.path.exists(uuid_path):
                os.unlink(uuid_path)

    results, _ = await asyncio.gather(do_search(), do_delete())
    assert results
