import asyncio

import pytest

pytest.importorskip("aiosqlite")

from engram_mcp import db as dbmod


def test_dirty_flag_ignores_invalid_metadata(tmp_path):
    db_path = str(tmp_path / "engram.db")

    asyncio.run(dbmod.init_db(db_path))
    asyncio.run(
        dbmod.upsert_project(
            db_path,
            project_id="proj1",
            project_name="Project One",
            project_type="text",
            directory="/tmp",
            directory_realpath="/tmp",
            embedding_dim=384,
            metadata={"model_name": "dummy"},
            faiss_index_uuid="uuid1",
        )
    )

    async def _corrupt_metadata():
        async with dbmod.get_connection(db_path) as db:
            await db.execute("UPDATE projects SET metadata = '{' WHERE project_id = ?", ("proj1",))
            await db.commit()

    asyncio.run(_corrupt_metadata())

    asyncio.run(dbmod.mark_project_dirty(db_path, "proj1"))
    proj = asyncio.run(dbmod.get_project(db_path, "proj1"))
    assert proj["index_dirty"] == 1

    asyncio.run(dbmod.clear_project_dirty(db_path, "proj1"))
    proj = asyncio.run(dbmod.get_project(db_path, "proj1"))
    assert proj["index_dirty"] == 0
