import pytest

pytest.importorskip("aiosqlite")

from engram_mcp.jobs import generate_job_id


def test_generate_job_id_unique():
    ids = {generate_job_id("index") for _ in range(10000)}
    assert len(ids) == 10000
