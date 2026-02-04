import pytest

pytest.importorskip("tokenizers")

from engram_mcp import chunking


def test_chunking_deterministic():
    text = "one two three four five six seven eight nine ten"
    chunks_a = chunking.chunk_text(
        text=text,
        base_id="base",
        meta={},
        target_tokens=5,
        overlap_tokens=1,
    )
    chunks_b = chunking.chunk_text(
        text=text,
        base_id="base",
        meta={},
        target_tokens=5,
        overlap_tokens=1,
    )
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
