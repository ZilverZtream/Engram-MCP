import pytest

pytest.importorskip("aiosqlite")

from engram_mcp import db as dbmod


def test_build_fts_query_modes():
    assert dbmod.build_fts_query("foo bar", mode="strict") == '"foo" AND "bar"'
    assert dbmod.build_fts_query("foo bar", mode="any") == '"foo" OR "bar"'


def test_build_fts_query_phrases():
    query = '"foo bar" baz'
    assert dbmod.build_fts_query(query, mode="strict") == '"foo bar" AND "baz"'
