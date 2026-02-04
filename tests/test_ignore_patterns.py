import pytest

pytest.importorskip("tokenizers")

from engram_mcp.parsing import iter_files
from engram_mcp.security import PathContext


def test_ignore_patterns(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "keep.txt").write_text("ok")
    (root / "ignore.log").write_text("no")
    path_context = PathContext([str(root)])
    files = list(iter_files(path_context, str(root), ["**/*.log"]))
    names = {p.name for p in files}
    assert "keep.txt" in names
    assert "ignore.log" not in names
