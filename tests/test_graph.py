"""Acceptance tests for the Polyglot GraphRAG extraction pipeline.

Layout
------
* Section A – Python AST extractor  (no external deps, always runs)
* Section B – Polyglot tree-sitter extractor  (skipped when tree-sitter-languages is absent)
  - B1: Systems   – Rust, Go
  - B2: Enterprise – Java, C#
  - B3: Embedded  – C, C++
  - B4: Web       – JavaScript, TypeScript
* Section C – DB schema migration & upsert helpers
"""

import asyncio

import pytest

pytest.importorskip("aiosqlite")

from engram_mcp import db as dbmod
from engram_mcp.parsing import (
    GraphNodeObj,
    LANGUAGE_CONFIG,
    _extract_python_graph,
)


# ---------------------------------------------------------------------------
# A – Python AST extractor
# ---------------------------------------------------------------------------


class TestExtractPythonGraph:
    def test_simple_functions(self):
        src = "def hello():\n    pass\n\ndef world():\n    pass\n"
        nodes, edges = _extract_python_graph(src, "lib.py")

        assert len(nodes) == 2
        assert edges == []
        names = {n.name for n in nodes}
        assert names == {"hello", "world"}
        assert all(n.node_type == "function" for n in nodes)
        assert all(n.file_path == "lib.py" for n in nodes)

    def test_async_function(self):
        src = "async def fetch():\n    pass\n"
        nodes, _ = _extract_python_graph(src, "net.py")

        assert len(nodes) == 1
        assert nodes[0].name == "fetch"
        assert nodes[0].node_type == "function"

    def test_class(self):
        src = "class Foo:\n    pass\n"
        nodes, _ = _extract_python_graph(src, "model.py")

        assert len(nodes) == 1
        assert nodes[0].name == "Foo"
        assert nodes[0].node_type == "class"
        assert nodes[0].node_id == "model.py:class:Foo"

    def test_class_with_methods(self):
        src = (
            "class Calculator:\n"
            "    def add(self, a, b):\n"
            "        return a + b\n"
            "    async def fetch(self):\n"
            "        pass\n"
        )
        nodes, _ = _extract_python_graph(src, "calc.py")

        names = {n.name for n in nodes}
        assert "Calculator" in names
        assert "add" in names
        assert "fetch" in names
        type_map = {n.name: n.node_type for n in nodes}
        assert type_map["Calculator"] == "class"
        assert type_map["add"] == "function"

    def test_syntax_error_returns_empty(self):
        src = "def broken(:\n"
        nodes, edges = _extract_python_graph(src, "bad.py")
        assert nodes == []
        assert edges == []

    def test_empty_file(self):
        nodes, edges = _extract_python_graph("", "empty.py")
        assert nodes == []
        assert edges == []

    def test_node_line_numbers(self):
        src = "# comment\n\ndef first():\n    pass\n\ndef second():\n    pass\n"
        nodes, _ = _extract_python_graph(src, "lines.py")

        by_name = {n.name: n for n in nodes}
        assert by_name["first"].start_line == 3
        assert by_name["second"].start_line == 6

    def test_metadata_language_tag(self):
        src = "def hi(): pass\n"
        nodes, _ = _extract_python_graph(src, "x.py")
        assert nodes[0].metadata == {"language": "python"}


# ---------------------------------------------------------------------------
# B – Polyglot tree-sitter extractor
# ---------------------------------------------------------------------------

ts_langs = pytest.importorskip("tree_sitter_languages")

from engram_mcp.parsing import _extract_polyglot_graph  # noqa: E402


# -- B1: Systems ----------------------------------------------------------


class TestSystemsRust:
    def test_function(self):
        src = "fn calculate_sum(a: i32, b: i32) -> i32 {\n    a + b\n}\n"
        nodes, _ = _extract_polyglot_graph(src, "main.rs", ".rs")

        assert any(n.name == "calculate_sum" and n.node_type == "function" for n in nodes)

    def test_struct(self):
        src = "struct Point {\n    x: f64,\n    y: f64,\n}\n"
        nodes, _ = _extract_polyglot_graph(src, "geom.rs", ".rs")

        assert any(n.name == "Point" and n.node_type == "class" for n in nodes)

    def test_impl_block_skipped(self):
        """impl blocks are silently skipped; only their inner fn items are kept."""
        src = (
            "struct Foo;\n"
            "impl Foo {\n"
            "    fn bar(&self) {}\n"
            "}\n"
        )
        nodes, _ = _extract_polyglot_graph(src, "foo.rs", ".rs")

        names = {n.name for n in nodes}
        assert "Foo" in names  # struct
        assert "bar" in names  # function inside impl
        # impl itself should not appear as a node (no navigable name)
        assert all(n.name != "Foo" or n.node_type == "class" for n in nodes)

    def test_metadata(self):
        src = "fn noop() {}\n"
        nodes, _ = _extract_polyglot_graph(src, "a.rs", ".rs")
        assert nodes[0].metadata == {"language": "rust"}


class TestSystemsGo:
    def test_function(self):
        src = "package main\n\nfunc CalculateSum(a, b int) int {\n\treturn a + b\n}\n"
        nodes, _ = _extract_polyglot_graph(src, "main.go", ".go")

        assert any(n.name == "CalculateSum" and n.node_type == "function" for n in nodes)

    def test_type_declaration(self):
        src = "package main\n\ntype Point struct {\n\tX float64\n\tY float64\n}\n"
        nodes, _ = _extract_polyglot_graph(src, "geom.go", ".go")

        assert any(n.name == "Point" and n.node_type == "class" for n in nodes)

    def test_method(self):
        src = (
            "package main\n\n"
            "type Calc struct{}\n\n"
            "func (c Calc) Add(a, b int) int {\n\treturn a + b\n}\n"
        )
        nodes, _ = _extract_polyglot_graph(src, "calc.go", ".go")

        names = {n.name for n in nodes}
        assert "Calc" in names
        assert "Add" in names


# -- B2: Enterprise --------------------------------------------------------


class TestEnterpriseJava:
    def test_class_and_method(self):
        src = (
            "public class UserManager {\n"
            "    public void Login() {}\n"
            "}\n"
        )
        nodes, _ = _extract_polyglot_graph(src, "UserManager.java", ".java")

        names = {n.name for n in nodes}
        assert "UserManager" in names
        assert "Login" in names
        type_map = {n.name: n.node_type for n in nodes}
        assert type_map["UserManager"] == "class"
        assert type_map["Login"] == "function"

    def test_interface(self):
        src = (
            "public interface Serializable {\n"
            "    void serialize();\n"
            "}\n"
        )
        nodes, _ = _extract_polyglot_graph(src, "Ser.java", ".java")

        assert any(n.name == "Serializable" and n.node_type == "class" for n in nodes)


class TestEnterpriseCSharp:
    def test_class_and_method(self):
        src = (
            "class UserManager {\n"
            "    void Login() {}\n"
            "}\n"
        )
        nodes, _ = _extract_polyglot_graph(src, "Program.cs", ".cs")

        names = {n.name for n in nodes}
        assert "UserManager" in names
        assert "Login" in names
        type_map = {n.name: n.node_type for n in nodes}
        assert type_map["UserManager"] == "class"
        assert type_map["Login"] == "function"

    def test_interface(self):
        src = (
            "interface IDisposable {\n"
            "    void Dispose();\n"
            "}\n"
        )
        nodes, _ = _extract_polyglot_graph(src, "iface.cs", ".cs")

        assert any(n.name == "IDisposable" and n.node_type == "class" for n in nodes)


# -- B3: Embedded ----------------------------------------------------------


class TestEmbeddedC:
    def test_function(self):
        src = "void hardware_init(int pin) {\n    // setup\n}\n"
        nodes, _ = _extract_polyglot_graph(src, "init.c", ".c")

        assert any(n.name == "hardware_init" and n.node_type == "function" for n in nodes)

    def test_struct(self):
        src = "struct Packet {\n    int length;\n    char data[256];\n};\n"
        nodes, _ = _extract_polyglot_graph(src, "net.c", ".c")

        assert any(n.name == "Packet" and n.node_type == "class" for n in nodes)

    def test_pointer_return_function(self):
        src = "int* allocate_buffer(int size) {\n    return 0;\n}\n"
        nodes, _ = _extract_polyglot_graph(src, "mem.c", ".c")

        assert any(n.name == "allocate_buffer" and n.node_type == "function" for n in nodes)


class TestEmbeddedCpp:
    def test_function(self):
        src = "void hardware_init(int pin) {\n}\n"
        nodes, _ = _extract_polyglot_graph(src, "init.cpp", ".cpp")

        assert any(n.name == "hardware_init" and n.node_type == "function" for n in nodes)

    def test_class(self):
        src = "class Motor {\npublic:\n    void start();\n};\n"
        nodes, _ = _extract_polyglot_graph(src, "motor.cpp", ".cpp")

        assert any(n.name == "Motor" and n.node_type == "class" for n in nodes)

    def test_class_and_function_together(self):
        src = (
            "class Motor {\npublic:\n    void start();\n};\n"
            "void hardware_init(int pin) {}\n"
        )
        nodes, _ = _extract_polyglot_graph(src, "init.cpp", ".cpp")

        names = {n.name for n in nodes}
        assert "Motor" in names
        assert "hardware_init" in names

    def test_header_class(self):
        src = "class Sensor {\npublic:\n    int read();\n};\n"
        nodes, _ = _extract_polyglot_graph(src, "sensor.hpp", ".hpp")

        assert any(n.name == "Sensor" and n.node_type == "class" for n in nodes)


# -- B4: Web ---------------------------------------------------------------


class TestWebJavaScript:
    def test_function_and_class(self):
        src = (
            "function greet(name) {\n    return 'hi ' + name;\n}\n"
            "class Handler {}\n"
        )
        nodes, _ = _extract_polyglot_graph(src, "app.js", ".js")

        names = {n.name for n in nodes}
        assert "greet" in names
        assert "Handler" in names


class TestWebTypeScript:
    def test_function_and_interface(self):
        src = (
            "function greet(name: string): string {\n    return name;\n}\n"
            "interface Config {\n    host: string;\n}\n"
        )
        nodes, _ = _extract_polyglot_graph(src, "app.ts", ".ts")

        names = {n.name for n in nodes}
        assert "greet" in names
        assert "Config" in names
        type_map = {n.name: n.node_type for n in nodes}
        assert type_map["greet"] == "function"
        assert type_map["Config"] == "class"


# -- B5: Registry coverage -------------------------------------------------


class TestLanguageConfigCoverage:
    """Verify every extension in LANGUAGE_CONFIG can produce at least one node
    from a trivial snippet (guards against typos in language names)."""

    SNIPPETS = {
        ".rs": "fn noop() {}\n",
        ".go": "package main\nfunc Noop() {}\n",
        ".c": "void noop() {}\n",
        ".cpp": "void noop() {}\n",
        ".h": "class Noop {};\n",
        ".hpp": "class Noop {};\n",
        ".java": "class Noop {}\n",
        ".cs": "class Noop {}\n",
        ".js": "function noop() {}\n",
        ".ts": "function noop() {}\n",
        ".tsx": "function noop() {}\n",
    }

    @pytest.mark.parametrize("ext", sorted(LANGUAGE_CONFIG.keys()))
    def test_extension(self, ext):
        snippet = self.SNIPPETS.get(ext)
        if snippet is None:
            pytest.skip(f"no snippet defined for {ext}")
        nodes, _ = _extract_polyglot_graph(snippet, f"file{ext}", ext)
        assert len(nodes) >= 1, f"No nodes extracted for {ext} from:\n{snippet}"


# ---------------------------------------------------------------------------
# C – Database schema & upsert
# ---------------------------------------------------------------------------


def _init(db_path: str) -> None:
    asyncio.run(dbmod.init_db(db_path))


def _create_project(db_path: str, project_id: str = "proj1") -> None:
    asyncio.run(
        dbmod.upsert_project(
            db_path,
            project_id=project_id,
            project_name="Test",
            project_type="code",
            directory="/tmp",
            directory_realpath="/tmp",
            embedding_dim=384,
            metadata={},
        )
    )


class TestGraphSchemaMigration:
    def test_tables_created(self, tmp_path):
        db_path = str(tmp_path / "engram.db")
        _init(db_path)

        async def _check():
            async with dbmod.get_connection(db_path) as db:
                # Migration recorded
                row = await db.execute_fetchone(
                    "SELECT 1 FROM schema_migrations WHERE name = 'graph_v1'"
                )
                assert row is not None

                # graph_nodes table exists
                row = await db.execute_fetchone(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='graph_nodes'"
                )
                assert row is not None

                # graph_edges table exists
                row = await db.execute_fetchone(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='graph_edges'"
                )
                assert row is not None

        asyncio.run(_check())

    def test_indexes_created(self, tmp_path):
        db_path = str(tmp_path / "engram.db")
        _init(db_path)

        async def _check():
            async with dbmod.get_connection(db_path) as db:
                rows = await db.execute_fetchall(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_graph_%'"
                )
                index_names = {r[0] for r in rows}
                assert "idx_graph_nodes_project_file" in index_names
                assert "idx_graph_nodes_project_name" in index_names
                assert "idx_graph_edges_project_source" in index_names
                assert "idx_graph_edges_project_target" in index_names

        asyncio.run(_check())

    def test_idempotent(self, tmp_path):
        """Calling init_db twice does not error."""
        db_path = str(tmp_path / "engram.db")
        _init(db_path)
        _init(db_path)  # second call must be a no-op


class TestUpsertGraphNodes:
    def test_insert_and_read(self, tmp_path):
        db_path = str(tmp_path / "engram.db")
        _init(db_path)
        _create_project(db_path)

        async def _run():
            await dbmod.upsert_graph_nodes(
                db_path,
                project_id="proj1",
                nodes=[
                    ("main.rs:function:foo", "function", "foo", "main.rs", 1, 3, {"language": "rust"}),
                    ("main.rs:class:Bar", "class", "Bar", "main.rs", 5, 10, {"language": "rust"}),
                ],
            )
            async with dbmod.get_connection(db_path) as db:
                rows = await db.execute_fetchall(
                    "SELECT node_id, node_type, name, file_path, start_line, end_line "
                    "FROM graph_nodes WHERE project_id = 'proj1' ORDER BY node_id"
                )
            return rows

        rows = asyncio.run(_run())
        assert len(rows) == 2
        # First row alphabetically is class:Bar
        assert rows[0][0] == "main.rs:class:Bar"
        assert rows[0][1] == "class"
        assert rows[0][2] == "Bar"
        # Second row is function:foo
        assert rows[1][0] == "main.rs:function:foo"
        assert rows[1][1] == "function"

    def test_upsert_updates_existing(self, tmp_path):
        db_path = str(tmp_path / "engram.db")
        _init(db_path)
        _create_project(db_path)

        async def _run():
            await dbmod.upsert_graph_nodes(
                db_path,
                project_id="proj1",
                nodes=[("a.py:function:f", "function", "f", "a.py", 1, 2, {})],
            )
            # Upsert again with updated line range
            await dbmod.upsert_graph_nodes(
                db_path,
                project_id="proj1",
                nodes=[("a.py:function:f", "function", "f", "a.py", 5, 10, {})],
            )
            async with dbmod.get_connection(db_path) as db:
                row = await db.execute_fetchone(
                    "SELECT start_line, end_line FROM graph_nodes WHERE node_id = 'a.py:function:f'"
                )
            return row

        row = asyncio.run(_run())
        assert row[0] == 5
        assert row[1] == 10

    def test_empty_nodes_is_noop(self, tmp_path):
        db_path = str(tmp_path / "engram.db")
        _init(db_path)
        _create_project(db_path)
        # Should not raise
        asyncio.run(dbmod.upsert_graph_nodes(db_path, project_id="proj1", nodes=[]))


class TestUpsertGraphEdges:
    def test_insert_and_read(self, tmp_path):
        db_path = str(tmp_path / "engram.db")
        _init(db_path)
        _create_project(db_path)

        async def _run():
            await dbmod.upsert_graph_edges(
                db_path,
                project_id="proj1",
                edges=[("a:function:caller", "a:function:callee", "calls")],
            )
            async with dbmod.get_connection(db_path) as db:
                row = await db.execute_fetchone(
                    "SELECT source_id, target_id, edge_type FROM graph_edges "
                    "WHERE project_id = 'proj1'"
                )
            return row

        row = asyncio.run(_run())
        assert row[0] == "a:function:caller"
        assert row[1] == "a:function:callee"
        assert row[2] == "calls"

    def test_empty_edges_is_noop(self, tmp_path):
        db_path = str(tmp_path / "engram.db")
        _init(db_path)
        _create_project(db_path)
        asyncio.run(dbmod.upsert_graph_edges(db_path, project_id="proj1", edges=[]))
