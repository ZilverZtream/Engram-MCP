# Engram MCP

<p align="center">
  <strong>A local, sovereign, hybrid memory system for MCP-compatible agents</strong>
</p>

<p align="center">
  Semantic search • Lexical search • Persistent memory • Non-blocking • Secure by default
</p>

---

## 🧠 What is Engram MCP?

**Engram MCP** is a high‑performance, open‑source memory server that gives AI agents *long‑term, searchable memory*.

In neuroscience, an **engram** is the physical trace of a memory in the brain.  
Engram MCP brings that concept to AI systems: durable, auditable, local memory that scales from small projects to universal personal knowledge bases.

Engram is **not** a chat log.  
It is **not** a volatile cache.  

It is a **cognitive substrate** for agents.

---

## ✨ Core Features

### 🔍 Hybrid Search (State of the Art)
- **Vector similarity search** (FAISS)
- **Lexical BM25 search** (SQLite FTS5, on‑disk)
- **Reciprocal Rank Fusion (RRF)** for best‑of‑both‑worlds ranking

### ⚡ Non‑Blocking Architecture
- Embeddings offloaded from the event loop
- Async SQLite via `aiosqlite`
- Long indexing jobs do **not** freeze search or control tools

### 🗂️ Persistent & Incremental Indexing
- Chunk‑level hashing & deduplication
- `ON CONFLICT` upserts
- Automatic re‑indexing when files change

### 🔐 Security‑First by Default
- Explicit **path whitelisting**
- Protection against path traversal & symlink escape
- Refuses to index outside configured roots

### 🧩 MCP‑Native
- Built on the **official MCP SDK / FastMCP**
- No manual JSON‑RPC parsing
- Forward‑compatible with protocol changes

### 🧪 Production‑Grade
- Modular architecture
- Clear separation of concerns
- Suitable for long‑running agent workloads

---

## 🏗️ Architecture Overview

```
engram_mcp/
├── server.py            # MCP entrypoint
├── config.py            # Configuration & security guards
├── indexing/
│   ├── indexer.py       # File discovery & chunking
│   └── workers.py       # Background execution
├── embeddings/
│   └── encoder.py       # Embedding model logic
├── database/
│   ├── schema.sql       # Tables, FTS5, triggers
│   └── store.py         # Async database access
├── search/
│   └── hybrid.py        # Vector + BM25 + RRF
├── utils/
│   ├── paths.py
│   └── hashing.py
└── engram_mcp.yaml.example
```

---

## 🚀 Quick Start

### 1️⃣ Install Engram MCP (pipx or uv)

**pipx (recommended for CLI usage)**
```bash
pipx install engram-mcp
engram-mcp
```

Enable vector search with FAISS CPU (optional):
```bash
pipx install "engram-mcp[cpu]"
```

Enable FAISS GPU builds (optional, Linux-first support):
```bash
pipx install "engram-mcp[gpu]"
```

**uv (fast, reproducible dev install)**
```bash
uv venv
uv pip install -e ".[cpu]"
uv run engram-mcp
```

> Base installs are **FTS-only** (no FAISS, no Numba JIT). CPU/GPU extras enable vector search.

---

### 2️⃣ Configure allowed paths

Engram reads its config from a **user-scoped location by default**:

| OS | Config path |
| --- | --- |
| Linux | `~/.config/engram/engram_mcp.yaml` |
| macOS | `~/.config/engram/engram_mcp.yaml` |
| Windows | `%APPDATA%\\engram\\engram_mcp.yaml` |

Create the file there (or set `ENGRAM_CONFIG_PATH` to a file *inside* that directory).

```bash
mkdir -p ~/.config/engram
cp engram_mcp.yaml.example ~/.config/engram/engram_mcp.yaml
```

Edit **`allowed_roots`**:

```yaml
allowed_roots:
  - /Users/you/Documents
  - /Users/you/Projects
```

> ⚠️ Engram **will refuse** to index anything outside these paths.

---

### 3️⃣ Run the server
```bash
engram-mcp
```

SentenceTransformers models are downloaded on first use unless you pre-download them (for air‑gapped setups, cache the models or point `model_name_*` to a local path).  
No surprises once the model cache is in place.

Engram MCP is now available to MCP‑compatible clients.

---

## ⚙️ Runtime Modes & Dependencies

Engram MCP starts in **FTS-only** mode by default (no FAISS, no Numba). Vector search is enabled when FAISS is installed.

Optional config flags in `engram_mcp.yaml`:
```yaml
vector_backend: auto   # auto | fts | faiss_cpu | faiss_gpu
enable_numba: false    # opt-in JIT kernels
search_cache_ttl_s: 300
search_cache_max_items: 512
```

On startup, Engram logs:
- storage paths (DB/index)
- vector search mode (and how to enable)
- numba status
- search cache status

---

## 🧠 MCP Tools

### `index_project`
Indexes a directory (must be within `allowed_roots`).

```json
{
  "directory": "/Users/you/Projects/my_repo",
  "project_name": "my_repo",
  "project_type": "code",
  "wait": true
}
```

---

### `search_memory`
Hybrid semantic + lexical search.

```json
{
  "query": "async sqlite performance issues",
  "project_id": "my_repo_123",
  "max_results": 10,
  "fts_mode": "strict"
}
```

---

### `update_project`
Updates an existing project (use `wait: false` to queue and poll job status).

```json
{
  "project_id": "my_repo_123",
  "wait": true
}
```

### `delete_project`
Removes all indexed content for a project root.

---

## 🔒 Security Model

Engram MCP **will refuse** to index paths that:

- Are outside `allowed_roots`
- Escape via symlinks
- Attempt traversal (`../`)

This prevents accidental indexing of:
- `/`
- `.ssh`
- System files
- Private or sensitive directories

Security is **opt‑in by configuration**, not implicit trust.

### Storage defaults (privacy-safe)
By default, Engram writes all state into a **user-scoped data directory** (never the current working directory):

| OS | Data directory |
| --- | --- |
| Linux | `~/.local/share/engram/` |
| macOS | `~/Library/Application Support/engram/` |
| Windows | `%APPDATA%\\engram\\` |

The SQLite DB and FAISS index files are created with **owner-only permissions**. Override `db_path`/`index_dir`
explicitly if you want storage in a custom location. On Windows, permissions are best-effort and may rely
on the existing directory ACLs.

### Query limits
`search_memory` enforces `max_query_chars` and `max_query_tokens` from config (defaults: 4096 chars / 256 tokens)
to prevent runaway embedding costs.

---

## ⚙️ Performance Notes

### Embeddings
- Executed off the event loop
- CPU → `ProcessPoolExecutor`
- CUDA → thread‑safe execution (no fork hazards)

### Search
- Vector search via FAISS
- Lexical search via SQLite FTS5
- No in‑memory BM25 structures

### Search mode
`search_memory` supports `fts_mode`:
- `strict` (default): `AND` across tokens
- `any`: `OR` across tokens

Quoted phrases are preserved in either mode.

## 🧪 Development

### Dependency locking
This repo uses **pip-tools**. Update locks with:

```bash
pip install pip-tools
pip-compile requirements.in
pip-compile requirements-dev.in
```

### Tests
```bash
pytest
```

### Indexing
- Streaming file readers
- No `f.read()` on large files
- Safe for multi‑GB corpora

---

## 🧠 Philosophy

Large Language Models are powerful — but **stateless**.

Engram MCP exists to give agents:
- Memory that persists
- Knowledge they can revisit
- Context they can build upon

This is not convenience infrastructure.

This is **cognition infrastructure**.

---

## 🛣️ Roadmap

- Binary vector quantization (32× memory reduction)
- Tantivy / Lucene backend option
- Namespace isolation & multi‑tenant memory
- Hot‑swappable embedding models
- Cross‑agent shared memory graphs

---

## 🤝 Contributing

Contributions are welcome.

If you are interested in:
- Retrieval systems
- Agent architectures
- Local‑first AI
- High‑performance Python

You’ll feel at home here.

---

## 📜 License

MIT License.

Build agents that remember.
