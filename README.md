# Engram MCP

<p align="center">
  <strong>A local, sovereign, hybrid memory system for MCP-compatible agents</strong>
</p>

<p align="center">
  Semantic search • Lexical search • Persistent memory • REM-style dreaming • Secure by default
</p>

---

## 🧠 What is Engram MCP?

**Engram MCP** is a high‑performance, open‑source memory server that gives AI agents *long‑term, searchable memory* with hybrid retrieval and optional REM‑style dreaming.

In neuroscience, an **engram** is the physical trace of a memory in the brain. Engram MCP brings that concept to AI systems: durable, auditable, local memory that scales from small projects to personal knowledge bases.

Engram is **not** a chat log.
It is **not** a volatile cache.
It is a **cognitive substrate** for agents.

---

## ✨ Core Capabilities

### 🔍 Hybrid Search (Semantic + Lexical)
- **Vector similarity search** via FAISS (optional)
- **Lexical BM25 search** via SQLite FTS5
- **Reciprocal Rank Fusion (RRF)** for best‑of‑both‑worlds ranking
- **Maximal Marginal Relevance (MMR)** re‑ranking for diversity

### 🧬 REM‑Style Dreaming (Insight Generation)
- **Dream cycles** connect co‑occurring chunks into new insights
- **REM trigger tool** (`trigger_rem_cycle`) queues a dream job
- Auto‑triggered dreaming after search activity when enabled
- Insights are stored as **virtual files** (`vfs://insights/...`) and searchable like any other memory

### 🧱 Persistent, Incremental Memory
- Chunk‑level hashing & deduplication
- Streaming file readers (no `f.read()` on large files)
- Automatic re‑indexing of changed files
- Embedding cache to avoid re‑embedding identical text

### ⚡ Non‑Blocking Architecture
- Async SQLite via `aiosqlite`
- Embeddings offloaded to worker pools (process or thread)
- Long indexing jobs do **not** block search or other tools

### 🔐 Security‑First by Default
- Explicit **path whitelisting** (`allowed_roots`)
- Protection against path traversal & symlink escape
- Owner‑only permissions for DB and index artifacts

### 🧩 MCP‑Native & Production‑Ready
- Built on the **official MCP SDK / FastMCP**
- Background job queue with cancellation and retention
- Index integrity checks (UUID + checksum) for FAISS artifacts

---

## 🏗️ Architecture Overview

```
server.py                # MCP entrypoint & tool definitions
engram_mcp/
├── config.py            # Configuration + validation
├── security.py          # Path guards + project validation
├── db.py                # SQLite schema + async data access
├── parsing.py           # File discovery + supported formats
├── chunking.py          # Token-aware chunking
├── embeddings.py        # Embedding backends + worker pools
├── indexing.py          # Indexing pipeline + FAISS sharding
├── search.py            # Hybrid search, RRF, MMR
├── dreaming.py          # Dream candidate discovery
├── generation.py        # Insight generation (transformers)
├── jobs.py              # Background job manager
└── locks.py             # Project locks (read/write)
```

---

## ✅ Supported Content Types

Engram indexes text and documents with streaming readers and size caps:

- **Text/code**: `.txt`, `.md`, `.py`, `.js`, `.ts`, `.json`, `.yaml`, `.toml`, `.sql`, `.html`, `.css`, `.sh`, and more
- **Documents**: `.pdf`, `.docx`

Ignored by default (configurable): `.git`, `node_modules`, virtual envs, IDE folders, build outputs, logs, etc.

---

## 🚀 Quick Start

### 1️⃣ Install Engram MCP

**pipx (recommended for CLI usage)**
```bash
pipx install engram-mcp
engram-mcp
```

Enable vector search with FAISS CPU (optional):
```bash
pipx install "engram-mcp[cpu]"
```

Enable FAISS GPU builds (optional, Linux‑first support):
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

SentenceTransformers models download on first use unless you pre-download them (for air‑gapped setups, cache the models or point `model_name_*` to a local path). No surprises once the model cache is in place.

Engram MCP is now available to MCP‑compatible clients.

---

## ⚙️ Configuration Highlights

You can control Engram behavior via `engram_mcp.yaml`:

```yaml
# Storage
# db_path: ~/.local/share/engram/memory.db
# index_dir: ~/.local/share/engram/indexes

# Search
vector_backend: auto      # auto | fts | faiss_cpu | faiss_gpu
fts_top_k: 200
vector_top_k: 200
return_k: 10
enable_mmr: true
mmr_lambda: 0.7

# Indexing
chunk_size_tokens: 200
overlap_tokens: 30
max_file_size_mb: 50
ignore_patterns:
  - "**/.git/**"
  - "**/node_modules/**"

# Embeddings
embedding_backend: sentence_transformers  # sentence_transformers | ollama | openai
model_name_text: paraphrase-multilingual-MiniLM-L12-v2
model_name_code: all-MiniLM-L6-v2
ollama_model: nomic-embed-text
ollama_url: http://localhost:11434
openai_embedding_model: text-embedding-3-small

# Dreaming / REM
enable_dreaming: false
dream_model_name: gpt-4o-mini
dream_threshold: 0.8

# Performance
enable_numba: false
search_cache_ttl_s: 300
search_cache_max_items: 512
embedding_cache_ttl_s: 0
embedding_cache_max_rows: 0
```

### Environment variables
- `ENGRAM_CONFIG_PATH`: override config path (must still live under the config directory)
- `ENGRAM_TOKENIZER_PATH`: optional local tokenizer file for accurate token counting
- `ENGRAM_GENERATION_MODEL`: transformers model for insight generation (default: `gpt2`)
- `ENGRAM_GENERATION_DEVICE`: `cpu` or `cuda` for generation (default: `cpu`)
- `ENGRAM_DREAM_CONTEXT_WINDOW`: max dream prompt window (default: `2048`)

### Embedding backends
- **sentence_transformers**: local models (default) via `SentenceTransformer`
- **ollama**: local Ollama server at `ollama_url`
- **openai**: OpenAI‑compatible `/embeddings` endpoint (set `openai_api_key`)
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

### `update_project`
Updates an existing project (use `wait: false` to queue and poll job status).

```json
{
  "project_id": "my_repo_123",
  "wait": true
}
```

### `search_memory`
Hybrid semantic + lexical search with optional MMR (`fts_mode`: `strict`, `any`, or `phrase`).

```json
{
  "query": "async sqlite performance issues",
  "project_id": "my_repo_123",
  "max_results": 10,
  "fts_mode": "strict",
  "use_mmr": true
}
```

### `get_chunk`
Fetch a single chunk by ID.

```json
{
  "project_id": "my_repo_123",
  "chunk_id": "...",
  "include_content": true
}
```

### `project_info`
Returns metadata for a project (chunk counts, embedding model, shard info, etc.).

### `project_health`
Validates FAISS index integrity, UUIDs, and checksums.

### `repair_project`
Rebuilds FAISS indexes from DB state.

### `delete_project`
Deletes a project and all index artifacts.

### `list_projects`
Returns all indexed projects.

### `list_jobs` / `cancel_job`
Background job management for index, update, and dream tasks.

### `dream_project`
Queues an insight-generation cycle from search co‑occurrences.

```json
{
  "project_id": "my_repo_123",
  "wait": false,
  "max_pairs": 10
}
```

### `trigger_rem_cycle`
Queues a REM‑style dream cycle (alias to dream job creation).

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

### Storage defaults (privacy‑safe)
By default, Engram writes all state into a **user‑scoped data directory** (never the current working directory):

| OS | Data directory |
| --- | --- |
| Linux | `~/.local/share/engram/` |
| macOS | `~/Library/Application Support/engram/` |
| Windows | `%APPDATA%\\engram\\` |

The SQLite DB and FAISS index files are created with **owner‑only permissions**. Override `db_path`/`index_dir`
explicitly if you want storage in a custom location.

### Query limits
`search_memory` enforces `max_query_chars` and `max_query_tokens` from config (defaults: 4096 chars / 256 tokens)
to prevent runaway embedding costs.

---

## ⚙️ Performance Notes

### Embeddings
- Offloaded from the event loop
- CPU → `ProcessPoolExecutor`
- CUDA → thread‑safe execution (no fork hazards)
- Remote embedding backends supported (Ollama / OpenAI)

### Search & Ranking
- Vector search via FAISS (optional)
- Lexical search via SQLite FTS5
- RRF fusion + optional MMR re‑ranking
- Search cache for repeated queries

### Indexing at scale
- Chunk sharding for large projects (configurable thresholds)
- IVF‑based FAISS indexes with configurable `nlist` and `nprobe`
- Index artifacts stored with UUID + checksum for integrity checks

---

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
