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

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourname/engram-mcp.git
cd engram-mcp
```

---

### 2️⃣ Configure allowed paths
```bash
cp engram_mcp.yaml.example engram_mcp.yaml
```

Edit **`allowed_roots`**:

```yaml
allowed_roots:
  - /Users/you/Documents
  - /Users/you/Projects
```

> ⚠️ Engram **will refuse** to index anything outside these paths.

---

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

All dependencies are resolved **at install time**.  
No runtime downloads. No surprises.

---

### 4️⃣ Run the server
```bash
python server.py
```

Engram MCP is now available to MCP‑compatible clients.

---

## 🧠 MCP Tools

### `index_project`
Indexes a directory (must be within `allowed_roots`).

```json
{
  "path": "/Users/you/Projects/my_repo"
}
```

---

### `search_memory`
Hybrid semantic + lexical search.

```json
{
  "query": "async sqlite performance issues",
  "limit": 10
}
```

---

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
