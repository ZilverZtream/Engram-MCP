from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


DEFAULT_IGNORE_PATTERNS: List[str] = [
    "**/.git/**",
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/venv/**",
    "**/*.log",
    "**/.DS_Store",
    "**/Thumbs.db",
    "**/target/**",
    "**/build/**",
    "**/dist/**",
    "**/.vscode/**",
    "**/.idea/**",
]


@dataclass
class EngramConfig:
    # Storage
    db_path: str = "memory.db"
    index_dir: str = "."  # where *.index files are stored

    # Security
    allowed_roots: List[str] = field(default_factory=list)

    # Indexing behavior
    ignore_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_IGNORE_PATTERNS))
    max_file_size_mb: int = 200
    chunk_size_tokens: int = 200
    overlap_tokens: int = 30

    # Performance
    query_timeout_s: int = 60
    index_timeout_s: int = 3600
    embedding_batch_size: int = 128
    embedding_cache_ttl_s: int = 0

    # Search
    fts_top_k: int = 200
    vector_top_k: int = 200
    return_k: int = 10
    enable_mmr: bool = True
    mmr_lambda: float = 0.7

    # Embeddings
    model_name_text: str = "paraphrase-multilingual-MiniLM-L12-v2"
    model_name_code: str = "all-MiniLM-L6-v2"

    # Optional: keep GPU embeddings in-process (recommended)
    prefer_thread_for_cuda: bool = True

    # FAISS IVF+PQ settings
    faiss_nlist: int = 100
    faiss_m: int = 8
    faiss_nbits: int = 8
    faiss_nprobe: int = 16

    # Sharding settings
    shard_chunk_threshold: int = 1_000_000
    shard_size: int = 250_000


def load_config(path: Optional[str] = None) -> EngramConfig:
    """Load config from YAML.

    Default path: ./engram_mcp.yaml

    Example:

        db_path: memory.db
        index_dir: .
        allowed_roots:
          - /Users/you/Documents
    """

    if path is None:
        path = os.path.join(os.getcwd(), "engram_mcp.yaml")

    cfg = EngramConfig()
    if not os.path.exists(path):
        return cfg

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Merge known fields only (avoid surprises)
    for k, v in data.items():
        if hasattr(cfg, k):
            # Special handling for ignore_patterns: merge with defaults instead of replacing
            if k == "ignore_patterns" and isinstance(v, list):
                # Combine user patterns with defaults (user patterns first for priority)
                cfg.ignore_patterns = list(v) + [p for p in DEFAULT_IGNORE_PATTERNS if p not in v]
            else:
                setattr(cfg, k, v)

    # Normalize allowed roots (resolve symlinks for security)
    cfg.allowed_roots = [os.path.realpath(p) for p in (cfg.allowed_roots or [])]
    cfg.index_dir = os.path.abspath(cfg.index_dir)
    cfg.db_path = os.path.abspath(cfg.db_path)
    if int(cfg.overlap_tokens) >= int(cfg.chunk_size_tokens):
        logging.warning(
            "overlap_tokens (%s) must be less than chunk_size_tokens (%s); adjusting overlap.",
            cfg.overlap_tokens,
            cfg.chunk_size_tokens,
        )
        cfg.overlap_tokens = max(0, int(cfg.chunk_size_tokens) - 1)

    return cfg
