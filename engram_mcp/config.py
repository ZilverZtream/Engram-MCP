from __future__ import annotations

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
            setattr(cfg, k, v)

    # Normalize allowed roots
    cfg.allowed_roots = [os.path.abspath(p) for p in (cfg.allowed_roots or [])]
    cfg.index_dir = os.path.abspath(cfg.index_dir)
    cfg.db_path = os.path.abspath(cfg.db_path)

    return cfg
