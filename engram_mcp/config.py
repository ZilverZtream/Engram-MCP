from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml
from .security import PathContext
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


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

    # Embedding backend selection
    embedding_backend: str = "sentence_transformers"  # "sentence_transformers" | "ollama" | "openai"
    ollama_model: str = "nomic-embed-text"
    ollama_url: str = "http://localhost:11434"
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"


class AllowedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Storage
    db_path: str = "memory.db"
    index_dir: str = "."

    # Security
    allowed_roots: List[str] = []

    # Indexing behavior
    ignore_patterns: List[str] = list(DEFAULT_IGNORE_PATTERNS)
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
    prefer_thread_for_cuda: bool = True

    # FAISS IVF+PQ settings
    faiss_nlist: int = 100
    faiss_m: int = 8
    faiss_nbits: int = 8
    faiss_nprobe: int = 16

    # Sharding settings
    shard_chunk_threshold: int = 1_000_000
    shard_size: int = 250_000

    # Embedding backend selection
    embedding_backend: str = "sentence_transformers"
    ollama_model: str = "nomic-embed-text"
    ollama_url: str = "http://localhost:11434"
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

    @field_validator("overlap_tokens")
    @classmethod
    def validate_overlap(cls, value: int, info):  # type: ignore[override]
        chunk_size = info.data.get("chunk_size_tokens", 200)
        if int(value) >= int(chunk_size):
            raise ValueError("overlap_tokens must be less than chunk_size_tokens")
        return value


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
        # 1) Honour an explicit environment variable.
        # 2) Fall back to the user-scoped config directory so that an
        #    attacker who drops a crafted engram_mcp.yaml into a shared
        #    /tmp or a cloned repo cannot silently alter allowed_roots.
        env_path = os.environ.get("ENGRAM_CONFIG_PATH")
        if env_path:
            path = env_path
        else:
            path = os.path.join(
                os.path.expanduser("~"), ".config", "engram", "engram_mcp.yaml"
            )

    cfg = EngramConfig()
    config_root = os.path.dirname(os.path.abspath(path)) or os.getcwd()
    path_context = PathContext([config_root])
    if not path_context.exists(path):
        return cfg
    with path_context.open_file(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if isinstance(data, dict) and isinstance(data.get("ignore_patterns"), list):
        ignore_patterns = list(data["ignore_patterns"])
        data["ignore_patterns"] = ignore_patterns + [
            p for p in DEFAULT_IGNORE_PATTERNS if p not in ignore_patterns
        ]

    try:
        validated = AllowedConfig.model_validate(data or {})
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc

    cfg = EngramConfig(**validated.model_dump())

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
