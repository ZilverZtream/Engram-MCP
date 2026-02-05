from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import yaml
from .security import PathContext
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, field_validator, Field


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
    db_path: str = field(default_factory=lambda: os.path.join(_default_data_dir(), "memory.db"))
    index_dir: str = field(
        default_factory=lambda: os.path.join(_default_data_dir(), "indexes")
    )  # where *.index files are stored

    # Security
    allowed_roots: List[str] = field(default_factory=list)

    # Indexing behavior
    ignore_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_IGNORE_PATTERNS))
    max_file_size_mb: int = 50
    chunk_size_tokens: int = 200
    overlap_tokens: int = 30
    tokenizer_path: str = field(default_factory=lambda: os.getenv("ENGRAM_TOKENIZER_PATH", ""))

    # Performance
    query_timeout_s: int = 60
    index_timeout_s: int = 3600
    embedding_batch_size: int = 128
    embedding_cache_ttl_s: int = 0
    embedding_cache_max_rows: int = 0

    # Search
    fts_top_k: int = 200
    vector_top_k: int = 200
    return_k: int = 10
    enable_mmr: bool = True
    mmr_lambda: float = 0.7
    enable_numba: bool = False
    max_query_chars: int = 4096
    max_query_tokens: int = 256
    search_cache_ttl_s: int = 300
    search_cache_max_items: int = 512
    vector_backend: str = "auto"

    # Embeddings
    model_name_text: str = "paraphrase-multilingual-MiniLM-L12-v2"
    model_name_code: str = "all-MiniLM-L6-v2"

    # Dreaming
    enable_dreaming: bool = False
    dream_model_name: str = "gpt-4o-mini"
    dream_threshold: float = 0.8

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
    openai_api_key: SecretStr = SecretStr("")
    openai_embedding_model: str = "text-embedding-3-small"


class AllowedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Storage
    db_path: str = Field(default_factory=lambda: os.path.join(_default_data_dir(), "memory.db"))
    index_dir: str = Field(default_factory=lambda: os.path.join(_default_data_dir(), "indexes"))

    # Security
    allowed_roots: List[str] = []

    # Indexing behavior
    ignore_patterns: List[str] = list(DEFAULT_IGNORE_PATTERNS)
    max_file_size_mb: int = 50
    chunk_size_tokens: int = 200
    overlap_tokens: int = 30
    tokenizer_path: str = Field(default_factory=lambda: os.getenv("ENGRAM_TOKENIZER_PATH", ""))

    # Performance
    query_timeout_s: int = 60
    index_timeout_s: int = 3600
    embedding_batch_size: int = 128
    embedding_cache_ttl_s: int = 0
    embedding_cache_max_rows: int = 0

    # Search
    fts_top_k: int = 200
    vector_top_k: int = 200
    return_k: int = 10
    enable_mmr: bool = True
    mmr_lambda: float = 0.7
    enable_numba: bool = False
    max_query_chars: int = 4096
    max_query_tokens: int = 256
    search_cache_ttl_s: int = 300
    search_cache_max_items: int = 512
    vector_backend: str = "auto"

    # Embeddings
    model_name_text: str = "paraphrase-multilingual-MiniLM-L12-v2"
    model_name_code: str = "all-MiniLM-L6-v2"

    # Dreaming
    enable_dreaming: bool = False
    dream_model_name: str = "gpt-4o-mini"
    dream_threshold: float = 0.8

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
    openai_api_key: SecretStr = SecretStr("")
    openai_embedding_model: str = "text-embedding-3-small"

    @field_validator("overlap_tokens")
    @classmethod
    def validate_overlap(cls, value: int, info):  # type: ignore[override]
        chunk_size = info.data.get("chunk_size_tokens", 200)
        if int(value) < 0:
            raise ValueError("overlap_tokens must be greater than or equal to 0")
        if int(value) >= int(chunk_size):
            raise ValueError("overlap_tokens must be less than chunk_size_tokens")
        return value

    @field_validator("vector_backend")
    @classmethod
    def validate_vector_backend(cls, value: str) -> str:
        allowed = {"auto", "fts", "faiss_cpu", "faiss_gpu"}
        normalized = str(value or "").lower()
        if normalized not in allowed:
            raise ValueError(f"vector_backend must be one of {sorted(allowed)}")
        return normalized

    @field_validator("embedding_batch_size")
    @classmethod
    def validate_embedding_batch_size(cls, value: int) -> int:
        min_value = 1
        max_value = 1000
        if not min_value <= int(value) <= max_value:
            raise ValueError(
                f"embedding_batch_size must be between {min_value} and {max_value}"
            )
        return value


def load_config(path: Optional[str] = None) -> EngramConfig:
    """Load config from YAML.

    Default path: ~/.config/engram/engram_mcp.yaml (unless ENGRAM_CONFIG_PATH is set
    within that directory)

    Example:

        db_path: ~/.local/share/engram/memory.db
        index_dir: ~/.local/share/engram/indexes
        allowed_roots:
          - /Users/you/Documents
    """

    if path is None:
        # 1) Honour an explicit environment variable *only* if it lives
        #    under the user-scoped config directory.
        # 2) Fall back to the user-scoped config directory so that an
        #    attacker who drops a crafted engram_mcp.yaml into a shared
        #    /tmp or a cloned repo cannot silently alter allowed_roots.
        env_path = os.environ.get("ENGRAM_CONFIG_PATH")
        user_config_dir = os.path.realpath(os.path.join(os.path.expanduser("~"), ".config", "engram"))
        if env_path:
            env_real = os.path.realpath(env_path)
            if env_real == user_config_dir or env_real.startswith(user_config_dir + os.sep):
                path = env_path
            else:
                logging.warning(
                    "Ignoring ENGRAM_CONFIG_PATH outside %s; using default user config path.",
                    user_config_dir,
                )
                env_path = None
        if not env_path:
            path = os.path.join(user_config_dir, "engram_mcp.yaml")

    cfg = EngramConfig()
    config_root = os.path.dirname(os.path.abspath(path)) or os.getcwd()
    path_context = PathContext([config_root])
    if not path_context.exists(path):
        _ensure_storage_dirs(cfg)
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

    _ensure_storage_dirs(cfg)

    # Push the resolved tokenizer path into the chunking module so it does
    # not need to reach for an env-var at runtime.
    from . import chunking as _chunking
    _chunking.configure_tokenizer_path(cfg.tokenizer_path)

    return cfg


def _default_data_dir() -> str:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "engram")
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "engram")
    return os.path.join(os.path.expanduser("~"), ".local", "share", "engram")


def _ensure_dir(path: str, *, mode: int = 0o700) -> None:
    os.makedirs(path, mode=mode, exist_ok=True)
    if os.name != "nt":
        try:
            os.chmod(path, mode)
        except OSError:
            logging.warning("Failed to chmod %s to %s", path, oct(mode), exc_info=True)


def _ensure_storage_dirs(cfg: EngramConfig) -> None:
    db_dir = os.path.dirname(os.path.abspath(cfg.db_path))
    _ensure_dir(db_dir, mode=0o700)
    _ensure_dir(os.path.abspath(cfg.index_dir), mode=0o700)
