from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Iterator

from tokenizers import Tokenizer


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    content: str
    token_count: int
    metadata: Dict[str, object]


@lru_cache(maxsize=1)
def _load_tokenizer() -> Tokenizer:
    tokenizer_path = os.getenv("ENGRAM_TOKENIZER_PATH")
    if not tokenizer_path:
        raise RuntimeError(
            "ENGRAM_TOKENIZER_PATH must be set to a local tokenizer.json; "
            "network downloads are disabled for safety."
        )
    if not os.path.exists(tokenizer_path):
        raise RuntimeError(f"Tokenizer not found at ENGRAM_TOKENIZER_PATH: {tokenizer_path}")
    return Tokenizer.from_file(tokenizer_path)


def token_count(text: str) -> int:
    if not text:
        return 0
    tokenizer = _load_tokenizer()
    return len(tokenizer.encode(text).ids)


def make_chunk_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()


def chunk_text(
    *,
    text: str,
    base_id: str,
    meta: Dict[str, object],
    target_tokens: int,
    overlap_tokens: int,
) -> List[Chunk]:
    """Sliding-window chunking over tokenizer tokens."""

    if not text.strip():
        return []

    target_tokens = max(50, int(target_tokens))
    overlap_tokens = max(0, int(overlap_tokens))
    tokenizer = _load_tokenizer()
    encoding = tokenizer.encode(text)
    offsets = encoding.offsets
    if not offsets:
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    total_tokens = len(offsets)

    while start < total_tokens:
        end = min(total_tokens, start + target_tokens)
        start_offset = offsets[start][0]
        end_offset = offsets[end - 1][1] if end > start else offsets[start][1]
        content = text[start_offset:end_offset].strip()
        if content:
            cid = make_chunk_id(base_id, str(idx), content)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    content=content,
                    token_count=end - start,
                    metadata={**meta, "chunk_index": idx, "token_range": f"{start}-{end-1}"},
                )
            )
            idx += 1
        if end == total_tokens:
            break
        start = max(0, end - overlap_tokens)

    return chunks


def chunk_lines(
    *,
    lines: Iterable[str],
    base_id: str,
    meta: Dict[str, object],
    target_tokens: int,
    overlap_tokens: int,
) -> List[Chunk]:
    text = "\n".join(lines)
    return chunk_text(
        text=text,
        base_id=base_id,
        meta=meta,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )
