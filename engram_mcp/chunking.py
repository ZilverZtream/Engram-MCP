from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    content: str
    token_count: int
    metadata: Dict[str, object]


@lru_cache(maxsize=50000)
def _approx_token_count(text: str) -> int:
    # Lightweight heuristic to avoid pulling in large tokenizers.
    # Roughly: 1 token ~= 0.75 words in English; we just use words as tokens.
    return max(1, len(text.split()))


def token_count(text: str) -> int:
    return _approx_token_count(text)


def make_chunk_id(*parts: str) -> str:
    h = hashlib.sha1()
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
    """Sliding-window chunking over approximate tokens."""

    words = text.split()
    if not words:
        return []

    target_tokens = max(50, int(target_tokens))
    overlap_tokens = max(0, int(overlap_tokens))

    chunks: List[Chunk] = []
    start = 0
    idx = 0

    while start < len(words):
        end = min(len(words), start + target_tokens)
        window = words[start:end]
        content = " ".join(window)
        cid = make_chunk_id(base_id, str(idx))
        tc = len(window)
        chunks.append(
            Chunk(
                chunk_id=cid,
                content=content,
                token_count=tc,
                metadata={**meta, "chunk_index": idx, "word_range": f"{start}-{end-1}"},
            )
        )
        idx += 1
        if end == len(words):
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
