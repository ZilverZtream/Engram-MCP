from __future__ import annotations

import hashlib
import logging
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
        logging.warning("ENGRAM_TOKENIZER_PATH not set; falling back to bert-base-uncased tokenizer.")
        return Tokenizer.from_pretrained("bert-base-uncased")
    if not os.path.exists(tokenizer_path):
        logging.warning(
            "Tokenizer not found at ENGRAM_TOKENIZER_PATH=%s; falling back to bert-base-uncased tokenizer.",
            tokenizer_path,
        )
        return Tokenizer.from_pretrained("bert-base-uncased")
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
    if overlap_tokens >= target_tokens:
        overlap_tokens = max(0, target_tokens - 1)
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
        new_start = max(0, end - overlap_tokens)
        if new_start <= start:
            break
        start = new_start

    return chunks


def chunk_lines(
    *,
    lines: Iterable[str],
    base_id: str,
    meta: Dict[str, object],
    target_tokens: int,
    overlap_tokens: int,
) -> List[Chunk]:
    target_tokens = max(50, int(target_tokens))
    overlap_tokens = max(0, int(overlap_tokens))
    if overlap_tokens >= target_tokens:
        overlap_tokens = max(0, target_tokens - 1)

    tokenizer = _load_tokenizer()
    max_chars = max(200_000, target_tokens * 20)
    buffer = ""
    chunks: List[Chunk] = []
    idx_offset = 0

    def _tail_text(text: str) -> str:
        if overlap_tokens <= 0 or not text:
            return ""
        encoding = tokenizer.encode(text)
        offsets = encoding.offsets
        if not offsets:
            return ""
        if len(offsets) <= overlap_tokens:
            return text
        start = offsets[-overlap_tokens][0]
        return text[start:]

    def _flush(text: str) -> None:
        nonlocal idx_offset
        if not text.strip():
            return
        batch = chunk_text(
            text=text,
            base_id=base_id,
            meta=meta,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        if batch:
            for c in batch:
                c.metadata["chunk_index"] = int(c.metadata.get("chunk_index", 0)) + idx_offset
            idx_offset += len(batch)
            chunks.extend(batch)

    for part in lines:
        if part is None:
            continue
        buffer += part
        if not buffer.endswith("\n"):
            buffer += "\n"
        if len(buffer) >= max_chars:
            _flush(buffer)
            buffer = _tail_text(buffer)

    if buffer:
        _flush(buffer)

    return chunks
