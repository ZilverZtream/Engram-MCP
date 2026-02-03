from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Iterator


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    content: str
    token_count: int
    metadata: Dict[str, object]


@lru_cache(maxsize=50000)
def _approx_token_count(text: str) -> int:
    # More accurate token estimation using character-to-token ratio
    # This works better for code and non-English text than whitespace splitting
    # Conservative estimate: ~4 characters per token (accounts for punctuation, operators, etc.)
    # This prevents chunks from exceeding model context windows due to underestimation
    char_count = len(text)
    if char_count == 0:
        return 1

    # Use character-based estimate as primary method
    char_based = max(1, char_count // 4)

    # Also count words as a sanity check (some texts are very sparse)
    word_count = len(text.split())

    # Return the larger of the two estimates to be conservative
    return max(char_based, word_count)


def token_count(text: str) -> int:
    return _approx_token_count(text)


def make_chunk_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()


def _word_iterator(text: str) -> Iterator[str]:
    """Memory-efficient word iterator using regex."""
    # Use finditer to yield words without loading them all into memory
    for match in re.finditer(r'\S+', text):
        yield match.group()


def chunk_text(
    *,
    text: str,
    base_id: str,
    meta: Dict[str, object],
    target_tokens: int,
    overlap_tokens: int,
) -> List[Chunk]:
    """Sliding-window chunking over approximate tokens with memory-efficient processing."""

    if not text.strip():
        return []

    target_tokens = max(50, int(target_tokens))
    overlap_tokens = max(0, int(overlap_tokens))

    # For very large files (>10MB), use streaming approach
    # Otherwise, use the existing approach for better performance on normal files
    if len(text) > 10 * 1024 * 1024:
        # Memory-efficient approach for large files
        words_iter = _word_iterator(text)
        chunks: List[Chunk] = []
        idx = 0
        window: List[str] = []
        word_position = 0

        for word in words_iter:
            window.append(word)
            if len(window) >= target_tokens:
                content = " ".join(window)
                cid = make_chunk_id(base_id, str(idx))
                tc = len(window)
                start_pos = word_position - len(window) + 1
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        content=content,
                        token_count=tc,
                        metadata={**meta, "chunk_index": idx, "word_range": f"{start_pos}-{word_position}"},
                    )
                )
                idx += 1
                # Keep overlap words
                window = window[-overlap_tokens:] if overlap_tokens > 0 else []
            word_position += 1

        # Add remaining words as final chunk
        if window:
            content = " ".join(window)
            cid = make_chunk_id(base_id, str(idx))
            tc = len(window)
            start_pos = word_position - len(window)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    content=content,
                    token_count=tc,
                    metadata={**meta, "chunk_index": idx, "word_range": f"{start_pos}-{word_position-1}"},
                )
            )

        return chunks
    else:
        # Original approach for normal-sized files (better performance)
        words = text.split()
        if not words:
            return []

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
