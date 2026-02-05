from __future__ import annotations

import hashlib
import logging
import os
import re
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


# Injected once at startup by config.load_config via configure_tokenizer_path.
# None means "not yet configured"; empty string means "no tokenizer available".
_TOKENIZER_PATH: str | None = None


def configure_tokenizer_path(path: str) -> None:
    """Inject the tokenizer path from EngramConfig at startup.

    Must be called before any token-counting or chunking operation.
    Clears the cached tokenizer so the next call to _load_tokenizer
    picks up the new path.
    """
    global _TOKENIZER_PATH
    _TOKENIZER_PATH = path
    _load_tokenizer.cache_clear()


@lru_cache(maxsize=1)
def _load_tokenizer() -> Tokenizer | None:
    tokenizer_path = _TOKENIZER_PATH if _TOKENIZER_PATH is not None else ""
    if not tokenizer_path:
        logging.warning(
            "tokenizer_path not configured; using regex token offsets to avoid network downloads."
        )
        return None
    if not os.path.exists(tokenizer_path):
        logging.warning(
            "Tokenizer not found at tokenizer_path=%s; using regex token offsets.",
            tokenizer_path,
        )
        return None
    try:
        return Tokenizer.from_file(tokenizer_path)
    except Exception:
        logging.warning(
            "Failed to load tokenizer from %s; using regex token offsets.",
            tokenizer_path,
            exc_info=True,
        )
        return None


def _regex_offsets(text: str) -> List[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _regex_token_count(text: str) -> int:
    """Count whitespace-delimited tokens without materialising offset tuples.

    O(1) memory regardless of token count — avoids the 50M-tuple spike
    that _regex_offsets would produce on a 100 MB file with dense tokens.
    """
    return sum(1 for _ in re.finditer(r"\S+", text))


def _get_offsets(text: str) -> List[tuple[int, int]]:
    tokenizer = _load_tokenizer()
    if tokenizer is None:
        return _regex_offsets(text)
    encoding = tokenizer.encode(text)
    return encoding.offsets


def token_count(text: str) -> int:
    if not text:
        return 0
    tokenizer = _load_tokenizer()
    if tokenizer is None:
        return _regex_token_count(text)
    return len(tokenizer.encode(text).tokens)


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
    offsets = _get_offsets(text)
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

    max_chars = max(200_000, target_tokens * 20)
    # Accumulate incoming parts in a list; join only at flush points.
    # This gives O(N) append cost instead of O(N²) repeated str concatenation.
    pending: List[str] = []
    pending_len = 0
    chunks: List[Chunk] = []
    idx_offset = 0

    def _tail_text(text: str) -> str:
        if overlap_tokens <= 0 or not text:
            return ""
        offsets = _get_offsets(text)
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
        # Append the raw text exactly as received — do NOT insert artificial
        # newlines.  The caller (read_text_streaming / _iter_lines) yields
        # text that already contains its own line endings; injecting a "\n"
        # at every chunk boundary corrupts content at 1 MB boundaries.
        pending.append(part)
        pending_len += len(part)
        if pending_len >= max_chars:
            buffer = "".join(pending)
            _flush(buffer)
            tail = _tail_text(buffer)
            pending = [tail] if tail else []
            pending_len = len(tail) if tail else 0

    if pending:
        _flush("".join(pending))

    return chunks
