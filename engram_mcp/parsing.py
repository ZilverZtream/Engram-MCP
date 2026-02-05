from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .chunking import Chunk, chunk_lines, chunk_text, make_chunk_id, token_count
from .security import PathContext


SUPPORTED_TEXT_EXTS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".cs", ".java", ".go", ".rs", ".cpp", ".c", ".h", ".hpp", ".sql", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".html", ".css", ".scss", ".xml", ".sh", ".bash", ".ps1",
}

SUPPORTED_DOC_EXTS = {".pdf", ".docx"}


def _is_ignored(path: str, ignore_patterns: List[str]) -> bool:
    basename = os.path.basename(path)
    for pat in ignore_patterns:
        if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(basename, pat):
            return True
        # fnmatch does not honour the "any directory depth" semantics of **;
        # strip leading **/ prefixes so that patterns like **/*.log also
        # match files directly at the root (where the relative path has no
        # directory component).
        stripped = pat
        while stripped.startswith("**/"):
            stripped = stripped[3:]
        if stripped != pat and fnmatch.fnmatch(basename, stripped):
            return True
    return False


def iter_files(path_context: PathContext, root: str, ignore_patterns: List[str]) -> Iterator[Path]:
    root_path = path_context.resolve_path(root)
    for p in path_context.iter_files(root_path):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root_path))
        if _is_ignored(rel, ignore_patterns):
            continue
        yield p


def read_text_streaming(
    path_context: PathContext,
    path: Path,
    root: Path,
    max_bytes: int,
    *,
    chunk_size: int = 1024 * 1024,
) -> Iterable[str]:
    # Avoid reading enormous files at once.
    size = path_context.stat(path).st_size
    if size > max_bytes:
        raise ValueError(f"File too large: {path} ({size} bytes)")
    with ExitStack() as stack:
        handle = stack.enter_context(
            path_context.open_file(path, "r", encoding="utf-8", errors="ignore")
        )
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            yield chunk


def hash_file(path_context: PathContext, path: Path, root: Path, max_bytes: int) -> str:
    size = path_context.stat(path).st_size
    if size > max_bytes:
        raise ValueError(f"File too large: {path} ({size} bytes)")
    h = hashlib.sha256()
    with ExitStack() as stack:
        handle = stack.enter_context(path_context.open_file(path, "rb"))
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_file_to_chunks(
    *,
    path_context: PathContext,
    path: Path,
    root: str,
    project_type: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    max_file_size_mb: int,
) -> List[Chunk]:
    root_path = path_context.resolve_path(root)
    if root_path not in path.parents and path != root_path:
        raise ValueError(f"Path escapes root: {path}")
    ext = path.suffix.lower()
    rel = str(path.relative_to(root))
    base_meta: Dict[str, object] = {
        "item_name": rel,
        "path": str(path),
        "ext": ext,
        "type": "file",
        "project_type": project_type,
    }

    max_bytes = int(max_file_size_mb) * 1024 * 1024

    if ext in SUPPORTED_TEXT_EXTS:
        base_id = make_chunk_id(str(path))
        return chunk_lines(
            lines=read_text_streaming(path_context, path, root=Path(root), max_bytes=max_bytes),
            base_id=base_id,
            meta=base_meta,
            target_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )

    if ext == ".pdf":
        # Streaming: page by page
        from pypdf import PdfReader

        # Check PDF file size before loading (prevent OOM/DoS)
        size = path_context.stat(path).st_size
        if size > max_bytes:
            raise ValueError(f"PDF file too large: {path} ({size} bytes)")

        max_text_chars = max_bytes * 4
        with ExitStack() as stack:
            handle = stack.enter_context(path_context.open_file(path, "rb"))
            reader = PdfReader(handle)
        chunks: List[Chunk] = []
        base_id = make_chunk_id(str(path))
        total_chars = 0
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_len = len(page_text)
            if total_chars + page_len > max_text_chars:
                raise ValueError(f"PDF extracted text too large: {path}")
            total_chars += page_len
            if not page_text.strip():
                continue
            page_meta = {**base_meta, "type": "pdf_page", "page": i + 1}
            # Keep pages as their own units, then chunk if needed
            if token_count(page_text) <= chunk_size_tokens * 2:
                cid = make_chunk_id(base_id, f"page:{i+1}", page_text)
                chunks.append(Chunk(cid, page_text, token_count(page_text), page_meta))
            else:
                chunks.extend(
                    chunk_text(
                        text=page_text,
                        base_id=make_chunk_id(base_id, f"page:{i+1}"),
                        meta=page_meta,
                        target_tokens=chunk_size_tokens,
                        overlap_tokens=overlap_tokens,
                    )
                )
        return chunks

    if ext == ".docx":
        from docx import Document

        with ExitStack() as stack:
            handle = stack.enter_context(path_context.open_file(path, "rb"))
            doc = Document(handle)
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        text = "\n".join(paras)
        base_id = make_chunk_id(str(path))
        return chunk_text(
            text=text,
            base_id=base_id,
            meta={**base_meta, "type": "docx"},
            target_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )

    return []


def hash_and_parse_file(
    *,
    path_context: PathContext,
    path: Path,
    root: str,
    project_type: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    max_file_size_mb: int,
) -> "tuple[List[Chunk], str]":
    """Stream a file to return both its chunks and SHA-256 hash.

    Text files are parsed line-by-line while hashing the raw bytes to avoid
    large in-memory buffers. Binary formats (PDF/DOCX) fall back to a second
    read via their parsers after hashing, prioritizing memory safety over
    single-read I/O.
    """
    import codecs

    max_bytes = int(max_file_size_mb) * 1024 * 1024
    root_path = path_context.resolve_path(root)
    if root_path not in path.parents and path != root_path:
        raise ValueError(f"Path escapes root: {path}")

    size = path_context.stat(path).st_size
    if size > max_bytes:
        raise ValueError(f"File too large: {path} ({size} bytes)")

    ext = path.suffix.lower()
    rel = str(path.relative_to(root))
    base_meta: Dict[str, object] = {
        "item_name": rel,
        "path": str(path),
        "ext": ext,
        "type": "file",
        "project_type": project_type,
    }

    if ext in SUPPORTED_TEXT_EXTS:
        hasher = hashlib.sha256()
        decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        buffer = ""

        def _iter_lines() -> Iterator[str]:
            nonlocal buffer
            with path_context.open_file(path, "rb") as fh:
                while True:
                    chunk = fh.read(1024 * 1024)
                    if not chunk:
                        break
                    hasher.update(chunk)
                    text_chunk = decoder.decode(chunk)
                    if not text_chunk:
                        continue
                    buffer += text_chunk
                    lines = buffer.splitlines(keepends=True)
                    if lines:
                        if not buffer.endswith(("\n", "\r")):
                            buffer = lines.pop() if lines else buffer
                        else:
                            buffer = ""
                        for line in lines:
                            yield line
                tail = buffer + decoder.decode(b"", final=True)
                if tail:
                    for line in tail.splitlines(keepends=True):
                        yield line

        base_id = make_chunk_id(str(path))
        chunks = chunk_lines(
            lines=_iter_lines(),
            base_id=base_id,
            meta=base_meta,
            target_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )
        return chunks, hasher.hexdigest()

    if ext == ".pdf":
        from pypdf import PdfReader

        content_hash = hash_file(path_context, path, Path(root), max_bytes)
        max_text_chars = max_bytes * 4
        with path_context.open_file(path, "rb") as fh:
            reader = PdfReader(fh)
        chunks: List[Chunk] = []
        base_id = make_chunk_id(str(path))
        total_chars = 0
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_len = len(page_text)
            if total_chars + page_len > max_text_chars:
                raise ValueError(f"PDF extracted text too large: {path}")
            total_chars += page_len
            if not page_text.strip():
                continue
            page_meta = {**base_meta, "type": "pdf_page", "page": i + 1}
            if token_count(page_text) <= chunk_size_tokens * 2:
                cid = make_chunk_id(base_id, f"page:{i+1}", page_text)
                chunks.append(Chunk(cid, page_text, token_count(page_text), page_meta))
            else:
                chunks.extend(
                    chunk_text(
                        text=page_text,
                        base_id=make_chunk_id(base_id, f"page:{i+1}"),
                        meta=page_meta,
                        target_tokens=chunk_size_tokens,
                        overlap_tokens=overlap_tokens,
                    )
                )
        return chunks, content_hash

    if ext == ".docx":
        from docx import Document

        content_hash = hash_file(path_context, path, Path(root), max_bytes)
        with path_context.open_file(path, "rb") as fh:
            doc = Document(fh)
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        text = "\n".join(paras)
        base_id = make_chunk_id(str(path))
        chunks = chunk_text(
            text=text,
            base_id=base_id,
            meta={**base_meta, "type": "docx"},
            target_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )
        return chunks, content_hash

    content_hash = hash_file(path_context, path, Path(root), max_bytes)
    return [], content_hash


@dataclass(frozen=True)
class ParsedFile:
    file_path: str
    mtime_ns: int
    size_bytes: int
    content_hash: str
    chunks: List[Chunk]


