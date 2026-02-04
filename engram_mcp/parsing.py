from __future__ import annotations

import asyncio
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
    for pat in ignore_patterns:
        if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(os.path.basename(path), pat):
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


@dataclass(frozen=True)
class ParsedFile:
    file_path: str
    mtime_ns: int
    size_bytes: int
    content_hash: str
    chunks: List[Chunk]


@dataclass(frozen=True)
class ParseDirectoryResult:
    chunks: List[Chunk]
    file_count: int
    file_metadata: Dict[str, ParsedFile]
    changed_files: List[str]


async def parse_directory(
    *,
    path_context: PathContext,
    root: str,
    project_type: str,
    ignore_patterns: List[str],
    chunk_size_tokens: int,
    overlap_tokens: int,
    max_file_size_mb: int,
    existing_metadata: Optional[Dict[str, ParsedFile]] = None,
) -> ParseDirectoryResult:
    chunks: List[Chunk] = []
    file_metadata: Dict[str, ParsedFile] = {}
    changed_files: List[str] = []
    existing_metadata = existing_metadata or {}

    files: List[Path] = []
    for p in iter_files(path_context, root, ignore_patterns):
        ext = p.suffix.lower()
        if ext in SUPPORTED_TEXT_EXTS or ext in SUPPORTED_DOC_EXTS:
            files.append(p)

    file_count = len(files)
    sem = asyncio.Semaphore(os.cpu_count() or 4)

    async def parse_one(path: Path) -> Optional[ParsedFile]:
        rel = str(path.relative_to(root))
        stat = path_context.stat(path)
        existing = existing_metadata.get(rel)
        file_hash: Optional[str] = None
        if existing and existing.mtime_ns == stat.st_mtime_ns and existing.size_bytes == stat.st_size:
            try:
                file_hash = await asyncio.to_thread(
                    hash_file,
                    path_context,
                    path,
                    Path(root),
                    max_file_size_mb * 1024 * 1024,
                )
            except Exception:
                logging.warning("Failed to hash %s", rel, exc_info=True)
            if file_hash and existing.content_hash == file_hash:
                file_metadata[rel] = existing
                return None

        if file_hash is None:
            try:
                file_hash = await asyncio.to_thread(
                    hash_file,
                    path_context,
                    path,
                    Path(root),
                    max_file_size_mb * 1024 * 1024,
                )
            except Exception:
                logging.warning("Failed to hash %s", rel, exc_info=True)
                return None

        async with sem:
            try:
                file_chunks = await asyncio.wait_for(
                    asyncio.to_thread(
                        parse_file_to_chunks,
                        path_context=path_context,
                        path=path,
                        root=root,
                        project_type=project_type,
                        chunk_size_tokens=chunk_size_tokens,
                        overlap_tokens=overlap_tokens,
                        max_file_size_mb=max_file_size_mb,
                    ),
                    timeout=30,
                )
            except Exception:
                logging.warning("Failed to parse %s", rel, exc_info=True)
                return None

        changed_files.append(rel)
        parsed = ParsedFile(
            file_path=rel,
            mtime_ns=stat.st_mtime_ns,
            size_bytes=stat.st_size,
            content_hash=file_hash or "",
            chunks=file_chunks,
        )
        file_metadata[rel] = parsed
        return parsed

    results: List[Optional[ParsedFile]] = []
    concurrency = max(1, int(os.cpu_count() or 4))
    worker_sem = asyncio.Semaphore(concurrency)

    async def _run(path: Path) -> None:
        async with worker_sem:
            res = await parse_one(path)
            results.append(res)

    async with asyncio.TaskGroup() as tg:
        for path in files:
            tg.create_task(_run(path))

    for res in results:
        if isinstance(res, ParsedFile) and res.chunks:
            chunks.extend(res.chunks)

    return ParseDirectoryResult(
        chunks=chunks,
        file_count=file_count,
        file_metadata=file_metadata,
        changed_files=changed_files,
    )
