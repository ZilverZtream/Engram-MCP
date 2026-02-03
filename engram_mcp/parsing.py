from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional

from .chunking import Chunk, chunk_text, make_chunk_id, token_count


SUPPORTED_TEXT_EXTS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".cs", ".java", ".go", ".rs", ".cpp", ".c", ".h", ".hpp", ".sql", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".html", ".css", ".scss", ".xml", ".sh", ".bash", ".ps1",
}

SUPPORTED_DOC_EXTS = {".pdf", ".docx"}


def _is_ignored(path: str, ignore_patterns: List[str]) -> bool:
    for pat in ignore_patterns:
        if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(os.path.basename(path), pat):
            return True
    return False


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def iter_files(root: str, ignore_patterns: List[str]) -> Iterator[Path]:
    root_p = Path(root)
    for p in root_p.rglob("*"):
        if not p.is_file():
            continue
        if not _is_within_root(p, root_p):
            continue
        rel = str(p.relative_to(root_p))
        if _is_ignored(rel, ignore_patterns):
            continue
        yield p


def read_text_streaming(path: Path, max_bytes: int) -> str:
    # Avoid reading enormous files at once.
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"File too large: {path} ({size} bytes)")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def hash_file(path: Path, max_bytes: int) -> str:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"File too large: {path} ({size} bytes)")
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_file_to_chunks(
    *,
    path: Path,
    root: str,
    project_type: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    max_file_size_mb: int,
) -> List[Chunk]:
    if not _is_within_root(path, Path(root)):
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
        text = read_text_streaming(path, max_bytes=max_bytes)
        base_id = make_chunk_id(str(path), str(path.stat().st_mtime_ns))
        return chunk_text(
            text=text,
            base_id=base_id,
            meta=base_meta,
            target_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )

    if ext == ".pdf":
        # Streaming: page by page
        from pypdf import PdfReader

        # Check PDF file size before loading (prevent OOM/DoS)
        size = path.stat().st_size
        if size > max_bytes:
            raise ValueError(f"PDF file too large: {path} ({size} bytes)")

        reader = PdfReader(str(path))
        chunks: List[Chunk] = []
        base_id = make_chunk_id(str(path), str(path.stat().st_mtime_ns))
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue
            page_meta = {**base_meta, "type": "pdf_page", "page": i + 1}
            # Keep pages as their own units, then chunk if needed
            if token_count(page_text) <= chunk_size_tokens * 2:
                cid = make_chunk_id(base_id, f"page:{i+1}")
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

        doc = Document(str(path))
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        text = "\n".join(paras)
        base_id = make_chunk_id(str(path), str(path.stat().st_mtime_ns))
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
    for p in iter_files(root, ignore_patterns):
        ext = p.suffix.lower()
        if ext in SUPPORTED_TEXT_EXTS or ext in SUPPORTED_DOC_EXTS:
            files.append(p)

    file_count = len(files)
    sem = asyncio.Semaphore(os.cpu_count() or 4)

    async def parse_one(path: Path) -> Optional[ParsedFile]:
        rel = str(path.relative_to(root))
        stat = path.stat()
        existing = existing_metadata.get(rel)
        file_hash: Optional[str] = None
        if existing and existing.mtime_ns == stat.st_mtime_ns and existing.size_bytes == stat.st_size:
            try:
                file_hash = await asyncio.to_thread(hash_file, path, max_file_size_mb * 1024 * 1024)
            except Exception:
                logging.warning("Failed to hash %s", rel, exc_info=True)
            if file_hash and existing.content_hash == file_hash:
                file_metadata[rel] = existing
                return None

        if file_hash is None:
            try:
                file_hash = await asyncio.to_thread(hash_file, path, max_file_size_mb * 1024 * 1024)
            except Exception:
                logging.warning("Failed to hash %s", rel, exc_info=True)
                return None

        async with sem:
            try:
                file_chunks = await asyncio.to_thread(
                    parse_file_to_chunks,
                    path=path,
                    root=root,
                    project_type=project_type,
                    chunk_size_tokens=chunk_size_tokens,
                    overlap_tokens=overlap_tokens,
                    max_file_size_mb=max_file_size_mb,
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

    async for res in _bounded_gather(files, parse_one, max(1, int(os.cpu_count() or 4))):
        if isinstance(res, ParsedFile) and res.chunks:
            chunks.extend(res.chunks)

    return ParseDirectoryResult(
        chunks=chunks,
        file_count=file_count,
        file_metadata=file_metadata,
        changed_files=changed_files,
    )


async def _bounded_gather(items: List[Path], worker, concurrency: int):
    iterator = iter(items)
    pending: set[asyncio.Task] = set()

    for _ in range(concurrency):
        try:
            item = next(iterator)
        except StopIteration:
            break
        pending.add(asyncio.create_task(worker(item)))

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            yield task.result()
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.add(asyncio.create_task(worker(item)))
