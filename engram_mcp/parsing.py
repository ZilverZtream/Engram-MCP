from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

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


def iter_files(root: str, ignore_patterns: List[str]) -> Iterator[Path]:
    root_p = Path(root)
    for p in root_p.rglob("*"):
        if not p.is_file():
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


def parse_file_to_chunks(
    *,
    path: Path,
    root: str,
    project_type: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    max_file_size_mb: int,
) -> List[Chunk]:
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


def parse_directory(
    *,
    root: str,
    project_type: str,
    ignore_patterns: List[str],
    chunk_size_tokens: int,
    overlap_tokens: int,
    max_file_size_mb: int,
) -> Tuple[List[Chunk], int]:
    chunks: List[Chunk] = []
    file_count = 0
    for p in iter_files(root, ignore_patterns):
        ext = p.suffix.lower()
        if ext not in SUPPORTED_TEXT_EXTS and ext not in SUPPORTED_DOC_EXTS:
            continue
        try:
            file_chunks = parse_file_to_chunks(
                path=p,
                root=root,
                project_type=project_type,
                chunk_size_tokens=chunk_size_tokens,
                overlap_tokens=overlap_tokens,
                max_file_size_mb=max_file_size_mb,
            )
            if file_chunks:
                chunks.extend(file_chunks)
            file_count += 1
        except Exception:
            # Best effort: skip unreadable files
            continue
    return chunks, file_count
