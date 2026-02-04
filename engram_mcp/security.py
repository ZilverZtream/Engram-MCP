from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


class PathNotAllowed(Exception):
    """Raised when an input path is outside the configured allowed roots."""


PROJECT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass(frozen=True)
class ProjectID:
    value: str

    def __post_init__(self) -> None:
        sanitized = os.path.basename(self.value)
        if sanitized != self.value or not PROJECT_ID_PATTERN.match(self.value):
            raise ValueError(
                f"Invalid project_id: {self.value}. Only alphanumeric characters, underscores, and hyphens are allowed."
            )

    def __str__(self) -> str:
        return self.value


class PathContext:
    def __init__(self, allowed_roots: Iterable[str]) -> None:
        roots = [self._normalize_root(p) for p in allowed_roots]
        self._allowed_roots = [r for r in roots if r]

    @property
    def allowed_roots(self) -> List[str]:
        return list(self._allowed_roots)

    def _normalize_root(self, root: str) -> Optional[str]:
        if not root:
            return None
        return os.path.realpath(os.path.abspath(root))

    def _normalize_path(self, path: str) -> str:
        return os.path.realpath(os.path.abspath(path))

    def _normalize_case(self, path: str) -> str:
        if os.name == "nt":
            return os.path.normcase(path)
        return path

    def ensure_allowed(self, path: str) -> str:
        if not self._allowed_roots:
            raise PathNotAllowed(
                "No allowed_roots configured. Set allowed_roots in engram_mcp.yaml to enable indexing/searching."
            )
        ap = self._normalize_path(path)
        norm_ap = self._normalize_case(ap)
        for root in self._allowed_roots:
            norm_root = self._normalize_case(root)
            try:
                common = os.path.commonpath([norm_ap, norm_root])
            except ValueError:
                continue
            if common == norm_root:
                return ap
        raise PathNotAllowed(
            f"Path '{ap}' is outside allowed_roots. Allowed roots: {self._allowed_roots}"
        )

    def resolve_path(self, path: str | Path) -> Path:
        ap = self.ensure_allowed(str(path))
        return Path(ap)

    def open_file(
        self,
        path: str | Path,
        mode: str,
        *,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
    ):
        resolved = self.resolve_path(path)
        if "r" not in mode:
            return open(resolved, mode, encoding=encoding, errors=errors)

        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(resolved, flags)
        return os.fdopen(fd, mode, encoding=encoding, errors=errors)

    def list_dir(self, path: str | Path) -> List[str]:
        resolved = self.resolve_path(path)
        return os.listdir(resolved)

    def iter_files(self, root: str | Path) -> Iterator[Path]:
        resolved_root = self.resolve_path(root)
        for dirpath, dirnames, filenames in os.walk(resolved_root, followlinks=False):
            for name in filenames:
                candidate = Path(dirpath) / name
                try:
                    self.ensure_allowed(str(candidate))
                except PathNotAllowed:
                    continue
                yield candidate

    def stat(self, path: str | Path) -> os.stat_result:
        resolved = self.resolve_path(path)
        return resolved.stat()

    def exists(self, path: str | Path) -> bool:
        try:
            resolved = self.resolve_path(path)
        except PathNotAllowed:
            return False
        return resolved.exists()

    def unlink(self, path: str | Path) -> None:
        resolved = self.resolve_path(path)
        resolved.unlink()

    def replace(self, src: str | Path, dest: str | Path) -> None:
        resolved_src = self.resolve_path(src)
        resolved_dest = self.resolve_path(dest)
        os.replace(resolved_src, resolved_dest)

    def makedirs(self, path: str | Path, *, exist_ok: bool = True) -> None:
        resolved = self.resolve_path(path)
        os.makedirs(resolved, exist_ok=exist_ok)

    def create_temp_file(self, *, dir_path: str | Path, suffix: str) -> str:
        resolved_dir = self.resolve_path(dir_path)
        temp = tempfile.NamedTemporaryFile(mode="wb", delete=False, dir=resolved_dir, suffix=suffix)
        temp.close()
        return temp.name

def is_within_allowed_roots(path: str, allowed_roots: Iterable[str]) -> bool:
    """Return True if path is within any allowed root (prefix match by commonpath)."""
    ap = os.path.realpath(path)
    for root in allowed_roots:
        ar = os.path.realpath(root)
        try:
            common = os.path.commonpath([ap, ar])
        except ValueError:
            # Different drives on Windows etc.
            continue
        if common == ar:
            return True
    return False


def enforce_allowed_roots(path: str, allowed_roots: Iterable[str]) -> str:
    """Validate and return normalized absolute path with symlinks resolved."""
    ap = os.path.realpath(path)
    roots = list(allowed_roots)
    if not roots:
        # Safe default: require explicit opt-in.
        raise PathNotAllowed(
            "No allowed_roots configured. Set allowed_roots in engram_mcp.yaml to enable indexing/searching."
        )
    if not is_within_allowed_roots(ap, roots):
        raise PathNotAllowed(
            f"Path '{ap}' is outside allowed_roots. Allowed roots: {roots}"
        )
    return ap
