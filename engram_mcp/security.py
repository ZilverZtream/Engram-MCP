from __future__ import annotations

import os
import re
import tempfile
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


class PathNotAllowed(Exception):
    """Raised when an input path is outside the configured allowed roots."""


PROJECT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
# Windows reserves these names (with or without an extension) as device
# handles.  Creating a file like "CON.index" on Windows interacts with
# the console driver instead of the filesystem.
_WINDOWS_RESERVED = re.compile(
    r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)", re.IGNORECASE
)
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class ProjectID:
    value: str

    def __post_init__(self) -> None:
        sanitized = os.path.basename(self.value)
        if sanitized != self.value or not PROJECT_ID_PATTERN.match(self.value):
            raise ValueError(
                f"Invalid project_id: {self.value}. Only alphanumeric characters, underscores, and hyphens are allowed."
            )
        if _WINDOWS_RESERVED.match(self.value):
            raise ValueError(
                f"Invalid project_id: {self.value}. "
                "Names that are reserved device identifiers on Windows (CON, PRN, AUX, NUL, COM1-9, LPT1-9) are not allowed."
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

    def ensure_allowed_nofollow(self, path: str | Path) -> str:
        if not self._allowed_roots:
            raise PathNotAllowed(
                "No allowed_roots configured. Set allowed_roots in engram_mcp.yaml to enable indexing/searching."
            )
        ap = os.path.abspath(str(path))
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
        if any(flag in mode for flag in ("w", "a", "+", "x")):
            raise ValueError("PathContext.open_file only supports read modes. Use write_bytes_atomic/write_text_atomic.")
        fd = self._open_file_no_symlink_race(path)
        return os.fdopen(fd, mode, encoding=encoding, errors=errors)

    def _open_file_no_symlink_race(self, path: str | Path) -> int:
        """Open *path* for reading without a symlink TOCTOU window.

        The path is validated and opened through dirfd-relative ``open`` calls
        with ``O_NOFOLLOW`` on each path component.
        """
        raw_abs = os.path.abspath(str(path))
        normalized_abs = self.ensure_allowed_nofollow(raw_abs)
        norm_ap = self._normalize_case(normalized_abs)

        best_root: Optional[str] = None
        for root in self._allowed_roots:
            norm_root = self._normalize_case(root)
            try:
                common = os.path.commonpath([norm_ap, norm_root])
            except ValueError:
                continue
            if common == norm_root and (best_root is None or len(root) > len(best_root)):
                best_root = root

        if not best_root:
            raise PathNotAllowed(
                f"Path '{normalized_abs}' is outside allowed_roots. Allowed roots: {self._allowed_roots}"
            )

        rel = os.path.relpath(normalized_abs, start=best_root)
        parts = [p for p in rel.split(os.sep) if p and p != "."]

        cloexec_flag = os.O_CLOEXEC if hasattr(os, "O_CLOEXEC") else 0
        nofollow_flag = os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
        dir_flag = os.O_DIRECTORY if hasattr(os, "O_DIRECTORY") else 0

        base_fd = os.open(best_root, os.O_RDONLY | dir_flag | nofollow_flag | cloexec_flag)
        current_fd = base_fd
        try:
            if not parts:
                raise IsADirectoryError(f"Expected a file path, got allowed root directory: {best_root}")

            for index, part in enumerate(parts):
                is_last = index == len(parts) - 1
                flags = os.O_RDONLY | nofollow_flag | cloexec_flag
                if not is_last:
                    flags |= dir_flag
                next_fd = os.open(part, flags, dir_fd=current_fd)
                if current_fd != base_fd:
                    os.close(current_fd)
                current_fd = next_fd

            fd_stat = os.fstat(current_fd)
            if not os.path.samestat(fd_stat, os.stat(normalized_abs)):
                raise PathNotAllowed(f"Path validation changed while opening '{normalized_abs}'")
            return current_fd
        except Exception:
            if current_fd != base_fd:
                os.close(current_fd)
            raise
        finally:
            os.close(base_fd)

    def list_dir(self, path: str | Path) -> List[str]:
        resolved = self.resolve_path(path)
        return os.listdir(resolved)

    def iter_files(self, root: str | Path) -> Iterator[Path]:
        resolved_root = self.resolve_path(root)
        stack = [Path(resolved_root)]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        candidate = Path(entry.path)
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                if entry.is_symlink():
                                    continue
                                stack.append(candidate)
                                continue
                            if entry.is_file(follow_symlinks=False):
                                if entry.is_symlink():
                                    try:
                                        self.ensure_allowed(str(candidate))
                                    except PathNotAllowed:
                                        continue
                                yield candidate
                        except OSError:
                            logging.debug("Skipping unreadable filesystem entry: %s", entry.path, exc_info=True)
                            continue
            except OSError:
                logging.debug("Skipping unreadable directory during traversal: %s", current, exc_info=True)
                continue

    def stat(self, path: str | Path) -> os.stat_result:
        resolved = self.resolve_path(path)
        return resolved.stat()

    def lstat(self, path: str | Path) -> os.stat_result:
        resolved = self.ensure_allowed_nofollow(path)
        return os.lstat(resolved)

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

    def create_temp_file(self, *, dir_path: str | Path, suffix: str) -> tuple[int, str]:
        resolved_dir = self.resolve_path(dir_path)
        fd, temp_path = tempfile.mkstemp(dir=resolved_dir, suffix=suffix)
        try:
            os.chmod(temp_path, 0o600)
        except OSError:
            logging.debug("Failed to chmod temp file %s", temp_path, exc_info=True)
        return fd, temp_path

    def chmod(self, path: str | Path, mode: int) -> None:
        resolved = self.resolve_path(path)
        os.chmod(resolved, mode)

    def write_text_atomic(self, path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
        data = text.encode(encoding)
        self.write_bytes_atomic(path, data)

    def write_bytes_atomic(self, path: str | Path, data: bytes) -> None:
        resolved = self.resolve_path(path)
        dir_path = resolved.parent
        fd, temp_path = self.create_temp_file(dir_path=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            try:
                os.chmod(temp_path, 0o600)
            except OSError:
                logging.debug("Failed to chmod temp file %s", temp_path, exc_info=True)
            os.replace(temp_path, resolved)
            try:
                os.chmod(resolved, 0o600)
            except OSError:
                logging.debug("Failed to chmod %s", resolved, exc_info=True)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    logging.debug("Failed to cleanup temp file %s", temp_path, exc_info=True)

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


def validate_project_field(value: str, *, field_name: str, max_length: int = 200) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")
    if len(cleaned) > max_length:
        raise ValueError(f"{field_name} exceeds max length of {max_length}.")
    if _CONTROL_CHARS.search(cleaned):
        raise ValueError(f"{field_name} contains control characters.")
    if not cleaned.isprintable():
        raise ValueError(f"{field_name} contains non-printable characters.")
    return cleaned
