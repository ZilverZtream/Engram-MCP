from __future__ import annotations

import os
from typing import Iterable


class PathNotAllowed(Exception):
    """Raised when an input path is outside the configured allowed roots."""


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
