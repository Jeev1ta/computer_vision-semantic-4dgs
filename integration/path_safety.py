"""
path_safety.py
--------------
Central path-safety helpers. Metadata files (mask_file entries, config paths,
dashboard requests) can be influenced by user input, so every path that is
resolved from such data must be (a) normalized and (b) provably contained in
an allowed root. File writes go through `ensure_writable` for the same reason.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Writes are allowed inside the project tree and the system temp directory
# (pytest tmp_path fixtures live there). Extra roots can be registered.
_ALLOWED_WRITE_ROOTS: List[Path] = [_PROJECT_ROOT, Path(tempfile.gettempdir()).resolve()]


def allow_writable_root(root) -> Path:
    """Register an additional directory tree that file writes may target."""
    resolved = Path(root).resolve()
    if resolved not in _ALLOWED_WRITE_ROOTS:
        _ALLOWED_WRITE_ROOTS.append(resolved)
    return resolved


def allowed_roots() -> Iterable[Path]:
    return tuple(_ALLOWED_WRITE_ROOTS)


def ensure_writable(path) -> Path:
    """Normalize an output path and refuse anything outside allowed roots."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved)
    resolved = resolved.resolve()
    for root in _ALLOWED_WRITE_ROOTS:
        if resolved == root or root in resolved.parents:
            return resolved
    raise ValueError(
        f"Refusing to write to {str(path)!r}: outside allowed directories "
        f"({', '.join(str(r) for r in _ALLOWED_WRITE_ROOTS)}). "
        "Call allow_writable_root() first if this destination is intended."
    )


def safe_resolve_under(base, relative) -> Path:
    """Resolve `relative` against `base`, refusing traversal outside `base`."""
    base_resolved = Path(base).resolve()
    resolved = (base_resolved / str(relative)).resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise ValueError(
            f"Unsafe path {str(relative)!r}: escapes allowed directory {base_resolved}"
        )
    return resolved
