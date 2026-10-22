"""
ply_compat.py
-------------
Single import point for PLY reading/writing. Uses the real `plyfile` package
when installed (preferred) and falls back to the bundled `plyfile_lite`
implementation otherwise, so the integration environment works offline with
zero pip installs.
"""

from __future__ import annotations

try:
    from plyfile import PlyData, PlyElement  # type: ignore  # noqa: F401
    PLY_BACKEND = "plyfile"
except ImportError:  # pragma: no cover - exercised on offline machines
    from .plyfile_lite import PlyData, PlyElement  # type: ignore  # noqa: F401
    PLY_BACKEND = "plyfile_lite"
