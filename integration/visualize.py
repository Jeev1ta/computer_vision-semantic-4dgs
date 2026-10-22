"""
visualize.py
------------
Implements guide section 19 "Option A -- Easiest": produce object-specific
Gaussian previews without touching the SIBR viewer or writing a custom
WebGL engine.

Two outputs:
  1. Per-object filtered PLY files (openable in the SIBR viewer, MeshLab,
     CloudCompare, or Open3D) -- `outputs/semantic/<scene>/colored_preview/`.
  2. A quick top-down matplotlib scatter colored by object id, useful as a
     figure for the report (guide section 32's suggested figure list) and
     as a fast sanity check without needing the full SIBR build.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .ply_compat import PlyData, PlyElement


# A small fixed palette so labels are visually stable across runs/figures.
_PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
    [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
], dtype=np.uint8)

_UNKNOWN_COLOR = np.array([120, 120, 120], dtype=np.uint8)  # neutral gray


def label_to_color(labels: np.ndarray, unknown_label: int = -1) -> np.ndarray:
    """(N,) int labels -> (N, 3) uint8 RGB, stable per label id."""
    colors = np.empty((len(labels), 3), dtype=np.uint8)
    unknown = labels == unknown_label
    colors[unknown] = _UNKNOWN_COLOR
    known_idx = np.where(~unknown)[0]
    colors[known_idx] = _PALETTE[labels[known_idx] % len(_PALETTE)]
    return colors


def save_colored_ply(xyz: np.ndarray, labels: np.ndarray, out_path: str, unknown_label: int = -1):
    """Writes a colored point cloud PLY (xyz + rgb) for quick viewing in any PLY tool."""
    colors = label_to_color(labels, unknown_label)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    data = np.empty(len(xyz), dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    el = PlyElement.describe(data, "vertex")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([el]).write(str(out_path))


def save_per_object_previews(
    xyz: np.ndarray,
    labels: np.ndarray,
    out_dir: str,
    label_names: Optional[Dict[int, str]] = None,
    unknown_label: int = -1,
):
    """
    Splits the scene into one PLY per object id (guide section 19: "render
    only Gaussians assigned to object ID 1"), plus one for the full colored
    scene and one for "unknown".
    """
    out_dir = Path(out_dir) / "colored_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_colored_ply(xyz, labels, out_dir / "full_scene_colored.ply", unknown_label)

    unique_ids = sorted(set(labels.tolist()))
    for obj_id in unique_ids:
        mask = labels == obj_id
        name = "unknown" if obj_id == unknown_label else (label_names or {}).get(obj_id, f"object_{obj_id}")
        save_colored_ply(xyz[mask], labels[mask], out_dir / f"{name}.ply", unknown_label)
        print(f"[visualize] wrote {mask.sum()} Gaussians -> {out_dir / f'{name}.ply'}")


def plot_topdown(
    xyz: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    label_names: Optional[Dict[int, str]] = None,
    unknown_label: int = -1,
    axes: str = "xz",
):
    """
    Quick top-down (bird's-eye) scatter plot colored by object id -- good
    enough for a report figure without needing the SIBR viewer build
    (guide section 32's figure checklist).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        idx = {"x": 0, "y": 1, "z": 2}
        a, b = axes[0], axes[1]
        colors = label_to_color(labels, unknown_label).astype(np.float32) / 255.0

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(xyz[:, idx[a]], xyz[:, idx[b]], c=colors, s=1.5, linewidths=0)
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title("Semantic Gaussian labels (top-down)")
        ax.set_aspect("equal", adjustable="datalim")

        # simple legend
        handles = []
        for obj_id in sorted(set(labels.tolist())):
            name = "unknown" if obj_id == unknown_label else (label_names or {}).get(obj_id, f"object_{obj_id}")
            color = _PALETTE[obj_id % len(_PALETTE)] / 255.0 if obj_id != unknown_label else _UNKNOWN_COLOR / 255.0
            handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                                       markersize=8, label=name))
        ax.legend(handles=handles, loc="upper right", fontsize=8)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualize] wrote figure -> {out_path}")

    except ImportError:
        # Fallback to PIL Image rendering if matplotlib is not installed
        from PIL import Image, ImageDraw, ImageFont

        idx_map = {"x": 0, "y": 1, "z": 2}
        ax_a, ax_b = idx_map[axes[0]], idx_map[axes[1]]

        img_w, img_h = 700, 700
        img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        pts_a = xyz[:, ax_a]
        pts_b = xyz[:, ax_b]

        min_a, max_a = pts_a.min(), pts_a.max()
        min_b, max_b = pts_b.min(), pts_b.max()

        pad = 0.1
        range_a = max(max_a - min_a, 1e-5) * (1 + 2 * pad)
        range_b = max(max_b - min_b, 1e-5) * (1 + 2 * pad)
        mid_a = (min_a + max_a) / 2
        mid_b = (min_b + max_b) / 2

        scale_a = (img_w - 100) / range_a
        scale_b = (img_h - 100) / range_b
        scale = min(scale_a, scale_b)

        colors = label_to_color(labels, unknown_label)

        for i in range(len(xyz)):
            px = int((pts_a[i] - mid_a) * scale + img_w / 2)
            py = int((mid_b - pts_b[i]) * scale + img_h / 2)  # flip Y
            r = 3
            c = tuple(colors[i])
            draw.ellipse([px - r, py - r, px + r, py + r], fill=c)

        # Legend
        y_off = 20
        unique_ids = sorted(set(labels.tolist()))
        for obj_id in unique_ids:
            name = "unknown" if obj_id == unknown_label else (label_names or {}).get(obj_id, f"object_{obj_id}")
            col = tuple(_PALETTE[obj_id % len(_PALETTE)]) if obj_id != unknown_label else tuple(_UNKNOWN_COLOR)
            draw.rectangle([img_w - 180, y_off, img_w - 165, y_off + 15], fill=col)
            draw.text((img_w - 155, y_off), f"{name} (id={obj_id})", fill=(0, 0, 0))
            y_off += 25

        draw.text((20, 20), "Semantic Gaussian labels (top-down)", fill=(0, 0, 0))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        print(f"[visualize] wrote PIL fallback figure -> {out_path}")

