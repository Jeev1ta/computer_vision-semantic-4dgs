"""
single_image.py
---------------
Honest 2.5D lifting for a SINGLE image with no multi-view scene data.

Truth-in-advertising: one image cannot give true metric 3D -- that requires
either multi-view geometry (the main pipeline) or a monocular depth network.
What we do here is REAL segmentation (segmentation.py) + an EXPLICITLY
LABELED depth assumption:

    depth_mode="plane_assumption"
        Each detected object is placed at a single constant depth, ordered by
        two documented monocular cues: objects lower in the frame and objects
        covering more area are treated as nearer. Lateral positions come from
        the real pinhole projection of each object's actual pixels at that
        depth, using a nominal ~60 deg horizontal FOV.

The result is a point cloud whose per-object SHAPES are genuinely segmented
from the image; only the depth axis is assumed. Every artifact produced here
carries `mode: "single_image_2.5d"` and `depth_source: "plane_assumption"` in
its metadata so nothing downstream mistakes it for a reconstruction.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .segmentation import Segment, segment_image

# nominal horizontal field of view for the pinhole lift (~60 deg -> f = W/0.866)
_DEFAULT_HFOV_DEG = 60.0
_NEAR_Z, _FAR_Z = 1.0, 4.0     # depth range of the normalized scene
_MAX_POINTS_PER_OBJECT = 400   # subsample mask pixels per object (grid-based)


def _subsample_points(mask: np.ndarray, max_points: int) -> np.ndarray:
    """Evenly subsample mask pixels on a grid (deterministic)."""
    ys, xs = np.where(mask)
    if len(ys) <= max_points:
        return np.stack([xs, ys], axis=1)
    step = int(np.ceil(np.sqrt(len(ys) / max_points)))
    keep = (ys % step == 0) & (xs % step == 0)
    if keep.sum() < 8:  # degenerate grid for tiny masks: take strided slice
        idx = np.linspace(0, len(ys) - 1, max_points).astype(int)
        return np.stack([xs[idx], ys[idx]], axis=1)
    return np.stack([xs[keep], ys[keep]], axis=1)


def _plane_assumption_depths(segments: List[Segment], img_h: int) -> List[float]:
    """
    Documented ordering heuristic: nearness = 0.6 * (how low the object's
    bottom edge sits in the frame) + 0.4 * (area fraction). Nearest object
    sits at _NEAR_Z, farthest at _FAR_Z, others linearly spaced between.
    """
    if not segments:
        return []
    img_w = segments[0].mask.shape[1]
    total_px = img_h * img_w
    nearness = []
    for s in segments:
        low_frac = s.bbox[3] / img_h                       # 0 top -> 1 bottom
        area_frac = min(1.0, s.area / max(1, total_px))
        nearness.append(0.6 * low_frac + 0.4 * area_frac)
    n = [float(x) for x in nearness]
    lo, hi = min(n), max(n)
    span = (hi - lo) or 1.0
    return [_NEAR_Z + (_FAR_Z - _NEAR_Z) * (1.0 - (x - lo) / span) for x in n]


def lift_single_image(image_rgb: np.ndarray,
                      segments: List[Segment],
                      depth_mode: str = "plane_assumption") -> dict:
    """
    Lift real segmentation results into an honest 2.5D point cloud.

    Returns dict with numpy arrays + provenance metadata:
        xyz (N,3), labels (N,), confidence (N,), label_names {id: str},
        objects [{id,label,area_frac,mean_rgb}], mode, depth_source,
        backend, image_size, hfov_deg
    """
    if depth_mode != "plane_assumption":
        raise ValueError(f"unknown depth_mode {depth_mode!r} "
                         "(only the explicitly-labeled 'plane_assumption' exists)")
    h, w = image_rgb.shape[:2]

    # sort by area (largest first) so object ids are stable: 1 = biggest
    segments = sorted(segments, key=lambda s: s.area, reverse=True)

    depths = _plane_assumption_depths(segments, h)
    focal = w / (2.0 * np.tan(np.radians(_DEFAULT_HFOV_DEG) / 2.0))
    cx, cy = w / 2.0, h / 2.0

    xyz_blocks, label_blocks, conf_blocks, objects_meta = [], [], [], []
    for i, (seg, z) in enumerate(zip(segments, depths), start=1):
        pts2d = _subsample_points(seg.mask, _MAX_POINTS_PER_OBJECT)
        if len(pts2d) == 0:
            continue
        us, vs = pts2d[:, 0].astype(np.float64), pts2d[:, 1].astype(np.float64)
        xs = (us - cx) * z / focal
        ys = -(vs - cy) * z / focal
        zs = np.full(len(us), z)

        px = image_rgb[vs.astype(int), us.astype(int)].astype(np.float64)
        xyz_blocks.append(np.stack([xs, ys, zs], axis=1))
        label_blocks.append(np.full(len(us), i, dtype=np.int64))
        conf_blocks.append(np.full(len(us), seg.score, dtype=np.float64))
        objects_meta.append({
            "id": i,
            "label": seg.label,
            "backend": seg.backend,
            "score": round(seg.score, 4),
            "area_frac": round(seg.area / float(h * w), 4),
            "mean_rgb": [int(c) for c in px.mean(axis=0)],
            "assumed_depth": round(z, 3),
        })

    if not xyz_blocks:
        raise ValueError("no non-empty segments to lift")

    xyz = np.concatenate(xyz_blocks).astype(np.float32)
    labels = np.concatenate(label_blocks)
    confidence = np.concatenate(conf_blocks)
    label_names = {o["id"]: o["label"] for o in objects_meta}

    return {
        "xyz": xyz,
        "labels": labels,
        "confidence": confidence,
        "label_names": label_names,
        "objects": objects_meta,
        "mode": "single_image_2.5d",
        "depth_source": "plane_assumption",
        "backend": segments[0].backend if segments else "none",
        "image_size": [w, h],
        "hfov_deg": _DEFAULT_HFOV_DEG,
    }


def segment_and_lift(image_rgb: np.ndarray,
                     backend: str = "auto",
                     max_objects: int = 10) -> dict:
    """Convenience: real segmentation + honest 2.5D lift in one call."""
    segments = segment_image(image_rgb, backend=backend, max_objects=max_objects)
    return lift_single_image(image_rgb, segments)
