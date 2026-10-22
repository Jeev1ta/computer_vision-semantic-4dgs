"""
make_fixture.py
---------------
Builds a small SYNTHETIC but fully valid scene used by tests and the demo
pipeline. The geometry is genuinely consistent: the "segmentation masks" are
exactly the projected extents of the two Gaussian clusters under the fixture's
calibrated cameras, so the multi-view voting has real work to do.

Disk persistence (optional, see `write_disk_artifacts`) uses the project's
existing writers. Nothing is ever written to data/ or the real outputs/ --
the fixture lives under tests/fixtures/synthetic_scene/ (in memory by default).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from integration.colmap_io import Camera
from integration.gaussian_io import GaussianCloud

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "synthetic_scene"

WIDTH, HEIGHT = 640, 480
FX = FY = 500.0
CX, CY = 320.0, 240.0

# two cameras on the x-axis looking down +z (identity rotation)
CAMERA_OFFSETS = {
    "00015.jpg": -0.5,
    "00020.jpg": +0.5,
}

# two object clusters + far background, in world coordinates.
# Clusters are FLAT SURFACE PATCHES (uniform over a constant-depth plane),
# like the surface tiling of a real 3DGS scene -- this makes the point-splat
# projected-back evaluation meaningful and the vote geometry unambiguous.
CLUSTERS = {
    1: {"label": "chair",
        "lo": np.array([-0.45, -0.15, 2.0]), "hi": np.array([-0.15, 0.15, 2.0]), "n": 3000},
    2: {"label": "table",
        "lo": np.array([0.15, -0.15, 2.0]), "hi": np.array([0.45, 0.15, 2.0]), "n": 3000},
}
N_BACKGROUND = 1000
BACKGROUND = {"lo": np.array([-1.2, -1.0, 4.0]), "hi": np.array([1.2, 1.0, 4.0])}


def make_camera(name: str, x_offset: float) -> Camera:
    """Identity-rotation camera at (x_offset, 0, 0) looking down +z."""
    return Camera(
        camera_id=1, image_id=len(name), name=name,
        width=WIDTH, height=HEIGHT,
        fx=FX, fy=FY, cx=CX, cy=CY,
        R=np.eye(3), t=np.array([-x_offset, 0.0, 0.0]),
    )


def build_cameras() -> Dict[str, Camera]:
    return {name: make_camera(name, off) for name, off in CAMERA_OFFSETS.items()}


def build_cloud(seed: int = 42):
    """Returns (cloud, gt_labels): two surface-patch clusters + background,
    and the ground-truth labels used ONLY to build consistent masks/eval."""
    rng = np.random.default_rng(seed)
    xyz_blocks, label_blocks = [], []
    for obj_id, spec in CLUSTERS.items():
        pts = rng.uniform(spec["lo"], spec["hi"], size=(spec["n"], 3))
        xyz_blocks.append(pts)
        label_blocks.append(np.full(spec["n"], obj_id))
    bg = rng.uniform(BACKGROUND["lo"], BACKGROUND["hi"], size=(N_BACKGROUND, 3))
    xyz_blocks.append(bg)
    label_blocks.append(np.full(N_BACKGROUND, -1))
    xyz = np.vstack(xyz_blocks)
    gt_labels = np.concatenate(label_blocks)
    cloud = GaussianCloud(xyz=xyz, opacity=np.ones(len(xyz)), scale=None,
                          rotation=None, n=len(xyz))
    return cloud, gt_labels


def build_masks(cloud: GaussianCloud, gt_labels: np.ndarray, cameras) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Rectangular "segmentation masks" = bounding box of each cluster's actual
    projection into each camera (what a perfect segmenter would return here).
    votes[image_name][object_id] -> (H, W) bool
    """
    masks: Dict[str, Dict[int, np.ndarray]] = {}
    for name, cam in cameras.items():
        uv = cam.world_to_camera(cloud.xyz)
        z = cam.world_to_camera(cloud.xyz)[:, 2]
        u = FX * uv[:, 0] / z + CX
        v = FY * uv[:, 1] / z + CY
        view_masks: Dict[int, np.ndarray] = {}
        for obj_id in CLUSTERS:
            pts = np.stack([u[gt_labels == obj_id], v[gt_labels == obj_id]], axis=1)
            inside = ((pts[:, 0] >= 0) & (pts[:, 0] < WIDTH) &
                      (pts[:, 1] >= 0) & (pts[:, 1] < HEIGHT))
            if not inside.any():
                continue
            x0, y0 = np.floor(pts[inside].min(axis=0)).astype(int) - 6
            x1, y1 = np.ceil(pts[inside].max(axis=0)).astype(int) + 6
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(WIDTH, x1), min(HEIGHT, y1)
            m = np.zeros((HEIGHT, WIDTH), dtype=bool)
            m[y0:y1, x0:x1] = True
            view_masks[obj_id] = m
        if view_masks:
            masks[name] = view_masks
    return masks


def build_mask_scores(masks) -> Dict[str, Dict[int, float]]:
    return {name: {obj_id: (0.95 if obj_id == 1 else 0.90) for obj_id in view}
            for name, view in masks.items()}


def build_gt_masks(cloud: GaussianCloud, gt_labels: np.ndarray, cameras):
    """Ground-truth REGION masks for eval: the tight bounding box of each
    cluster's true projection (what a human polygon annotation would trace),
    with no extra padding -- stricter than the vote masks, same granularity."""
    gt = {}
    for name, cam in cameras.items():
        uv = cam.world_to_camera(cloud.xyz)
        z = cam.world_to_camera(cloud.xyz)[:, 2]
        u = FX * uv[:, 0] / z + CX
        v = FY * uv[:, 1] / z + CY
        view_gt = {}
        for obj_id in CLUSTERS:
            sel = gt_labels == obj_id
            m = np.zeros((HEIGHT, WIDTH), dtype=bool)
            uu, vv = u[sel], v[sel]
            inside = (uu >= 0) & (uu < WIDTH) & (vv >= 0) & (vv < HEIGHT)
            if not inside.any():
                continue
            x0, y0 = np.floor(uu[inside].min()).astype(int), np.floor(vv[inside].min()).astype(int)
            x1, y1 = np.ceil(uu[inside].max()).astype(int), np.ceil(vv[inside].max()).astype(int)
            m[max(0, y0):min(HEIGHT, y1), max(0, x0):min(WIDTH, x1)] = True
            view_gt[obj_id] = m
        gt[name] = view_gt
    return gt


def fixture_config(output_root=None) -> dict:
    """Config dict equivalent to configs/config.yaml but aimed at the fixture."""
    repo_root = Path(__file__).resolve().parent.parent
    out_root = Path(output_root) if output_root else FIXTURE_ROOT / "outputs"
    return {
        "scene_name": "synthetic_scene",
        "paths": {
            "gaussian_splatting_repo": str(repo_root / "third_party" / "gaussian-splatting"),
            "dataset_dir": str(FIXTURE_ROOT),
            "model_dir": str(FIXTURE_ROOT),
            "mask_dir": str(FIXTURE_ROOT),
            "semantic_dir": str(out_root / "semantic"),
        },
        "association": {
            "strategy": "weighted",
            "min_views": 2,
            "min_mask_area": 25,
            "min_opacity": 0.05,
        },
        "segmentation": {"backend": "classical", "max_objects": 8},
        "ablation": [
            {"name": "baseline_A_single_view", "strategy": "single_view"},
            {"name": "baseline_B_majority", "strategy": "majority", "min_views": 1},
            {"name": "proposed_weighted", "strategy": "weighted", "min_views": 2},
        ],
    }
