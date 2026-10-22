"""
eval_semantic.py
-----------------
Implements the "projected-back evaluation" strategy from guide section 22:
instead of inventing an arbitrary 3D semantic-accuracy metric, we project
the PREDICTED Gaussian labels back into a small set of held-out evaluation
views and compare against manually-annotated ground-truth masks there.

    SAM2 prediction -> 2D->3D association -> predicted object Gaussians
        -> project back to evaluation views -> compare with GT masks
        -> IoU / precision / recall  (guide section 21.2 / 22)

Ground-truth format expected: a directory of PNG masks per (view, object),
named like `gt_masks/<image_name>/<object_id>.png` (binary 0/255), created
by manually annotating ~10 held-out images (guide section 22).

The report is per-view AND aggregated per object, so a single bad view is
visible instead of being averaged away.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from . import projection
from .colmap_io import Camera
from .gaussian_io import GaussianCloud


def render_predicted_mask(
    cloud: GaussianCloud,
    labels: np.ndarray,
    object_id: int,
    camera: Camera,
    dilate_px: int = 1,
) -> np.ndarray:
    """
    Projects every Gaussian labeled `object_id` into `camera` and rasterizes
    a binary prediction mask at the camera's resolution. This is a simple
    point-splat rasterization (not a full alpha-blended render) -- good
    enough for IoU-style evaluation without needing the CUDA rasterizer.
    """
    obj_mask = labels == object_id
    if not obj_mask.any():
        return np.zeros((camera.height, camera.width), dtype=bool)

    uv, visible = projection.project_points_batch(
        cloud.xyz[obj_mask], camera.R, camera.t,
        camera.fx, camera.fy, camera.cx, camera.cy, camera.width, camera.height,
    )
    xs = np.round(uv[visible, 0]).astype(np.int64)
    ys = np.round(uv[visible, 1]).astype(np.int64)

    pred = np.zeros((camera.height, camera.width), dtype=bool)
    valid = (xs >= 0) & (xs < camera.width) & (ys >= 0) & (ys < camera.height)
    pred[ys[valid], xs[valid]] = True

    if dilate_px > 0:
        pred = _dilate(pred, dilate_px)
    return pred


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Cheap binary dilation with no cv2/scipy dependency required."""
    out = mask.copy()
    for _ in range(px):
        shifted = np.zeros_like(out)
        shifted[1:, :] |= out[:-1, :]
        shifted[:-1, :] |= out[1:, :]
        shifted[:, 1:] |= out[:, :-1]
        shifted[:, :-1] |= out[:, 1:]
        out = out | shifted
    return out


def iou_precision_recall(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union > 0 else float("nan")
    precision = intersection / pred.sum() if pred.sum() > 0 else float("nan")
    recall = intersection / gt.sum() if gt.sum() > 0 else float("nan")
    return {"iou": float(iou), "precision": float(precision), "recall": float(recall)}


def evaluate_scene(
    cloud: GaussianCloud,
    labels: np.ndarray,
    cameras: Dict[str, Camera],
    gt_mask_dir: str,
    label_names: Dict[int, str],
) -> dict:
    """
    Walks gt_mask_dir/<image_name>/<object_id>.png, renders the matching
    predicted mask, and aggregates IoU/precision/recall per view and per
    object (guide section 21.2's metric table + 21.3's experiments).

    Returns
    -------
    {object_name: {"n_views": int,
                   "mean_iou"/"mean_precision"/"mean_recall": float,
                   "per_view": {image_name: {iou, precision, recall}}}}
    """
    from PIL import Image

    gt_root = Path(gt_mask_dir)
    per_object_scores: Dict[int, list] = {}
    per_view: Dict[int, dict] = {}

    for view_dir in sorted(gt_root.iterdir()):
        if not view_dir.is_dir():
            continue
        image_name = view_dir.name
        if image_name not in cameras:
            print(f"[eval] skipping {image_name}: no matching camera")
            continue
        camera = cameras[image_name]

        for gt_file in sorted(view_dir.glob("*.png")):
            object_id = int(gt_file.stem)
            gt_mask = np.array(Image.open(gt_file).convert("L")) > 127

            pred_mask = render_predicted_mask(cloud, labels, object_id, camera)
            scores = iou_precision_recall(pred_mask, gt_mask)
            per_object_scores.setdefault(object_id, []).append(scores)
            per_view.setdefault(object_id, {})[image_name] = scores

    summary = {}
    for object_id, score_list in per_object_scores.items():
        name = label_names.get(object_id, f"object_{object_id}")
        ious = [s["iou"] for s in score_list if not np.isnan(s["iou"])]
        precs = [s["precision"] for s in score_list if not np.isnan(s["precision"])]
        recs = [s["recall"] for s in score_list if not np.isnan(s["recall"])]
        summary[name] = {
            "n_views": len(score_list),
            "mean_iou": float(np.mean(ious)) if ious else float("nan"),
            "mean_precision": float(np.mean(precs)) if precs else float("nan"),
            "mean_recall": float(np.mean(recs)) if recs else float("nan"),
            "per_view": per_view.get(object_id, {}),
        }
    return summary
