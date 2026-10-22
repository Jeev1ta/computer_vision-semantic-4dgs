"""
association.py
---------------
Implements guide section 16's `assign_gaussian_labels()` end to end, using
projection.py + mask_io.py + voting.py. This is "the part you actually
write" (guide section 13): it fuses the geometry from 3DGS with the
semantics from SAM2.

Pipeline per camera c:
    1. Project all N Gaussian centers into camera c              (projection.py)
    2. For each object mask visible in camera c's image,
       sample the mask at each projected pixel                   (projection.py)
    3. Record a vote (object_id) for every Gaussian whose
       projected pixel fell inside that mask                     -> votes[:, c]
    4. Optionally record a per-vote weight (mask score,
       distance-to-camera, Gaussian scale, ...)                   -> weights[:, c]

After all cameras are processed, `voting.py` aggregates votes across the
camera axis into a single label + confidence per Gaussian.

Overlap policy: a Gaussian whose pixel falls inside SEVERAL object masks in
the same view casts its vote for the most credible mask -- higher SAM2 score
first, larger mask area as tie-break, lowest object id as final tie-break.
The previous behavior let Python dict iteration order decide, which made
results depend on the order masks happened to be loaded in.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import projection
from . import voting
from .colmap_io import Camera
from .gaussian_io import GaussianCloud

UNKNOWN_LABEL = voting.UNKNOWN_LABEL


def _align_mask_to_camera(mask: np.ndarray, cam: Camera) -> np.ndarray:
    """
    Resize a binary mask to the camera's exact (height, width) if they differ.
    Masks come from the image files while COLMAP records the (possibly resized)
    camera resolution; a mismatch would silently shift every lookup.
    """
    target = (cam.height, cam.width)
    if mask.shape == target:
        return mask
    try:
        import cv2
        return cv2.resize(mask.astype(np.uint8), (target[1], target[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    except ImportError:
        from PIL import Image
        img = Image.fromarray(mask.astype(np.uint8) * 255)
        img = img.resize((target[1], target[0]), Image.NEAREST)
        return np.array(img) > 127


def build_vote_matrix(
    cloud: GaussianCloud,
    cameras: Dict[str, Camera],
    masks: Dict[str, Dict[int, np.ndarray]],
    mask_scores: Optional[Dict[str, Dict[int, float]]] = None,
    min_mask_area: int = 25,
) -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
    """
    Projects every Gaussian into every camera that has at least one mask,
    and builds the dense (N, C) vote / weight matrices consumed by voting.py.

    Parameters
    ----------
    cloud : loaded via gaussian_io.load_gaussian_cloud
    cameras : loaded via colmap_io.load_cameras -- keyed by image filename
    masks : loaded via mask_io.load_masks -- masks[image_name][object_id] -> HxW bool
    mask_scores : optional masks[image_name][object_id] -> float SAM2 confidence,
                  used to weight votes and resolve overlaps (guide section 17)
    min_mask_area : masks with fewer True pixels than this are ignored

    Returns
    -------
    votes      : (N, C) int64, C = number of camera views that had at least one mask
    weights    : (N, C) float64, per-vote evidence weight (1.0 if no weighting info given)
    used_views : list of image filenames; columns of votes/weights line up with this
    stats      : per-view diagnostics (masks used/ignored, per-object hit counts,
                 fraction of Gaussians that received a vote)
    """
    n = cloud.n
    used_views = [name for name in masks.keys() if name in cameras]
    skipped_views = [name for name in masks.keys() if name not in cameras]
    if not used_views:
        raise ValueError(
            "None of the masked images have a matching camera. "
            "Mask filenames must match the COLMAP image names exactly "
            f"(masked views: {sorted(masks.keys())}; cameras: {sorted(cameras.keys())})."
        )

    votes = np.full((n, len(used_views)), UNKNOWN_LABEL, dtype=np.int64)
    weights = np.zeros((n, len(used_views)), dtype=np.float64)
    stats: dict = {"views": {}, "skipped_views_no_camera": skipped_views}

    for col, image_name in enumerate(used_views):
        cam = cameras[image_name]
        uv, visible = projection.project_points_batch(
            cloud.xyz, cam.R, cam.t, cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height
        )

        # per-camera distance term, used later as a simple weighting signal
        cam_center = cam.camera_center()
        dist = np.linalg.norm(cloud.xyz - cam_center, axis=1)

        view_stat = {"masks_used": 0, "masks_ignored_tiny": 0, "masks_resized": 0, "per_object_hits": {}}
        # overlap resolution state for this view
        best_score = np.full(n, -1.0)
        best_area = np.full(n, -1.0)

        for object_id, mask in masks[image_name].items():
            if mask.shape != (cam.height, cam.width):
                mask = _align_mask_to_camera(mask, cam)
                view_stat["masks_resized"] += 1
            mask_area = int(mask.sum())
            if mask_area < min_mask_area:
                view_stat["masks_ignored_tiny"] += 1
                continue

            hit = projection.sample_mask_batch(mask, uv, visible)
            if not hit.any():
                continue

            score = 1.0
            if mask_scores and image_name in mask_scores and object_id in mask_scores[image_name]:
                score = float(mask_scores[image_name][object_id])

            # per-vote evidence weight for every Gaussian inside this mask
            w = voting.compute_projection_weight(
                scale=cloud.scale[hit] if cloud.scale is not None else None,
                distance_to_camera=dist[hit],
                mask_confidence=np.full(int(hit.sum()), score),
            )

            # overlap resolution: score first, then larger mask, then lower id
            # (object ids arrive in ascending order, so strict > keeps the
            # lowest id on exact ties)
            take = hit & (
                (best_score < 0)
                | (score > best_score)
                | ((score == best_score) & (mask_area > best_area))
            )
            votes[take, col] = object_id
            best_score[take] = score
            best_area[take] = float(mask_area)

            # w is computed for the *hit* subset; keep the entries belonging
            # to the resolved winners
            weights[take, col] = w[take[hit]]

            view_stat["masks_used"] += 1
            view_stat["per_object_hits"][str(object_id)] = int(hit.sum())

        view_stat["vote_fraction"] = float((votes[:, col] != UNKNOWN_LABEL).mean())
        stats["views"][image_name] = view_stat

    return votes, weights, used_views, stats


def assign_gaussian_labels(
    cloud: GaussianCloud,
    cameras: Dict[str, Camera],
    masks: Dict[str, Dict[int, np.ndarray]],
    mask_scores: Optional[Dict[str, Dict[int, float]]] = None,
    strategy: str = "weighted",   # "single_view" | "majority" | "weighted"
    min_views: int = 1,
    min_opacity: float = 0.0,
):
    """
    Full end-to-end association (guide section 16). Returns:
        labels       : (N,) int64, UNKNOWN_LABEL where no confident label
        confidence   : (N,) float64 in [0, 1]
        n_supporting : (N,) int64, number of views that voted (any class)
        used_views   : list[str], camera image names used as vote columns
        stats        : dict, per-view diagnostics from build_vote_matrix

    Notes
    -----
    * `min_views` applies to the majority/weighted strategies. For
      `single_view` it is ignored (a single view contributes at most one
      supporting vote, so requiring more would label nothing).
    * Low-opacity Gaussians are filtered BEFORE voting (their projections
      are unreliable) and keep UNKNOWN_LABEL.
    """
    if min_opacity > 0:
        keep = cloud.filter_by_opacity(min_opacity)
    else:
        keep = np.ones(cloud.n, dtype=bool)

    votes_full, weights_full, used_views, stats = build_vote_matrix(
        cloud, cameras, masks, mask_scores
    )

    votes = votes_full[keep]
    weights = weights_full[keep]

    if strategy == "single_view":
        labels_k, confidence_k, n_sup_k, chosen_view = voting.single_view_labels(votes)
        stats["single_view_chosen"] = used_views[chosen_view] if used_views else None
    elif strategy == "majority":
        # vote counts break ties; per-vote weights only order exact ties
        labels_k, confidence_k, n_sup_k = voting.majority_vote_batch(
            votes, min_views=min_views, tie_weights=weights
        )
    elif strategy == "weighted":
        labels_k, confidence_k, n_sup_k = voting.confidence_weighted_vote_batch(
            votes, weights, min_views=min_views
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    labels = np.full(cloud.n, UNKNOWN_LABEL, dtype=np.int64)
    confidence = np.zeros(cloud.n, dtype=np.float64)
    n_supporting = np.zeros(cloud.n, dtype=np.int64)

    labels[keep] = labels_k
    confidence[keep] = confidence_k
    n_supporting[keep] = n_sup_k

    return labels, confidence, n_supporting, used_views, stats
