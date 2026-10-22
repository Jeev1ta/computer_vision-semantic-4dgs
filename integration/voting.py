"""
voting.py
---------
Multi-view semantic voting (guide section 17) -- the core research
contribution of the project ("does aggregating votes across calibrated
views improve 2D->3D label transfer over a single view?").

Implements the three systems from the ablation table in guide section 23:

    Baseline A  -- single-view lifting            (n_views_used = 1)
    Baseline B  -- multi-view majority voting      (equal weights)
    Proposed    -- confidence-weighted voting      (per-view weights)

All functions operate on a per-Gaussian "vote matrix":
    votes[g, c] in {-1, 0, 1, ..., K-1}
        -1  = camera c did not vote for gaussian g (occluded / out of frame /
              not inside any object mask)
        k>=0 = camera c voted for object id k

This dense representation is simple to reason about and test; for very
large scenes it can be swapped for a sparse (gaussian_id, cam_id, label,
weight) table without changing the public API below.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


UNKNOWN_LABEL = -1  # "a good semantic system should be allowed to say unknown" (guide 17)


def majority_vote(labels: Sequence[int]):
    """
    Scalar reference implementation (guide section 28), used directly by
    unit tests and as the ground-truth spec for the vectorized version below.
    Ties are broken deterministically by the lowest label id (Counter keeps
    first-inserted order, so we pick the smallest tied id explicitly).
    """
    labels = [x for x in labels if x != UNKNOWN_LABEL]
    if not labels:
        return UNKNOWN_LABEL, 0.0
    counts = Counter(labels)
    best = max(counts.values())
    winners = sorted(l for l, c in counts.items() if c == best)
    return winners[0], counts[winners[0]] / len(labels)


@dataclass
class VotingConfig:
    min_views: int = 1              # "require a minimum number of supporting views" (guide 17)
    min_mask_area: int = 25         # "ignore masks with extremely tiny area"
    unknown_label: int = UNKNOWN_LABEL


def majority_vote_batch(
    votes: np.ndarray,
    min_views: int = 1,
    tie_weights: Optional[np.ndarray] = None,
):
    """
    Baseline B: equal-weight multi-view majority voting, vectorized.

    Parameters
    ----------
    votes : (N, C) int array, entries are object ids or UNKNOWN_LABEL.
    min_views : minimum number of *non-unknown* votes required before a
        label is assigned (otherwise the Gaussian stays UNKNOWN_LABEL).
    tie_weights : optional (N, C) array of per-vote evidence weights. When two
        labels receive the same number of votes, the one whose votes carry the
        higher total weight wins; remaining ties go to the lowest label id.
        Without it, ties are broken by lowest label id (deterministic).

    Returns
    -------
    labels : (N,) int array of assigned labels
    confidence : (N,) float array in [0, 1] = winning_votes / total_votes
    n_supporting : (N,) int array = number of non-unknown votes seen
    """
    n_gaussians, n_cams = votes.shape
    max_label = int(votes.max()) if votes.size else -1
    n_classes = max_label + 1  # labels are 0..max_label

    labels = np.full(n_gaussians, UNKNOWN_LABEL, dtype=np.int64)
    confidence = np.zeros(n_gaussians, dtype=np.float64)
    n_supporting = (votes != UNKNOWN_LABEL).sum(axis=1)

    if n_classes <= 0:
        return labels, confidence, n_supporting

    # bincount-per-row via one-hot sum -- fine for a semester-scale scene
    # (hundreds of thousands of Gaussians x tens of cameras x few objects).
    onehot_counts = np.zeros((n_gaussians, n_classes), dtype=np.int64)
    for c in range(n_classes):
        onehot_counts[:, c] = (votes == c).sum(axis=1)

    if tie_weights is not None:
        assert tie_weights.shape == votes.shape
        weighted_sums = np.zeros((n_gaussians, n_classes), dtype=np.float64)
        for c in range(n_classes):
            weighted_sums[:, c] = np.where(votes == c, tie_weights, 0.0).sum(axis=1)
        # lexicographic max: count first, then summed weight; argmax picks the
        # first (lowest id) among exact ties -> fully deterministic.
        order = onehot_counts * 1e6 + weighted_sums
    else:
        order = onehot_counts

    best_label = order.argmax(axis=1)
    best_count = onehot_counts.max(axis=1)

    has_support = (n_supporting >= min_views) & (best_count > 0)
    labels[has_support] = best_label[has_support]
    with np.errstate(divide="ignore", invalid="ignore"):
        conf = np.where(n_supporting > 0, best_count / np.maximum(n_supporting, 1), 0.0)
    confidence[has_support] = conf[has_support]

    return labels, confidence, n_supporting


def confidence_weighted_vote_batch(
    votes: np.ndarray,
    weights: np.ndarray,
    min_views: int = 1,
    min_total_weight: float = 0.0,
):
    """
    Proposed method: confidence-weighted multi-view voting.

    weights : (N, C) float array, same shape as `votes`. Each entry is the
        per-(gaussian, camera) evidence weight, e.g.:
            weight = mask_confidence * projection_quality * visibility_weight
        (guide section 17, "Better version: visibility/confidence weighting").
        A weight of 0 for a camera that didn't vote is fine and expected.

    Ties in total weighted score are broken by vote count (more cameras
    agreeing is stronger), then by lowest label id -- deterministic.

    Returns
    -------
    labels, confidence, n_supporting  (same shapes/semantics as majority_vote_batch;
        `confidence` here is winning_weight / total_weight instead of a raw
        vote fraction)
    """
    assert votes.shape == weights.shape, "votes and weights must be the same shape"
    n_gaussians, n_cams = votes.shape
    max_label = int(votes.max()) if votes.size else -1
    n_classes = max_label + 1

    labels = np.full(n_gaussians, UNKNOWN_LABEL, dtype=np.int64)
    confidence = np.zeros(n_gaussians, dtype=np.float64)
    n_supporting = (votes != UNKNOWN_LABEL).sum(axis=1)

    if n_classes <= 0:
        return labels, confidence, n_supporting

    weighted_scores = np.zeros((n_gaussians, n_classes), dtype=np.float64)
    class_counts = np.zeros((n_gaussians, n_classes), dtype=np.int64)
    for c in range(n_classes):
        mask_c = (votes == c)
        weighted_scores[:, c] = np.where(mask_c, weights, 0.0).sum(axis=1)
        class_counts[:, c] = mask_c.sum(axis=1)

    total_weight = weighted_scores.sum(axis=1)
    # lexicographic max: weighted score first, then vote count, then lowest id
    order = weighted_scores + class_counts * 1e-9
    best_label = order.argmax(axis=1)
    best_score = weighted_scores.max(axis=1)

    has_support = (
        (n_supporting >= min_views)
        & (best_score > 0)
        & (total_weight >= min_total_weight)
    )
    labels[has_support] = best_label[has_support]
    with np.errstate(divide="ignore", invalid="ignore"):
        conf = np.where(total_weight > 0, best_score / np.maximum(total_weight, 1e-12), 0.0)
    confidence[has_support] = conf[has_support]

    return labels, confidence, n_supporting


def single_view_labels(votes: np.ndarray, view_index: Optional[int] = None):
    """
    Baseline A: label each Gaussian purely from ONE chosen camera's vote
    (no aggregation at all). Useful as the weakest-baseline comparison
    point in the ablation table (guide section 23).

    view_index : which camera column to use. If None, the view with the most
        non-unknown votes (i.e. the most object coverage) is selected --
        previously this defaulted to column 0, which silently depended on
        dict insertion order of the masks. Pass an explicit index to pin it.

    NOTE: `min_views` does not apply to this baseline -- a single view can
    contribute at most one supporting vote per Gaussian, so requiring >=2
    would label nothing. The ablation runner passes min_views=1 here.
    """
    if view_index is None:
        support_per_view = (votes != UNKNOWN_LABEL).sum(axis=0)
        view_index = int(np.argmax(support_per_view)) if len(support_per_view) else 0
    col = votes[:, view_index]
    labels = col.copy()
    confidence = np.where(col != UNKNOWN_LABEL, 1.0, 0.0)
    n_supporting = (col != UNKNOWN_LABEL).astype(np.int64)
    return labels, confidence, n_supporting, view_index


def compute_projection_weight(
    scale: Optional[np.ndarray],
    distance_to_camera: np.ndarray,
    mask_confidence: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Optional helper implementing the "projection_quality" term from guide
    section 17's weighting formula. Kept simple and explainable for a
    semester report:

        weight = mask_confidence * (1 / (1 + normalized_distance))
                 * (mean_gaussian_scale, if available, else 1.0)

    Larger/closer Gaussians with more confident masks contribute more.
    All inputs are (N,) arrays; returns (N,) weight array in roughly [0, 1].
    """
    n = len(distance_to_camera)
    dist = np.asarray(distance_to_camera, dtype=np.float64)
    dist_norm = dist / (np.median(dist[dist > 0]) + 1e-8) if np.any(dist > 0) else np.ones(n)
    distance_term = 1.0 / (1.0 + dist_norm)

    if scale is not None:
        mean_scale = scale.mean(axis=1)
        scale_term = mean_scale / (np.median(mean_scale) + 1e-8)
        scale_term = np.clip(scale_term, 0.1, 3.0)
    else:
        scale_term = np.ones(n)

    conf_term = mask_confidence if mask_confidence is not None else np.ones(n)

    weight = conf_term * distance_term * scale_term
    return np.clip(weight, 0.0, None)
