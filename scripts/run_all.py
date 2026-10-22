"""
run_all.py
----------
Master verification runner. Executes, in order:

  1. the unit test suite (pytest)
  2. a synthetic-but-geometrically-consistent end-to-end exercise:
     fixture cameras + fixture Gaussian cloud + projected segmentation masks
     -> real 2D->3D association for all three voting strategies
     -> real projected-back evaluation against the fixture ground truth
  3. a printed comparison table (single-view vs majority vs weighted)

All numbers come from the real algorithm; nothing is invented. The multi-view
disk pipeline (COLMAP + trained PLY + SAM2 masks) is exercised the same way
by `python -m integration.run_semantic_pipeline --config configs/config.yaml`.

Usage:  python scripts/run_all.py [--skip-tests]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.association import assign_gaussian_labels          # noqa: E402
from integration.colmap_io import Camera                            # noqa: E402
from integration.eval_semantic import evaluate_scene, render_predicted_mask  # noqa: E402
from scripts.make_fixture import (                                  # noqa: E402
    CLUSTERS, build_cameras, build_cloud, build_gt_masks, build_mask_scores,
    build_masks,
)


def run_fixture_study() -> dict:
    print("=" * 64)
    print("STEP 2: synthetic end-to-end study (real algorithm, fixture scene)")
    print("=" * 64)

    cloud, gt_labels = build_cloud(seed=42)
    cameras = build_cameras()
    masks = build_masks(cloud, gt_labels, cameras)
    scores = build_mask_scores(masks)
    print(f"fixture: {cloud.n} Gaussians, {len(cameras)} cameras, "
          f"{sum(len(v) for v in masks.values())} object masks")

    results = {}
    for strategy in ("single_view", "majority", "weighted"):
        labels, confidence, n_supporting, used_views, stats = assign_gaussian_labels(
            cloud, cameras, masks, mask_scores=scores,
            strategy=strategy, min_views=2 if strategy != "single_view" else 1,
        )
        n_labeled = int((labels != -1).sum())
        per_object = {spec["label"]: int((labels == obj_id).sum())
                      for obj_id, spec in CLUSTERS.items()}
        results[strategy] = {"labels": labels, "confidence": confidence,
                             "n_labeled": n_labeled, "per_object": per_object}
        print(f"\n  strategy={strategy:12s} labeled {n_labeled}/{cloud.n} "
              f"({100.0 * n_labeled / cloud.n:.1f}%)  per-object: {per_object}")

        # label correctness against the fixture's ground truth
        known = gt_labels != -1
        correct = int((labels[known] == gt_labels[known]).sum())
        print(f"  accuracy on object Gaussians: {correct}/{int(known.sum())} "
              f"({100.0 * correct / max(1, int(known.sum())):.1f}%)")

    # projected-back evaluation of the weighted run against strict pixel GT
    print("\n  projected-back evaluation (strategy=weighted):")
    gt_masks = build_gt_masks(cloud, gt_labels, cameras)
    eval_cameras = cameras
    for obj_id, label in ((1, "chair"), (2, "table")):
        ious, precs, recs = [], [], []
        from integration.eval_semantic import iou_precision_recall
        for name, cam in eval_cameras.items():
            pred = render_predicted_mask(cloud, results["weighted"]["labels"], obj_id, cam)
            scores_obj = iou_precision_recall(pred, gt_masks[name][obj_id])
            ious.append(scores_obj["iou"]); precs.append(scores_obj["precision"]); recs.append(scores_obj["recall"])
        import numpy as np
        print(f"    {label:6s} IoU={np.mean(ious):.3f} precision={np.mean(precs):.3f} "
              f"recall={np.mean(recs):.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    if not args.skip_tests:
        print("=" * 64)
        print("STEP 1: unit test suite")
        print("=" * 64)
        import pytest
        rc = pytest.main(["-q", str(ROOT / "tests")])
        if rc != 0:
            sys.exit("unit tests failed -- fix before running the study")

    run_fixture_study()

    print()
    print("=" * 64)
    print("ALL STAGES COMPLETED")
    print("=" * 64)


if __name__ == "__main__":
    main()
