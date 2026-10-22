"""
scripts/run_evaluation.py
---------------------------
Runs the projected-back semantic evaluation (guide section 21.2 / 22)
against a hand-annotated ground-truth set produced by
scripts/annotate_ground_truth.py, for one or more semantic runs
(e.g. every system in an ablation sweep).

Usage:
    python scripts/run_evaluation.py --config configs/config.yaml \
        --gt_dir outputs/eval/gt_masks \
        --semantic_dir outputs/semantic/my_scene
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from integration.colmap_io import load_cameras          # noqa: E402
from integration.gaussian_io import GaussianCloud         # noqa: E402
from integration.eval_semantic import evaluate_scene      # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--gt_dir", required=True, help="Ground truth dir from annotate_ground_truth.py")
    p.add_argument("--semantic_dir", required=True, help="Output dir from run_semantic_pipeline.py")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    semantic_dir = Path(args.semantic_dir)
    xyz = np.load(semantic_dir / "gaussian_xyz.npy")
    labels = np.load(semantic_dir / "gaussian_labels.npy")
    with open(semantic_dir / "object_metadata.json") as f:
        meta = json.load(f)
    label_names = {obj["id"]: obj["label"] for obj in meta["objects"]}

    cameras = load_cameras(cfg["paths"]["dataset_dir"],
                            gaussian_splatting_repo=cfg["paths"]["gaussian_splatting_repo"])

    cloud = GaussianCloud(xyz=xyz, opacity=np.ones(len(xyz)), scale=None, rotation=None, n=len(xyz))

    summary = evaluate_scene(cloud, labels, cameras, args.gt_dir, label_names)

    print("\n=== Semantic evaluation (guide section 21.2) ===")
    print(f"{'object':15s} {'n_views':>8s} {'mIoU':>8s} {'precision':>10s} {'recall':>8s}")
    for name, s in summary.items():
        print(f"{name:15s} {s['n_views']:8d} {s['mean_iou']:8.3f} "
              f"{s['mean_precision']:10.3f} {s['mean_recall']:8.3f}")

    out_path = semantic_dir / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
