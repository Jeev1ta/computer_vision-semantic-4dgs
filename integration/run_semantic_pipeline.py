"""
run_semantic_pipeline.py
-------------------------
The single command that ties the whole bridge together (guide section 16's
`if __name__ == "__main__":` block, fully implemented, plus the ablation
sweep from section 23).

Usage:
    python -m integration.run_semantic_pipeline --config configs/config.yaml
    python -m integration.run_semantic_pipeline --config configs/config.yaml --ablation

Prerequisites (see README.md for the full step order):
    1. 3DGS trained:            outputs/3dgs/<scene>/point_cloud/iteration_*/point_cloud.ply
    2. COLMAP data converted:   data/<scene>/sparse/0/{cameras,images}.bin
    3. SAM2 masks generated:    outputs/masks/<scene>/{masks,metadata}/  (via sam2_runner.py)

This script itself has NO CUDA/torch dependency -- it only needs numpy,
plyfile, opencv/pillow and pyyaml, so it can run in a lightweight
"integration" environment separate from both the 3DGS and SAM2 envs
(guide section 5's recommended machine architecture).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .colmap_io import load_cameras
from .gaussian_io import load_gaussian_cloud, find_latest_ply
from .mask_io import load_masks, load_mask_scores, load_object_metadata, write_semantic_object_metadata
from .association import assign_gaussian_labels
from .visualize import save_per_object_previews, plot_topdown


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _safe_output_dir(path_str) -> Path:
    """Validate a config-supplied output path: forbid '..' traversal components
    so a crafted config can never redirect writes outside the project tree."""
    p = Path(str(path_str))
    if ".." in p.parts:
        raise ValueError(f"Path traversal forbidden in config path: {path_str!r}")
    return p


def run_once(cfg: dict, strategy_override: str = None, min_views_override: int = None,
             out_subdir: str = None):
    paths = cfg["paths"]
    assoc_cfg = cfg["association"]

    strategy = strategy_override or assoc_cfg.get("strategy", "weighted")
    min_views = min_views_override if min_views_override is not None else assoc_cfg.get("min_views", 1)
    # a single view can contribute at most one supporting vote per Gaussian,
    # so min_views>1 is meaningless (it would label nothing) -- pin to 1.
    effective_min_views = 1 if strategy == "single_view" else min_views

    # --- 1. Load trained Gaussians (guide section 15) ----------------------
    ply_path = find_latest_ply(paths["model_dir"])
    print(f"[pipeline] loading Gaussians from {ply_path}")
    cloud = load_gaussian_cloud(ply_path)
    print(f"[pipeline] {cloud.n} Gaussians loaded")

    # --- 2. Load COLMAP cameras (guide section 8/9) -------------------------
    print(f"[pipeline] loading cameras from {paths['dataset_dir']}")
    cameras = load_cameras(paths["dataset_dir"], gaussian_splatting_repo=paths["gaussian_splatting_repo"])
    print(f"[pipeline] {len(cameras)} cameras loaded")

    # --- 3. Load SAM2 masks (guide section 12) -------------------------------
    print(f"[pipeline] loading masks from {paths['mask_dir']}")
    masks = load_masks(paths["mask_dir"])
    mask_scores = load_mask_scores(paths["mask_dir"])
    object_meta = load_object_metadata(paths["mask_dir"])
    label_names = {obj_id: m["label"] for obj_id, m in object_meta.items()}
    print(f"[pipeline] {len(masks)} masked views, {len(label_names)} objects: {label_names}")

    # --- 4. 2D -> 3D association (guide section 13-17) -----------------------
    labels, confidence, n_supporting, used_views, stats = assign_gaussian_labels(
        cloud, cameras, masks,
        mask_scores=mask_scores,
        strategy=strategy,
        min_views=effective_min_views,
        min_opacity=assoc_cfg.get("min_opacity", 0.0),
    )

    n_labeled = int((labels != -1).sum())
    print(f"[pipeline] strategy={strategy} (min_views={effective_min_views}): "
          f"{n_labeled}/{cloud.n} Gaussians labeled "
          f"({100.0 * n_labeled / max(cloud.n, 1):.1f}%)")
    for obj_id, name in label_names.items():
        count = int((labels == obj_id).sum())
        print(f"    - {name} (id={obj_id}): {count} Gaussians")

    # --- 5. Save outputs (guide section 18) -----------------------------------
    # validated (no '..' components) before anything is created
    semantic_dir = _safe_output_dir(paths["semantic_dir"])
    if out_subdir:
        semantic_dir = semantic_dir / _safe_output_dir(out_subdir)
    semantic_dir.mkdir(parents=True, exist_ok=True)

    np.save(semantic_dir / "gaussian_xyz.npy", cloud.xyz)
    np.save(semantic_dir / "gaussian_labels.npy", labels)
    np.save(semantic_dir / "gaussian_confidence.npy", confidence)

    source_views = {
        obj_id: [f["source_frame"] for f in m.get("frames", []) if f["source_frame"] in used_views]
        for obj_id, m in object_meta.items()
    }
    write_semantic_object_metadata(
        semantic_dir / "object_metadata.json", label_names, source_views
    )

    # --- 6. Quick visualization outputs (guide section 19-20) ----------------
    save_per_object_previews(cloud.xyz, labels, str(semantic_dir), label_names)
    plot_topdown(cloud.xyz, labels, str(semantic_dir / "topdown_preview.png"), label_names)

    metrics = {
        "strategy": strategy,
        "min_views": effective_min_views,
        "n_gaussians": int(cloud.n),
        "n_labeled": n_labeled,
        "label_rate": n_labeled / max(cloud.n, 1),
        "per_object_counts": {name: int((labels == obj_id).sum()) for obj_id, name in label_names.items()},
        "n_cameras_used": len(used_views),
        "used_views": used_views,
        "per_view_stats": stats.get("views", {}),
    }
    if strategy == "single_view" and stats.get("single_view_chosen"):
        metrics["single_view_chosen"] = stats["single_view_chosen"]
    # semantic_dir was validated by _safe_output_dir above (no '..' components)
    (semantic_dir / "run_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[pipeline] done. Outputs in {semantic_dir}")
    return metrics


def run_ablation(cfg: dict):
    """Runs every system in cfg['ablation'] and writes a combined comparison table
    (guide section 23's baseline/proposed comparison, section 21.3's E1-E4)."""
    results = []
    for system in cfg.get("ablation", []):
        name = system["name"]
        print(f"\n=== Ablation system: {name} ===")
        metrics = run_once(
            cfg,
            strategy_override=system.get("strategy"),
            min_views_override=system.get("min_views"),
            out_subdir=f"ablation/{name}",
        )
        metrics["name"] = name
        results.append(metrics)

    out_path = Path(cfg["paths"]["semantic_dir"]) / "ablation" / "comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Ablation summary ===")
    print(f"{'system':30s} {'label_rate':>10s} {'n_labeled':>10s}")
    for r in results:
        print(f"{r['name']:30s} {r['label_rate']*100:9.1f}% {r['n_labeled']:10d}")
    print(f"\nFull comparison table written to {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run the semantic 3DGS 2D->3D association pipeline.")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--ablation", action="store_true",
                    help="Run all systems in configs/config.yaml's `ablation` list (guide section 23).")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.ablation:
        run_ablation(cfg)
    else:
        run_once(cfg)
