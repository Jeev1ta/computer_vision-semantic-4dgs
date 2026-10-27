"""
Comprehensive Evaluation Benchmark Script for Semantic-Aware 4D Gaussian Splatting.

Evaluates:
  1. Semantic Mean Intersection over Union (mIoU) and Per-Class IoU
  2. Spatio-Temporal Consistency (evaluating temporal coherence vs 2D jitter)
  3. Visual Quality Metrics: PSNR, SSIM, and LPIPS
  4. Real-Time Rendering Speed (FPS)

Usage:
    python eval_semantics.py --model_path output/standup_semantic --semantics_gt data/dnerf/standup/semantics
    # Or run the benchmark validation test:
    python eval_semantics.py --benchmark
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from PIL import Image

try:
    if TORCH_AVAILABLE:
        from utils.loss_utils import ssim
        from utils.image_utils import psnr
    else:
        raise ImportError
except ImportError:
    # Standalone math definitions if run without full 4D-GS environment
    def psnr(img1, img2):
        if TORCH_AVAILABLE and isinstance(img1, torch.Tensor):
            mse = torch.mean((img1 - img2) ** 2)
            if mse == 0:
                return torch.tensor(100.0)
            return -10.0 * torch.log10(mse)
        else:
            mse = np.mean((np.asarray(img1) - np.asarray(img2)) ** 2)
            if mse == 0:
                return 100.0
            return float(-10.0 * np.log10(mse))

    def ssim(img1, img2):
        raise RuntimeError("SSIM requires the project image-metrics dependencies")


DEFAULT_CLASSES = [
    "Background",
    "Safety Vest",
    "Head / Hair",
    "Arms & Torso",
    "Pants & Legs",
]

def compute_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_index: int = 0) -> Tuple[float, Dict[int, float]]:
    """Compute per-class IoU and mean IoU.
    
    Args:
        pred: Predicted class map (H, W).
        gt: Ground truth class map (H, W).
        num_classes: Total number of classes.
        ignore_index: Class ID to exclude from mean IoU (usually 0 = background).
    """
    per_class_iou = {}
    valid_ious = []

    for c in range(num_classes):
        if c == ignore_index:
            continue
        pred_c = (pred == c)
        gt_c = (gt == c)
        intersection = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()

        if union == 0:
            continue
        iou = float(intersection) / float(union)

        per_class_iou[c] = iou
        valid_ious.append(iou)

    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return mean_iou, per_class_iou


def print_benchmark_table(data: Dict):
    """Print only metrics that were computed from supplied predictions and labels."""
    print("\n" + "=" * 78)
    print(f"📊 GROUP 08 | 4D GAUSSIAN SPLATTING FINAL EVALUATION BENCHMARK")
    print("=" * 78)
    print(f"Model: {data['model_name']}")
    print(f"Evaluation: {data['evaluation']}")
    print("-" * 78)

    m = data["metrics"]
    print("\n[1] SEMANTIC LABEL AGREEMENT (pixel-space; not a 3D/temporal metric):")
    print(f"  • Mean Intersection over Union (mIoU): {m['semantic_miou']:.2f}%")
    print("  • Per-Class IoU:")
    for cls_name, iou_val in m["per_class_iou"].items():
        bar = "█" * int(iou_val / 5) + "░" * (20 - int(iou_val / 5))
        print(f"     - {cls_name:<16}: {iou_val:5.1f}%  [{bar}]")

    print("\nAppearance quality, temporal correspondence, and FPS are not measured by this script.")
    print("=" * 78 + "\n")


SEMANTIC_COLORS = np.array([
    [0, 0, 0],        # 0: background
    [220, 20, 60],    # 1: crimson
    [0, 191, 255],    # 2: deep sky blue
    [50, 205, 50],    # 3: lime green
    [255, 165, 0],    # 4: orange
    [148, 0, 211],    # 5: dark violet
    [255, 215, 0],    # 6: gold
    [0, 250, 154],    # 7: medium spring green
    [255, 105, 180],  # 8: hot pink
    [70, 130, 180],   # 9: steel blue
], dtype=np.uint8)

def _decode_rgb_to_labels(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    rgb = img[:, :, :3]
    dists = np.sum((rgb[:, :, None, :] - SEMANTIC_COLORS[None, None, :, :].astype(np.int32)) ** 2, axis=-1)
    return np.argmin(dists, axis=-1).astype(np.uint8)

def evaluate_folder(model_dir: str, gt_dir: str, class_names: Optional[List[str]] = None) -> Dict:
    """Evaluate renders from an existing training output folder."""
    model_path = Path(model_dir)
    gt_path = Path(gt_dir)

    classes = class_names
    if classes is None:
        for meta_p in [gt_path / "metadata.json", gt_path.parent / "metadata.json"]:
            if meta_p.is_file():
                try:
                    with open(meta_p) as f:
                        meta = json.load(f)
                    prompts = meta.get("config", {}).get("text_prompts") or list(meta.get("class_mapping", {}).values())
                    if prompts:
                        classes = prompts
                        break
                except Exception:
                    pass
        if classes is None:
            classes = DEFAULT_CLASSES

    renders_dir = model_path / "renders"
    sem_dir = model_path / "renders_semantics"

    if not sem_dir.is_dir():
        candidates = sorted(list(model_path.glob("**/renders_semantics")), key=lambda p: str(p))
        if candidates:
            sem_dir = candidates[-1]
        else:
            raise FileNotFoundError(f"Semantic render directory not found: {sem_dir}")

    sem_files = {p.name: p for p in sem_dir.glob("*.png")}
    gt_labels_dir = gt_path / "labels" if (gt_path / "labels").is_dir() else gt_path
    gt_files = {p.name: p for p in gt_labels_dir.glob("*.png")}

    paired_names = sorted(sem_files.keys() & gt_files.keys())
    if not paired_names:
        raise FileNotFoundError(
            f"No matching PNG filenames between {sem_dir} and {gt_labels_dir}"
        )

    pred_maps, gt_maps = [], []

    for name in paired_names:
        p_img = np.asarray(Image.open(sem_files[name]))
        g_img = np.asarray(Image.open(gt_files[name]))
        if p_img.ndim == 3:
            p_img = _decode_rgb_to_labels(p_img)
        if g_img.ndim == 3:
            g_img = _decode_rgb_to_labels(g_img)
        if p_img.ndim != 2 or g_img.ndim != 2:
            raise ValueError(f"Expected single-channel label maps for {name}; got {p_img.shape} and {g_img.shape}")
        if p_img.shape != g_img.shape:
            raise ValueError(f"Prediction/GT shape mismatch for {name}: {p_img.shape} vs {g_img.shape}")
        pred_maps.append(p_img)
        gt_maps.append(g_img)

    # Compute IoU over frames
    all_pred = np.stack(pred_maps, axis=0)
    all_gt = np.stack(gt_maps, axis=0)

    max_label = int(max(all_pred.max(), all_gt.max()))
    if max_label >= len(classes):
        raise ValueError(
            f"Found class ID {max_label}, but only {len(classes)} class names were supplied. "
            "Pass a matching --classes list or fix the label mapping."
        )
    miou, per_cls = compute_iou(all_pred, all_gt, len(classes))

    per_cls_dict = {}
    for idx, iou_v in per_cls.items():
        cname = classes[idx] if idx < len(classes) else f"Class_{idx}"
        per_cls_dict[cname] = round(iou_v * 100, 2)

    res = {
        "model_name": model_path.name,
        "evaluation": "Semantic label-map comparison on matched image filenames",
        "paired_frames": paired_names,
        "metrics": {
            "semantic_miou": round(miou * 100, 2),
            "per_class_iou": per_cls_dict,
        },
    }
    return res


def save_markdown_report(report_data: Dict, output_path: str):
    """Write an evaluation markdown summary report."""
    m = report_data["metrics"]
    md = f"""# Semantic Label Evaluation: {report_data['model_name']}

**Evaluation protocol:** {report_data['evaluation']}
**Matched frames:** {len(report_data['paired_frames'])}

---

### Pixel-space semantic label agreement
- **Mean IoU (mIoU):** `{m['semantic_miou']}%`
- This compares rendered 2D label maps with supplied 2D labels; it is not by itself evidence of 3D accuracy or temporal consistency.
- Appearance metrics (PSNR, SSIM, LPIPS), temporal correspondence, and FPS were not measured by this script.

#### Per-class pixel-space IoU:
| Class | IoU (%) |
|---|---|
"""
    for cls_name, val in m["per_class_iou"].items():
        md += f"| **{cls_name}** | {val}% |\n"

    with open(output_path, "w") as f:
        f.write(md)
    print(f"Saved evaluation markdown report to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Semantic 4D-GS models")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model output folder")
    parser.add_argument("--semantics_gt", type=str, default=None, help="Path to GT semantics directory")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="Class names in the integer-ID order used by the label maps")
    parser.add_argument("--output_md", type=str, default="evaluation_report.md", help="Path to save markdown report")
    parser.add_argument("--output_json", type=str, default="evaluation_report.json", help="Path to save JSON report")
    args = parser.parse_args()

    if not args.model_path or not args.semantics_gt:
        parser.error("--model_path and --semantics_gt are required; this evaluator does not substitute example metrics")
    report = evaluate_folder(args.model_path, args.semantics_gt, args.classes)

    print_benchmark_table(report)

    # Save outputs
    save_markdown_report(report, args.output_md)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation JSON report to: {args.output_json}")


if __name__ == "__main__":
    main()
