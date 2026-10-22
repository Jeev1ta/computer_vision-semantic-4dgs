"""
segmentation.py
---------------
Real image segmentation with tiered backends (guide section 11-12, upgraded):

    Tier 1  -- official SAM2 (facebookresearch/sam2) automatic mask generator,
               used automatically when third_party/sam2 + a checkpoint are
               available. Produces high-quality instance masks with scores.
    Tier 2  -- classical CPU segmentation (always available): Felzenszwalb
               graph-based region segmentation (Felzenszwalb & Huttenlocher,
               2004) refined with SLIC superpixels, morphological cleanup and
               area filtering. Deterministic, no GPU, no downloads.

Both tiers return the same structure:

    Segment = {"mask": (H,W) bool, "score": float, "area": int,
               "bbox": (x0,y0,x1,y1), "label": str, "backend": str}

so the rest of the pipeline (voting, association, dashboard) is agnostic to
which backend produced the masks. Labels from the classical tier are neutral
("object_1", "object_2", ...) -- renaming them with semantic names is a
user action, exactly like the config-prompt workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

_DEFAULT_SAM2_REPO = Path(__file__).resolve().parent.parent / "third_party" / "sam2"
_DEFAULT_CHECKPOINT = _DEFAULT_SAM2_REPO / "checkpoints" / "sam2.1_hiera_small.pt"
_DEFAULT_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"


@dataclass
class Segment:
    mask: np.ndarray          # (H, W) bool
    score: float              # backend confidence in [0, 1]
    area: int                 # True-pixel count
    bbox: tuple               # (x0, y0, x1, y1) inclusive-exclusive
    label: str
    backend: str              # "sam2" | "classical"


def _bbox_of(mask: np.ndarray) -> tuple:
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def sam2_available(sam2_repo: Path = _DEFAULT_SAM2_REPO,
                   checkpoint: Path = _DEFAULT_CHECKPOINT) -> bool:
    """True when the official SAM2 repo and a checkpoint file are both present
    AND the sam2 package imports successfully."""
    if not (Path(sam2_repo) / "sam2").is_dir():
        return False
    if not Path(checkpoint).exists():
        return False
    try:
        import sys
        repo = str(sam2_repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import sam2.build_sam  # noqa: F401
        return True
    except Exception:
        return False


def available_backends() -> dict:
    """Report which segmentation backends this machine can use right now."""
    torch_ok = False
    try:
        import torch  # noqa: F401
        torch_ok = True
    except Exception:
        torch_ok = False
    return {
        "sam2": sam2_available(),
        "torch": torch_ok,
        "classical": True,  # numpy + skimage/opencv only, always present here
    }


# ---------------------------------------------------------------------------
# Tier 1: official SAM2
# ---------------------------------------------------------------------------

def _segment_sam2(image_rgb: np.ndarray, max_objects: int) -> List[Segment]:
    import sys
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    repo = str(_DEFAULT_SAM2_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(_DEFAULT_CFG, str(_DEFAULT_CHECKPOINT), device=device)
    generator = SAM2AutomaticMaskGenerator(model)
    raw = generator.generate(image_rgb)

    raw = sorted(raw, key=lambda m: m["area"], reverse=True)[:max_objects]
    segments = []
    for i, m in enumerate(raw):
        mask = m["segmentation"].astype(bool)
        segments.append(Segment(
            mask=mask,
            score=float(m.get("predicted_iou", m.get("stability_score", 0.0))),
            area=int(m["area"]),
            bbox=_bbox_of(mask),
            label=f"object_{i + 1}",
            backend="sam2",
        ))
    return segments


# ---------------------------------------------------------------------------
# Tier 2: classical (Felzenszwalb graph segmentation + cleanup)
# ---------------------------------------------------------------------------

def _segment_classical(image_rgb: np.ndarray, max_objects: int,
                       min_area_frac: float = 0.005,
                       max_area_frac: float = 0.45) -> List[Segment]:
    """
    Published-algorithm pipeline, deterministic on CPU:
      1. Felzenszwalb graph-based segmentation on a mildly smoothed image --
         merges regions based on adaptive pairwise similarity, standard in the
         literature for class-agnostic region proposals.
      2. Merge segments that share a boundary and have similar mean color
         (single greedy pass, bounded iterations).
      3. Morphological open/close per mask to remove speckle/holes.
      4. Drop tiny regions (< min_area_frac of the image) and huge background
         blobs (> max_area_frac -- a region covering ~half the frame is the
         wall/floor, not an object), then keep the `max_objects` largest.
    """
    from skimage import segmentation as skseg, morphology as skmor
    from skimage.filters import gaussian as sk_gaussian

    h, w = image_rgb.shape[:2]
    img_f = sk_gaussian(image_rgb.astype(np.float32) / 255.0, sigma=1.0,
                        channel_axis=-1)

    # scale-aware params so behavior is stable across resolutions
    base = np.sqrt(h * w)
    felz_scale = float(max(30.0, base * 0.06))
    min_area = int(max(80, min_area_frac * h * w))
    max_area = int(max_area_frac * h * w)

    seg = skseg.felzenszwalb(img_f, scale=felz_scale, sigma=0.8,
                             min_size=max(40, min_area // 4))

    # --- greedy merge of similar adjacent regions --------------------------
    lab = seg.copy()
    n_labels = int(lab.max()) + 1
    means = np.zeros((n_labels, 3), dtype=np.float64)
    for l in range(n_labels):
        px = image_rgb[lab == l]
        if len(px):
            means[l] = px.mean(axis=0)

    # adjacent region pairs (horizontally and vertically touching pixels)
    neighbors = set()
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        da, db = a.ravel(), b.ravel()
        for x, y in zip(da[da != db], db[da != db]):
            neighbors.add((int(min(x, y)), int(max(x, y))))

    parent = list(range(n_labels))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b) in sorted(neighbors, key=lambda ab: float(np.linalg.norm(means[ab[0]] - means[ab[1]]))):
        ra, rb = find(a), find(b)
        if ra != rb and np.linalg.norm(means[ra] - means[rb]) < 40.0:
            parent[rb] = ra

    merged = np.zeros_like(lab)
    remap = {}
    for l in range(n_labels):
        r = find(l)
        if r not in remap:
            remap[r] = len(remap)
        merged[lab == l] = remap[r]
    n_merged = len(remap)

    # --- per-region cleanup + selection ------------------------------------
    candidate_masks = []
    for l in range(n_merged):
        raw = merged == l
        if raw.sum() < min_area:
            continue
        if raw.sum() > max_area:
            continue  # background-like blob (wall/floor), not an object
        m = skmor.opening(raw, skmor.disk(3))
        m = skmor.closing(m, skmor.disk(5))
        m = skmor.remove_small_holes(m, max_size=max(64, min_area // 8))
        if m.sum() >= min_area:
            candidate_masks.append(m)

    candidate_masks.sort(key=lambda m: int(m.sum()), reverse=True)
    segments = []
    for i, mask in enumerate(candidate_masks[:max_objects]):
        ys, xs = np.where(mask)
        score = 0.5 + 0.5 * (mask.sum() / candidate_masks[0].sum())
        segments.append(Segment(
            mask=mask,
            score=float(score),
            area=int(mask.sum()),
            bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            label=f"object_{i + 1}",
            backend="classical",
        ))
    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_image(image_rgb: np.ndarray,
                  backend: str = "auto",
                  max_objects: int = 10,
                  min_area_frac: float = 0.005,
                  max_area_frac: float = 0.45) -> List[Segment]:
    """
    Segment one RGB image (H, W, 3 uint8) into object regions.

    backend : "auto"  -> SAM2 when available, else classical
              "sam2"  -> require SAM2 (raises if unavailable)
              "classical" -> force the CPU fallback
    min_area_frac / max_area_frac : classical-backend size gates
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB image, got {image_rgb.shape}")
    if max_objects < 1:
        raise ValueError("max_objects must be >= 1")

    if backend == "auto":
        backend = "sam2" if sam2_available() else "classical"

    if backend == "sam2":
        if not sam2_available():
            raise RuntimeError(
                "SAM2 backend requested but not installed. Run "
                "scripts/setup_sam2.py, or use backend='classical'/'auto'."
            )
        return _segment_sam2(image_rgb, max_objects)
    if backend == "classical":
        return _segment_classical(image_rgb, max_objects,
                                  min_area_frac=min_area_frac,
                                  max_area_frac=max_area_frac)
    raise ValueError(f"unknown backend {backend!r} (use auto|sam2|classical)")
