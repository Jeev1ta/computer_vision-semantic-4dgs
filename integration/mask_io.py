"""
mask_io.py
----------
Loads the SAM2 masks written by sam2_runner.py back into memory in the
shape the association pipeline expects (guide section 16's
`load_masks()` stub):

    masks[image_name][object_id] -> (H, W) bool array

Also loads/saves the project-wide `object_metadata.json`
(guide section 18) that records label names + which source views were used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


def load_masks(mask_dir: str) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Scans <mask_dir>/masks/object_<ID>_frame_<NAME>.npy and
    <mask_dir>/metadata/object_<ID>.json (written by sam2_runner.save_mask,
    which stores a LIST of per-frame entries per object so the same object
    can contribute votes from several representative frames -- guide
    section 17's multi-view voting) and returns
    masks[image_name][object_id] -> bool mask.
    """
    mask_dir = Path(mask_dir)
    masks_subdir = mask_dir / "masks"
    meta_subdir = mask_dir / "metadata"

    if not masks_subdir.exists():
        raise FileNotFoundError(
            f"No masks/ directory under {mask_dir}. Run sam2_runner.py first "
            "(guide section 11-12)."
        )

    masks: Dict[str, Dict[int, np.ndarray]] = {}
    for meta_file in sorted(meta_subdir.glob("object_*.json")):
        with open(meta_file) as f:
            meta = json.load(f)
        object_id = meta["object_id"]
        for entry in meta.get("frames", []):
            source_frame = entry["source_frame"]
            mask_path = mask_dir / entry["mask_file"]
            mask = np.load(mask_path)
            masks.setdefault(source_frame, {})[object_id] = mask.astype(bool)

    return masks


def load_mask_scores(mask_dir: str) -> Dict[str, Dict[int, float]]:
    """masks[image_name][object_id] -> SAM2 confidence score, for weighted voting."""
    mask_dir = Path(mask_dir)
    meta_subdir = mask_dir / "metadata"
    scores: Dict[str, Dict[int, float]] = {}
    for meta_file in sorted(meta_subdir.glob("object_*.json")):
        with open(meta_file) as f:
            meta = json.load(f)
        object_id = meta["object_id"]
        for entry in meta.get("frames", []):
            scores.setdefault(entry["source_frame"], {})[object_id] = entry.get("score", 1.0)
    return scores


def load_object_metadata(mask_dir: str) -> Dict[int, dict]:
    """Object id -> {'label': ..., 'frames': [...]} across all objects."""
    mask_dir = Path(mask_dir)
    meta_subdir = mask_dir / "metadata"
    result = {}
    for meta_file in sorted(meta_subdir.glob("object_*.json")):
        with open(meta_file) as f:
            meta = json.load(f)
        result[meta["object_id"]] = meta
    return result


def write_semantic_object_metadata(
    out_path: str,
    label_names: Dict[int, str],
    source_views: Dict[int, list],
    unknown_label: int = -1,
    extra: dict = None,
):
    """
    Writes the final outputs/semantic/<scene>/object_metadata.json (guide section 18).
    Canonical schema:
        {"objects": [{"id": int, "label": str, "source_views": [str]}, ...],
         "unknown_label": int}
    `extra` (optional) merges additional provenance fields into the top level
    (e.g. mode / depth_source / segmentation_backend for single-image runs).
    """
    payload = {
        "objects": [
            {"id": obj_id, "label": label_names[obj_id], "source_views": source_views.get(obj_id, [])}
            for obj_id in sorted(label_names)
        ],
        "unknown_label": unknown_label,
    }
    if extra:
        payload.update(extra)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
