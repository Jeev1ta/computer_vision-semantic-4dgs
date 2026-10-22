"""
sam2_runner.py
--------------
Thin wrapper around the OFFICIAL SAM2ImagePredictor (facebookresearch/sam2)
to implement the workflow described in guide section 11-12:

    representative image -> user click/box prompt -> SAM2 -> binary mask
        -> masks/object_XX_frame_YYYYY.{npy,png} + metadata/object_XX.json

IMPORTANT (guide section 5): this file must be run inside the *separate*
`sam2` conda environment (Python >= 3.10, torch >= 2.5.1), NOT inside the
3DGS training environment. It imports the official `sam2` package directly
-- we do not reimplement any segmentation logic ourselves, only the
glue/bookkeeping around it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


_DEFAULT_SAM2_REPO = Path(__file__).resolve().parent.parent / "third_party" / "sam2"


def _import_official_sam2(sam2_repo: Path = _DEFAULT_SAM2_REPO):
    repo = Path(sam2_repo)
    if not repo.exists():
        raise FileNotFoundError(
            f"Could not find the official sam2 repo at {repo}. Clone it first "
            "(guide section 6.3): git clone https://github.com/facebookresearch/sam2.git"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from sam2.build_sam import build_sam2            # noqa: E402
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
    return build_sam2, SAM2ImagePredictor


@dataclass
class ObjectPrompt:
    """One user-supplied prompt for one object on one representative frame."""
    object_id: int
    label: str
    source_frame: str          # image filename, e.g. "00020.jpg"
    points: Optional[List[Tuple[int, int]]] = None   # [(x, y), ...]
    point_labels: Optional[List[int]] = None          # 1 = foreground, 0 = background
    box: Optional[Tuple[int, int, int, int]] = None   # (x0, y0, x1, y1)


class SAM2MaskGenerator:
    """
    Loads a SAM2 checkpoint once and generates masks for a list of
    ObjectPrompts, following the exact `set_image()` / `predict()` pattern
    documented in the official README (guide section 11.3).
    """

    def __init__(
        self,
        checkpoint: str = "third_party/sam2/checkpoints/sam2.1_hiera_small.pt",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        device: str = "cuda",
        sam2_repo: str = str(_DEFAULT_SAM2_REPO),
    ):
        build_sam2, SAM2ImagePredictor = _import_official_sam2(Path(sam2_repo))
        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        self._current_image_path: Optional[str] = None

    def _ensure_image(self, image_path: str):
        if self._current_image_path == image_path:
            return
        if Image is None:
            raise RuntimeError("Pillow is required to load images for SAM2.")
        image = np.array(Image.open(image_path).convert("RGB"))
        self.predictor.set_image(image)
        self._current_image_path = image_path

    def predict(self, image_path: str, prompt: ObjectPrompt):
        """Returns (best_mask: HxW bool array, score: float)."""
        self._ensure_image(image_path)

        point_coords = np.array(prompt.points) if prompt.points else None
        point_labels = np.array(prompt.point_labels) if prompt.point_labels else None
        box = np.array(prompt.box) if prompt.box else None

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool), float(scores[best_idx])


def save_mask(
    mask: np.ndarray,
    score: float,
    prompt: ObjectPrompt,
    out_dir: str,
) -> dict:
    """
    Persists a mask exactly as laid out in guide section 12:
        masks/object_01_frame_00020.npy
        masks/object_01_frame_00020.png
        metadata/object_01.json

    NOTE: multi-view voting (guide section 17) needs the SAME object
    prompted/segmented on SEVERAL representative frames, not just one. So
    `metadata/object_<ID>.json` stores a LIST of per-frame entries and this
    function APPENDS to it rather than overwriting -- call it once per
    (object, frame) you want to contribute a vote.

    Returns the single entry that was just written (also appended into the
    object's metadata list).
    """
    out_dir = Path(out_dir)
    masks_dir = out_dir / "masks"
    meta_dir = out_dir / "metadata"
    masks_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    frame_stem = Path(prompt.source_frame).stem
    base = f"object_{prompt.object_id:02d}_frame_{frame_stem}"

    np.save(masks_dir / f"{base}.npy", mask)
    if Image is not None:
        Image.fromarray((mask * 255).astype(np.uint8)).save(masks_dir / f"{base}.png")

    entry = {
        "object_id": prompt.object_id,
        "label": prompt.label,
        "source_frame": prompt.source_frame,
        "mask_file": str((masks_dir / f"{base}.npy").relative_to(out_dir)),
        "score": score,
    }

    meta_path = meta_dir / f"object_{prompt.object_id:02d}.json"
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
        frames = existing.get("frames", [])
    else:
        frames = []

    frames = [fr for fr in frames if fr["source_frame"] != prompt.source_frame]  # replace if re-run
    frames.append(entry)

    with open(meta_path, "w") as f:
        json.dump({"object_id": prompt.object_id, "label": prompt.label, "frames": frames}, f, indent=2)

    return entry


def run_prompts(
    prompts: Sequence[ObjectPrompt],
    images_dir: str,
    out_dir: str,
    checkpoint: str = "third_party/sam2/checkpoints/sam2.1_hiera_small.pt",
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
    device: str = "cuda",
) -> List[dict]:
    """
    Batch entry point: run every ObjectPrompt through SAM2 and persist masks.
    Mirrors guide section 25 week-5 milestone ("Install/test SAM2 -> Object masks").
    """
    generator = SAM2MaskGenerator(checkpoint=checkpoint, model_cfg=model_cfg, device=device)
    results = []
    for prompt in prompts:
        image_path = str(Path(images_dir) / prompt.source_frame)
        mask, score = generator.predict(image_path, prompt)
        meta = save_mask(mask, score, prompt, out_dir)
        results.append(meta)
        print(f"[sam2] object {prompt.object_id} ('{prompt.label}') "
              f"on {prompt.source_frame}: score={score:.3f}")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate a SAM2 mask from a single click prompt.")
    p.add_argument("--image", required=True)
    p.add_argument("--x", type=int, required=True)
    p.add_argument("--y", type=int, required=True)
    p.add_argument("--object_id", type=int, default=1)
    p.add_argument("--label", default="object")
    p.add_argument("--out_dir", default="outputs/masks/my_scene")
    p.add_argument("--checkpoint", default="third_party/sam2/checkpoints/sam2.1_hiera_small.pt")
    p.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    prompt = ObjectPrompt(
        object_id=args.object_id,
        label=args.label,
        source_frame=Path(args.image).name,
        points=[(args.x, args.y)],
        point_labels=[1],
    )
    run_prompts(
        [prompt],
        images_dir=str(Path(args.image).parent),
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
    )
