"""
scripts/generate_masks_from_config.py
--------------------------------------
Reads the `objects:` list in configs/config.yaml and runs the official
SAM2ImagePredictor on each entry, saving masks in the layout mask_io.py
expects. This is the concrete script behind guide section 25's Week 5
milestone ("Install/test SAM2 -> Object masks").

IMPORTANT: run this INSIDE the sam2 conda environment (guide section 5),
not the integration environment -- it needs torch + the CUDA SAM2 extension.

Usage:
    conda activate sam2
    python scripts/generate_masks_from_config.py --config configs/config.yaml
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from integration.sam2_runner import ObjectPrompt, run_prompts  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    sam2_cfg = cfg["sam2"]
    images_dir = str(Path(paths["dataset_dir"]) / "images")

    prompts = []
    for obj in cfg.get("objects", []):
        prompts.append(ObjectPrompt(
            object_id=obj["object_id"],
            label=obj["label"],
            source_frame=obj["source_frame"],
            points=[tuple(pt) for pt in obj.get("points", [])] or None,
            point_labels=obj.get("point_labels"),
            box=tuple(obj["box"]) if obj.get("box") else None,
        ))

    if not prompts:
        print("No `objects:` entries found in config. Edit configs/config.yaml first "
              "(guide section 12: pick a representative frame + click point per object).")
        return

    run_prompts(
        prompts,
        images_dir=images_dir,
        out_dir=paths["mask_dir"],
        checkpoint=sam2_cfg["checkpoint"],
        model_cfg=sam2_cfg["model_cfg"],
        device=sam2_cfg.get("device", "cuda"),
    )
    print(f"\nDone. {len(prompts)} masks written to {paths['mask_dir']}")


if __name__ == "__main__":
    main()
