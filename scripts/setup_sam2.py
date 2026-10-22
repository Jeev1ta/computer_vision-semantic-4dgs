"""
setup_sam2.py
-------------
Optional one-command upgrade to REAL SAM2 segmentation:

  1. clones facebookresearch/sam2 into third_party/sam2 (if missing)
  2. downloads the official sam2.1_hiera_tiny checkpoint (~40 MB) into
     third_party/sam2/checkpoints/ (if missing)
  3. verifies `import sam2` works and prints the verdict

The pipeline auto-detects SAM2 afterwards (segmentation.backend: auto).
Torch must already be importable; on CPU-only machines SAM2 runs slowly but
correctly.

Usage:  python scripts/setup_sam2.py [--model small]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAM2_REPO = ROOT / "third_party" / "sam2"

CHECKPOINTS = {
    "tiny": ("sam2.1_hiera_tiny.pt",
             "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
             "sam2.1_hiera_tiny.pt"),
    "small": ("sam2.1_hiera_small.pt",
              "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
              "sam2.1_hiera_small.pt"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(CHECKPOINTS), default="tiny",
                        help="checkpoint size to download (default: tiny, ~40 MB)")
    args = parser.parse_args()

    # 0. torch present?
    try:
        import torch  # noqa: F401
        print(f"[setup] torch {torch.__version__} ok (cuda={torch.cuda.is_available()})")
    except Exception as err:
        print(f"[setup] torch is not importable ({err}). Install torch first, "
              "e.g. pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return 1

    # 1. clone the official repo (argument-list subprocess, shell=False:
    #    no shell is involved and every argument is a script-controlled literal)
    if (SAM2_REPO / "sam2").is_dir():
        print(f"[setup] SAM2 repo already present at {SAM2_REPO}")
    else:
        SAM2_REPO.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/sam2.git", str(SAM2_REPO)],
            check=True, shell=False,
        )

    # 2. download the checkpoint from the official Meta URL
    ckpt_name, url = CHECKPOINTS[args.model]
    ckpt_dir = SAM2_REPO / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / ckpt_name
    if ckpt_path.exists():
        print(f"[setup] checkpoint already present: {ckpt_path}")
    else:
        print(f"[setup] downloading {url} ...")
        urllib.request.urlretrieve(url, ckpt_path)
        print(f"[setup] saved {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.0f} MB)")

    # 3. verify import + availability
    sys.path.insert(0, str(SAM2_REPO))
    try:
        from sam2.build_sam import build_sam2  # noqa: F401
        print("[setup] `import sam2` OK")
    except Exception as err:
        print(f"[setup] WARNING: sam2 imports failed ({err}).\n"
              f"        Try: pip install -e {SAM2_REPO}")
        return 1

    from integration.segmentation import available_backends
    print(f"[setup] backends now: {available_backends()}")
    print("[setup] done -- segmentation.backend: auto will now use SAM2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
