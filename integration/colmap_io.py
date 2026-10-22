"""
colmap_io.py
------------
Loads COLMAP camera intrinsics/extrinsics produced by the official 3DGS
`convert.py` preprocessing step (guide section 7 / 8).

We deliberately DO NOT reimplement the binary parsing ourselves. Instead we
import the exact reader functions shipped in
`third_party/gaussian-splatting/scene/colmap_loader.py`
(`read_cameras_binary`, `read_images_binary`, `qvec2rotmat`) -- this is
the same code path the official training script uses, so our camera math
stays consistent with the reconstructed scene. That module only depends on
numpy, so it can be imported into the lightweight "integration" environment
without needing the full CUDA/torch 3DGS install (guide section 5).

COLMAP convention (see colmap_loader.py):
    X_cam = R @ X_world + t
where R = qvec2rotmat(qvec), t = tvec. This is a WORLD-TO-CAMERA transform.

Note on lens distortion: the official `convert.py` undistorts all images and
writes SIMPLE_RADIAL/OPENCV parameters accordingly. Because both 3DGS
training and this pipeline operate on the undistorted images, we use only
the focal lengths and principal point and ignore the distortion coefficients
(mirroring what the official pipeline assumes downstream).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

# --- locate + import the official (pure-numpy) COLMAP reader --------------
_DEFAULT_3DGS_PATH = Path(__file__).resolve().parent.parent / "third_party" / "gaussian-splatting"


def _import_official_colmap_loader(gaussian_splatting_repo: Path = _DEFAULT_3DGS_PATH):
    repo = Path(gaussian_splatting_repo)
    loader_file = repo / "scene" / "colmap_loader.py"
    if not loader_file.exists():
        raise FileNotFoundError(
            f"Could not find the official COLMAP loader at {loader_file}. "
            "Clone the official repo first:\n"
            "  git clone https://github.com/graphdeco-inria/gaussian-splatting.git "
            "--recursive third_party/gaussian-splatting"
        )
    scene_dir = str(repo / "scene")
    if scene_dir not in sys.path:
        sys.path.insert(0, scene_dir)
    import colmap_loader  # noqa: E402  (from third_party/gaussian-splatting/scene)
    return colmap_loader


@dataclass
class Camera:
    """A single calibrated camera, ready for projecting 3D points."""
    camera_id: int
    image_id: int
    name: str                # e.g. "00001.jpg" -- matches images/ and masks/
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray            # (3, 3) world -> camera rotation
    t: np.ndarray            # (3,)   world -> camera translation

    @property
    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0,      self.cx],
            [0.0,     self.fy,  self.cy],
            [0.0,     0.0,      1.0],
        ], dtype=np.float64)

    def world_to_camera(self, xyz_world: np.ndarray) -> np.ndarray:
        """xyz_world: (N, 3) -> (N, 3) in camera coordinates."""
        return xyz_world @ self.R.T + self.t

    def camera_center(self) -> np.ndarray:
        """Camera center in world coordinates: C = -R^T t."""
        return -self.R.T @ self.t


def _intrinsics_to_fx_fy_cx_cy(model: str, params: np.ndarray):
    """
    Convert a COLMAP camera model's raw parameter vector into fx, fy, cx, cy.
    Distortion coefficients (k1, k2, p1, p2, ...) are intentionally dropped --
    convert.py undistorts the images before training, so the remaining
    pinhole parameters describe the images on disk.
    """
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        f, cx, cy = params[0], params[1], params[2]
        return float(f), float(f), float(cx), float(cy)
    if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV",
                 "THIN_PRISM_FISHEYE", "RADIAL", "RADIAL_FISHEYE"):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return float(fx), float(fy), float(cx), float(cy)
    if model == "FOV":
        f, cx, cy = params[0], params[1], params[2]
        return float(f), float(f), float(cx), float(cy)
    raise ValueError(
        f"Unsupported COLMAP camera model '{model}'. "
        "Add its fx/fy/cx/cy extraction rule here (see colmap_loader.CAMERA_MODELS)."
    )


def _find_sparse_dir(dataset_path: Path, sparse_subdir: str = "") -> Path:
    """Locate the COLMAP sparse directory: explicit override, then sparse/0,
    then sparse/, then any numeric subdirectory of sparse/."""
    candidates = []
    if sparse_subdir:
        candidates.append(dataset_path / sparse_subdir)
    sparse_root = dataset_path / "sparse"
    candidates.append(sparse_root / "0")
    candidates.append(sparse_root)
    if sparse_root.exists():
        for child in sorted(sparse_root.iterdir()):
            if child.is_dir() and child.name.isdigit():
                candidates.append(child)
    for cand in candidates:
        if (cand / "cameras.bin").exists() or (cand / "cameras.txt").exists():
            return cand
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"No COLMAP model (cameras.bin/.txt + images.bin/.txt) found in: {tried}. "
        "Run `python convert.py -s <dataset> --resize` first (guide section 7.3)."
    )


def load_cameras(
    dataset_path: str,
    sparse_subdir: str = "",
    gaussian_splatting_repo: str = str(_DEFAULT_3DGS_PATH),
) -> Dict[str, Camera]:
    """
    Load every registered camera/image pair from a COLMAP `sparse/` folder
    (guide section 8/9). Returns a dict keyed by image filename, e.g.
    cameras["00001.jpg"] -> Camera(...), so it lines up 1:1 with SAM2 masks
    generated from the same image.

    Binary models are preferred; text models are read as a fallback.
    """
    colmap_loader = _import_official_colmap_loader(gaussian_splatting_repo)

    sparse_dir = _find_sparse_dir(Path(dataset_path), sparse_subdir)

    if (sparse_dir / "cameras.bin").exists():
        cams_raw = colmap_loader.read_cameras_binary(str(sparse_dir / "cameras.bin"))
        imgs_raw = colmap_loader.read_images_binary(str(sparse_dir / "images.bin"))
    else:
        cams_raw = colmap_loader.read_cameras_text(str(sparse_dir / "cameras.txt"))
        imgs_raw = colmap_loader.read_images_text(str(sparse_dir / "images.txt"))

    if not imgs_raw:
        raise ValueError(
            f"COLMAP model at {sparse_dir} contains zero registered images. "
            "Re-run COLMAP and make sure the reconstruction succeeded."
        )

    cameras: Dict[str, Camera] = {}
    for image_id, img in imgs_raw.items():
        cam_raw = cams_raw[img.camera_id]
        fx, fy, cx, cy = _intrinsics_to_fx_fy_cx_cy(cam_raw.model, cam_raw.params)
        R = colmap_loader.qvec2rotmat(img.qvec)   # world -> camera rotation
        t = np.array(img.tvec, dtype=np.float64)  # world -> camera translation

        cameras[img.name] = Camera(
            camera_id=int(img.camera_id),
            image_id=int(image_id),
            name=img.name,
            width=int(cam_raw.width),
            height=int(cam_raw.height),
            fx=fx, fy=fy, cx=cx, cy=cy,
            R=R, t=t,
        )
    return cameras


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Sanity-check COLMAP camera loading.")
    p.add_argument("--dataset", required=True, help="Path to data/<scene> (post convert.py)")
    args = p.parse_args()
    cams = load_cameras(args.dataset)
    print(f"Loaded {len(cams)} cameras.")
    for name, c in list(cams.items())[:3]:
        print(f"  {name}: {c.width}x{c.height}, fx={c.fx:.1f}, fy={c.fy:.1f}, "
              f"cx={c.cx:.1f}, cy={c.cy:.1f}")
