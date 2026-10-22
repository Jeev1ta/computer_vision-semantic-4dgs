"""
gaussian_io.py
--------------
Reads a trained 3DGS model's `point_cloud.ply` (guide section 15/16) without
needing the full CUDA/torch training environment.

The official repo's `GaussianModel.load_ply` (scene/gaussian_model.py) reads
exactly these same PLY fields but immediately wraps them as CUDA
`nn.Parameter`s, which forces a GPU + the full training env just to read
positions. Since the integration layer only needs the *geometry*
(xyz / opacity / scale / rotation) and not gradients, we mirror the same
field-parsing logic here with plain numpy + plyfile, matching the field
names verified directly from the official source:

    x, y, z                      -> xyz center
    opacity                      -> raw opacity (apply sigmoid to get [0,1])
    scale_0, scale_1, scale_2    -> raw log-scale (apply exp for real scale)
    rot_0..rot_3                 -> raw quaternion (normalize for real rotation)
    f_dc_0/1/2 (+ f_rest_*)      -> spherical-harmonics color (not needed for
                                     semantic labeling, loaded optionally)

This keeps the official CUDA install completely untouched (guide section 16
architectural note): if this reader has a bug, `train.py` / the SIBR viewer
still work fine because we never modify that repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .ply_compat import PlyData


@dataclass
class GaussianCloud:
    xyz: np.ndarray                 # (N, 3) float64 -- Gaussian centers, world space
    opacity: np.ndarray             # (N,)   float32 in [0, 1] (sigmoid applied)
    scale: Optional[np.ndarray]     # (N, 3) float32 (exp applied), or None
    rotation: Optional[np.ndarray]  # (N, 4) float32 unit quaternion, or None
    n: int

    def filter_by_opacity(self, min_opacity: float = 0.05) -> np.ndarray:
        """Boolean mask of Gaussians worth projecting (guide 17: 'reject tiny/uncertain')."""
        return self.opacity >= min_opacity


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_gaussian_cloud(ply_path: str, load_scale_rotation: bool = True) -> GaussianCloud:
    """
    Load a trained model's `point_cloud/iteration_<N>/point_cloud.ply`.

    Typical path (matches Scene.save() in the official repo, scene/__init__.py):
        outputs/3dgs/<scene>/point_cloud/iteration_30000/point_cloud.ply
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(
            f"{ply_path} not found. Train the baseline first (guide section 9):\n"
            "  python train.py -s <dataset> -m <output_dir>"
        )

    plydata = PlyData.read(str(ply_path))
    v = plydata.elements[0]
    names = {p.name for p in v.properties}

    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float64)

    if "opacity" in names:
        opacity = _sigmoid(np.asarray(v["opacity"]).astype(np.float32))
    else:
        opacity = np.ones(len(xyz), dtype=np.float32)

    scale = None
    rotation = None
    if load_scale_rotation:
        scale_names = sorted([n for n in names if n.startswith("scale_")],
                              key=lambda n: int(n.split("_")[-1]))
        if scale_names:
            scale = np.stack([np.asarray(v[n]) for n in scale_names], axis=1).astype(np.float32)
            scale = np.exp(scale)  # 3DGS stores log-scale

        rot_names = sorted([n for n in names if n.startswith("rot_")],
                            key=lambda n: int(n.split("_")[-1]))
        if rot_names:
            rotation = np.stack([np.asarray(v[n]) for n in rot_names], axis=1).astype(np.float32)
            norm = np.linalg.norm(rotation, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            rotation = rotation / norm

    return GaussianCloud(xyz=xyz, opacity=opacity, scale=scale, rotation=rotation, n=len(xyz))


def find_latest_ply(model_path: str) -> str:
    """Convenience: find the highest-iteration point_cloud.ply under a model dir."""
    pc_dir = Path(model_path) / "point_cloud"
    if not pc_dir.exists():
        raise FileNotFoundError(f"No point_cloud/ directory under {model_path}")
    iters = []
    for d in pc_dir.iterdir():
        if d.is_dir() and d.name.startswith("iteration_"):
            try:
                iters.append(int(d.name.split("_")[-1]))
            except ValueError:
                continue
    if not iters:
        raise FileNotFoundError(f"No iteration_* checkpoints under {pc_dir}")
    latest = max(iters)
    return str(pc_dir / f"iteration_{latest}" / "point_cloud.ply")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Sanity-check trained Gaussian PLY loading.")
    p.add_argument("--model_path", required=True, help="e.g. outputs/3dgs/my_scene")
    args = p.parse_args()
    ply = find_latest_ply(args.model_path)
    cloud = load_gaussian_cloud(ply)
    print(f"Loaded {cloud.n} Gaussians from {ply}")
    print(f"  xyz range: {cloud.xyz.min(0)} .. {cloud.xyz.max(0)}")
    print(f"  opacity mean: {cloud.opacity.mean():.3f}")
