"""
projection.py
--------------
The single most important piece of custom math in the whole project
(guide section 13-14 and 28): projecting a 3D Gaussian center into a 2D
camera image so we can look up its SAM2 mask label.

Convention (matches colmap_io.Camera / COLMAP):
    X_cam = R @ X_world + t          (world -> camera, R,t from COLMAP)
    u = fx * Xc / Zc + cx
    v = fy * Yc / Zc + cy

Two APIs are provided:
  - project_point(...)        scalar version, used by tests / sanity checks
                               (mirrors guide section 28's tiny reference fn)
  - project_points_batch(...) vectorized version used by the real pipeline,
                               operates on all N Gaussians against one
                               camera in a single numpy call.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def project_point(X, fx, fy, cx, cy) -> np.ndarray:
    """
    Scalar pinhole projection of a single point already in CAMERA coordinates.
    This is the exact function sketched in guide section 28 -- kept as a
    thin, easily-testable reference implementation.
    """
    Xc, Yc, Zc = X
    if Zc <= 0:
        raise ValueError("Point is behind the camera (Zc <= 0).")
    u = fx * Xc / Zc + cx
    v = fy * Yc / Zc + cy
    return np.array([u, v])


def world_to_camera(xyz_world: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(N,3) world points -> (N,3) camera-space points."""
    return xyz_world @ R.T + t


def project_points_batch(
    xyz_world: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized projection of ALL Gaussians into ONE camera.

    Returns
    -------
    uv : (N, 2) float array of pixel coordinates (undefined where not visible)
    visible : (N,) bool mask -- True if the point is in front of the camera
              AND lands inside the image bounds. Only trust `uv` where
              `visible` is True (guide section 14's "important implementation
              warning": always gate on this before doing a mask lookup).
    """
    xyz_cam = world_to_camera(xyz_world, R, t)
    Zc = xyz_cam[:, 2]

    in_front = Zc > 1e-6
    # avoid divide-by-zero warnings for points behind the camera
    safe_Zc = np.where(in_front, Zc, 1.0)

    u = fx * xyz_cam[:, 0] / safe_Zc + cx
    v = fy * xyz_cam[:, 1] / safe_Zc + cy

    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    visible = in_front & in_bounds

    uv = np.stack([u, v], axis=1)
    return uv, visible


def inside_mask(mask: np.ndarray, u: float, v: float) -> bool:
    """
    Scalar mask lookup at a projected pixel (guide section 28).
    `mask` is an (H, W) boolean array from SAM2.
    """
    x = int(round(u))
    y = int(round(v))
    if y < 0 or y >= mask.shape[0]:
        return False
    if x < 0 or x >= mask.shape[1]:
        return False
    return bool(mask[y, x])


def sample_mask_batch(mask: np.ndarray, uv: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """
    Vectorized mask lookup for many projected points at once.

    mask     : (H, W) bool array
    uv       : (N, 2) float pixel coords (from project_points_batch)
    visible  : (N,) bool, points to actually sample (others forced False)

    Returns (N,) bool array: True where the (rounded) pixel falls inside
    the mask's True region.
    """
    h, w = mask.shape[:2]
    result = np.zeros(len(uv), dtype=bool)

    xs = np.round(uv[:, 0]).astype(np.int64)
    ys = np.round(uv[:, 1]).astype(np.int64)

    valid = visible & (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    result[valid] = mask[ys[valid], xs[valid]]
    return result
