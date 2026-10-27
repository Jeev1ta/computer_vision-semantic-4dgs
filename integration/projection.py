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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized projection of ALL Gaussians into ONE camera.

    Returns
    -------
    uv : (N, 2) float array of pixel coordinates (undefined where not visible)
    visible : (N,) bool mask -- True if the point is in front of the camera
              AND lands inside the image bounds. Only trust `uv` where
              `visible` is True (guide section 14's "important implementation
              warning": always gate on this before doing a mask lookup).
    Zc : (N,) float array of camera-space depths. Use with
         ``compute_depth_buffer`` and ``filter_occluded`` to gate out
         points hidden behind closer surfaces.
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
    return uv, visible, Zc


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


# ---------------------------------------------------------------------------
# Depth-buffer-based occlusion culling  (Phase 4 addition)
# ---------------------------------------------------------------------------

def compute_depth_buffer(
    uv: np.ndarray,
    Zc: np.ndarray,
    visible: np.ndarray,
    width: int,
    height: int,
    dilate_px: int = 1,
) -> np.ndarray:
    """
    Build a discrete (H, W) depth buffer from projected Gaussian centers.

    For every pixel that at least one visible Gaussian projects to, the buffer
    stores the **minimum** camera-space depth.  A small min-filter dilation
    (default 1 px) fills single-pixel gaps between sparse point projections so
    that nearby occluded points are still culled.

    Parameters
    ----------
    uv       : (N, 2) projected pixel coordinates
    Zc       : (N,) camera-space depths
    visible  : (N,) bool, frustum-visible mask from ``project_points_batch``
    width, height : camera resolution
    dilate_px : radius of the min-filter dilation applied after rasterisation

    Returns
    -------
    depth_buffer : (H, W) float64 array, ``np.inf`` where no point landed.
    """
    depth_flat = np.full(height * width, np.inf, dtype=np.float64)

    idx = np.where(visible)[0]
    if len(idx) == 0:
        return depth_flat.reshape(height, width)

    xs = np.round(uv[idx, 0]).astype(np.int64)
    ys = np.round(uv[idx, 1]).astype(np.int64)
    zs = Zc[idx].astype(np.float64)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    linear = ys[valid] * width + xs[valid]
    np.minimum.at(depth_flat, linear, zs[valid])

    depth_buffer = depth_flat.reshape(height, width)

    # Min-filter dilation: propagate nearest-surface depth into neighboring
    # pixels so that sparse point clouds still occlude correctly.
    for _ in range(dilate_px):
        padded = np.full((height + 2, width + 2), np.inf, dtype=np.float64)
        padded[1:-1, 1:-1] = depth_buffer
        depth_buffer = np.minimum(depth_buffer, padded[:-2, 1:-1])   # up
        depth_buffer = np.minimum(depth_buffer, padded[2:,  1:-1])   # down
        depth_buffer = np.minimum(depth_buffer, padded[1:-1, :-2])   # left
        depth_buffer = np.minimum(depth_buffer, padded[1:-1, 2:])    # right

    return depth_buffer


def filter_occluded(
    uv: np.ndarray,
    Zc: np.ndarray,
    visible: np.ndarray,
    depth_buffer: np.ndarray,
    rel_tol: float = 0.05,
    abs_tol: float = 0.01,
) -> np.ndarray:
    """
    Remove occluded points from the ``visible`` mask using a depth-buffer test.

    A visible point is reclassified as **occluded** when its camera-space
    depth exceeds the nearest surface at its projected pixel by more than a
    combined relative + absolute tolerance:

        occluded  ⟺  Zc  >  depth_buffer[v, u] + rel_tol·Zc + abs_tol

    Parameters
    ----------
    uv           : (N, 2) projected pixel coordinates
    Zc           : (N,) camera-space depths
    visible      : (N,) bool, frustum-visible mask (will NOT be mutated)
    depth_buffer : (H, W) from ``compute_depth_buffer``
    rel_tol      : relative depth tolerance (fraction of point depth)
    abs_tol      : absolute depth tolerance (in world units)

    Returns
    -------
    visible_unoccluded : (N,) bool — copy of ``visible`` with occluded
                         entries set to False.
    """
    height, width = depth_buffer.shape
    result = visible.copy()

    idx = np.where(visible)[0]
    if len(idx) == 0:
        return result

    xs = np.round(uv[idx, 0]).astype(np.int64)
    ys = np.round(uv[idx, 1]).astype(np.int64)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    surface_depth = np.full(len(idx), np.inf)
    surface_depth[valid] = depth_buffer[ys[valid], xs[valid]]

    point_depth = Zc[idx].astype(np.float64)
    tolerance = rel_tol * point_depth + abs_tol
    is_occluded = point_depth > (surface_depth + tolerance)

    result[idx[is_occluded]] = False
    return result
