"""
test_occlusion.py
-----------------
Unit tests for depth-based occlusion culling in the static integration pipeline.
Tests:
  1. Two points along identical ray (Z=2 vs Z=10): front point is visible, back point is culled.
  2. Side-by-side points at different image coordinates: both remain visible.
  3. Depth tolerance margin: points within tolerance stay visible, beyond tolerance get culled.
  4. Min-filter dilation: neighboring pixels receive nearest-surface depth.
  5. End-to-end build_vote_matrix: occluded background Gaussian does not receive foreground votes.
"""

import numpy as np
import pytest

from integration.projection import (
    project_points_batch,
    compute_depth_buffer,
    filter_occluded,
    sample_mask_batch,
)
from integration.association import build_vote_matrix, UNKNOWN_LABEL
from integration.colmap_io import Camera
from integration.gaussian_io import GaussianCloud


def test_two_points_same_ray_occlusion():
    """
    Two points along line of sight (X=0, Y=0):
    Point 0 at Z=2.0 (foreground)
    Point 1 at Z=10.0 (background)
    Both project to pixel (320, 240).
    Depth buffer should record Z=2.0 at (240, 320).
    filter_occluded must keep Point 0 visible and mark Point 1 occluded.
    """
    xyz = np.array([
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 10.0],
    ])
    R = np.eye(3)
    t = np.zeros(3)
    fx = fy = 500.0
    cx, cy = 320.0, 240.0
    width, height = 640, 480

    uv, visible, Zc = project_points_batch(xyz, R, t, fx, fy, cx, cy, width, height)
    assert visible.all(), "Both points are initially in frustum"

    depth_buf = compute_depth_buffer(uv, Zc, visible, width, height, dilate_px=0)
    assert np.isclose(depth_buf[240, 320], 2.0)

    unoccluded = filter_occluded(uv, Zc, visible, depth_buf, rel_tol=0.05, abs_tol=0.01)
    assert unoccluded[0] is True or unoccluded[0] == 1, "Foreground point must remain visible"
    assert unoccluded[1] is False or unoccluded[1] == 0, "Background point must be culled by occlusion"


def test_side_by_side_different_coords_both_visible():
    """
    Two points at different (X, Y) locations:
    Point 0 at X=-1, Z=2
    Point 1 at X=+1, Z=10
    Because they project to different pixels, neither occludes the other.
    """
    xyz = np.array([
        [-1.0, 0.0, 2.0],
        [1.0, 0.0, 10.0],
    ])
    R = np.eye(3)
    t = np.zeros(3)
    fx = fy = 500.0
    cx, cy = 320.0, 240.0
    width, height = 640, 480

    uv, visible, Zc = project_points_batch(xyz, R, t, fx, fy, cx, cy, width, height)
    depth_buf = compute_depth_buffer(uv, Zc, visible, width, height, dilate_px=0)

    unoccluded = filter_occluded(uv, Zc, visible, depth_buf)
    assert unoccluded.tolist() == [True, True]


def test_depth_tolerance_boundary():
    """
    Surface is at Z=4.0.
    rel_tol=0.10, abs_tol=0.05 -> threshold for Z=4.2: 4.0 + 0.10*4.2 + 0.05 = 4.47 > 4.2 (visible)
    For Z=5.0: threshold 4.0 + 0.10*5.0 + 0.05 = 4.55 < 5.0 (occluded)
    """
    xyz = np.array([
        [0.0, 0.0, 4.0],
        [0.0, 0.0, 4.2],  # within tolerance
        [0.0, 0.0, 5.0],  # exceeds tolerance
    ])
    R = np.eye(3)
    t = np.zeros(3)
    width, height = 640, 480

    uv, visible, Zc = project_points_batch(xyz, R, t, 500.0, 500.0, 320.0, 240.0, width, height)
    depth_buf = compute_depth_buffer(uv, Zc, visible, width, height, dilate_px=0)

    unoccluded = filter_occluded(uv, Zc, visible, depth_buf, rel_tol=0.10, abs_tol=0.05)
    assert unoccluded[0] is True or unoccluded[0] == 1
    assert unoccluded[1] is True or unoccluded[1] == 1
    assert unoccluded[2] is False or unoccluded[2] == 0


def test_mask_sampling_with_occlusion_culling():
    """
    Verify that mask sampling respects occlusion:
    A foreground point (Z=2) and background point (Z=8) both project onto
    a foreground mask. With occlusion culling, the background point does NOT hit.
    """
    mask = np.zeros((480, 640), dtype=bool)
    mask[230:250, 310:330] = True  # mask covers (320, 240)

    xyz = np.array([
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 8.0],
    ])
    R = np.eye(3)
    t = np.zeros(3)

    uv, visible, Zc = project_points_batch(xyz, R, t, 500.0, 500.0, 320.0, 240.0, 640, 480)

    # Without occlusion culling: both hit
    naive_hit = sample_mask_batch(mask, uv, visible)
    assert naive_hit.tolist() == [True, True]

    # With occlusion culling: only point 0 hits
    depth_buf = compute_depth_buffer(uv, Zc, visible, 640, 480, dilate_px=1)
    unoccluded = filter_occluded(uv, Zc, visible, depth_buf)
    culled_hit = sample_mask_batch(mask, uv, unoccluded)
    assert culled_hit.tolist() == [True, False]


def test_build_vote_matrix_end_to_end_occlusion():
    """
    Test build_vote_matrix with and without enable_occlusion_culling:
    Point 0: foreground object at (0, 0, 2)
    Point 1: occluded background point at (0, 0, 8)
    Camera 1 has a mask at (320, 240).
    With occlusion culling enabled, Point 1 must receive UNKNOWN_LABEL.
    With occlusion culling disabled, Point 1 falsely receives object_id 1.
    """
    cloud = GaussianCloud(
        xyz=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 8.0]], dtype=np.float32),
        opacity=np.array([0.9, 0.9], dtype=np.float32),
        scale=None,
        rotation=None,
        n=2,
    )
    cam = Camera(
        camera_id=1,
        image_id=1,
        name="cam0.jpg",
        R=np.eye(3),
        t=np.zeros(3),
        fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        width=640, height=480,
    )
    cameras = {"cam0.jpg": cam}

    mask = np.zeros((480, 640), dtype=bool)
    mask[220:260, 300:340] = True
    masks = {"cam0.jpg": {1: mask}}

    # Run with occlusion culling (default)
    votes_culled, _, _, _ = build_vote_matrix(
        cloud, cameras, masks, enable_occlusion_culling=True, min_mask_area=10
    )
    assert votes_culled[0, 0] == 1, "Foreground point must receive object label"
    assert votes_culled[1, 0] == UNKNOWN_LABEL, "Occluded background point must be UNKNOWN"

    # Run without occlusion culling (naive baseline)
    votes_naive, _, _, _ = build_vote_matrix(
        cloud, cameras, masks, enable_occlusion_culling=False, min_mask_area=10
    )
    assert votes_naive[0, 0] == 1
    assert votes_naive[1, 0] == 1, "Naive projection falsely leaks label onto occluded background"
