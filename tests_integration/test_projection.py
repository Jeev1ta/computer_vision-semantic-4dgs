import numpy as np

try:
    import pytest
except ImportError:
    class _PytestFallback:
        def raises(self, exc):
            class _Context:
                def __enter__(self): return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    return exc_type is not None and issubclass(exc_type, exc)
            return _Context()
    pytest = _PytestFallback()


from integration.projection import (
    project_point,
    project_points_batch,
    inside_mask,
    sample_mask_batch,
    world_to_camera,
)


def test_project_point_center():
    # A point straight ahead of the camera should land near the principal point.
    p = project_point(X=[0.0, 0.0, 2.0], fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    assert np.allclose(p, [320.0, 240.0])


def test_project_point_offset():
    # Known offset: moving 1 unit right at depth 2 with fx=500 -> u shifts by 250 px.
    p = project_point(X=[1.0, 0.0, 2.0], fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    assert np.allclose(p, [320.0 + 250.0, 240.0])


def test_project_point_behind_camera_raises():
    with pytest.raises(ValueError):
        project_point(X=[0.0, 0.0, -1.0], fx=500.0, fy=500.0, cx=320.0, cy=240.0)



def test_world_to_camera_identity():
    R = np.eye(3)
    t = np.zeros(3)
    xyz = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 5.0]])
    out = world_to_camera(xyz, R, t)
    assert np.allclose(out, xyz)


def test_project_points_batch_matches_scalar():
    fx = fy = 500.0
    cx, cy = 320.0, 240.0
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.0])
    xyz = np.array([
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [-1.0, -1.0, -1.0],  # behind camera -> should be marked not visible
    ])
    uv, visible = project_points_batch(xyz, R, t, fx, fy, cx, cy, width=640, height=480)

    assert visible.tolist() == [True, True, False]
    assert np.allclose(uv[0], [320.0, 240.0])
    assert np.allclose(uv[1], [320.0 + 250.0, 240.0])


def test_project_points_batch_out_of_bounds():
    fx = fy = 500.0
    cx, cy = 320.0, 240.0
    R = np.eye(3)
    t = np.zeros(3)
    # A point far to the side will project outside a small image.
    xyz = np.array([[100.0, 0.0, 1.0]])
    uv, visible = project_points_batch(xyz, R, t, fx, fy, cx, cy, width=640, height=480)
    assert visible[0] == False  # noqa: E712


def test_inside_mask_bounds():
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    assert inside_mask(mask, u=5, v=5) is True
    assert inside_mask(mask, u=4, v=5) is False
    assert inside_mask(mask, u=-1, v=5) is False
    assert inside_mask(mask, u=5, v=100) is False


def test_sample_mask_batch_matches_scalar():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:8, 2:8] = True  # a small square object

    uv = np.array([
        [5.0, 5.0],   # inside
        [1.0, 1.0],   # outside
        [50.0, 50.0], # way out of range
    ])
    visible = np.array([True, True, True])

    hit = sample_mask_batch(mask, uv, visible)
    expected = [inside_mask(mask, u, v) for (u, v) in uv]
    assert hit.tolist() == expected


def test_sample_mask_batch_respects_visible_flag():
    mask = np.ones((10, 10), dtype=bool)
    uv = np.array([[5.0, 5.0], [5.0, 5.0]])
    visible = np.array([True, False])
    hit = sample_mask_batch(mask, uv, visible)
    assert hit.tolist() == [True, False]
