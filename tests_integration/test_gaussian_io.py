import numpy as np
from plyfile import PlyData, PlyElement

from integration.gaussian_io import load_gaussian_cloud, find_latest_ply


def _write_synthetic_ply(path, n=5):
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(n, 3)).astype(np.float32)
    opacity = rng.normal(size=n).astype(np.float32)  # raw (pre-sigmoid), like real 3DGS output
    scale = np.log(rng.uniform(0.01, 0.1, size=(n, 3))).astype(np.float32)  # stored as log-scale
    rot = rng.normal(size=(n, 4)).astype(np.float32)
    f_dc = rng.normal(size=(n, 3)).astype(np.float32)

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("opacity", "f4"),
             ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
             ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]

    data = np.empty(n, dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["opacity"] = opacity
    data["scale_0"], data["scale_1"], data["scale_2"] = scale[:, 0], scale[:, 1], scale[:, 2]
    data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]

    el = PlyElement.describe(data, "vertex")
    PlyData([el]).write(str(path))
    return xyz, opacity, scale, rot


def test_load_gaussian_cloud_roundtrip(tmp_path):
    ply_path = tmp_path / "point_cloud.ply"
    xyz, opacity_raw, scale_raw, rot_raw = _write_synthetic_ply(ply_path, n=5)

    cloud = load_gaussian_cloud(str(ply_path))

    assert cloud.n == 5
    assert np.allclose(cloud.xyz, xyz, atol=1e-5)

    # opacity should have sigmoid applied -> in (0, 1)
    assert np.all(cloud.opacity > 0) and np.all(cloud.opacity < 1)
    expected_opacity = 1.0 / (1.0 + np.exp(-opacity_raw))
    assert np.allclose(cloud.opacity, expected_opacity, atol=1e-5)

    # scale should have exp applied
    assert np.allclose(cloud.scale, np.exp(scale_raw), atol=1e-4)

    # rotation should be unit-normalized
    norms = np.linalg.norm(cloud.rotation, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_filter_by_opacity(tmp_path):
    ply_path = tmp_path / "point_cloud.ply"
    _write_synthetic_ply(ply_path, n=5)
    cloud = load_gaussian_cloud(str(ply_path))
    mask = cloud.filter_by_opacity(min_opacity=0.5)
    assert mask.dtype == bool
    assert mask.shape == (5,)


def test_find_latest_ply(tmp_path):
    model_dir = tmp_path / "my_scene"
    (model_dir / "point_cloud" / "iteration_7000").mkdir(parents=True)
    (model_dir / "point_cloud" / "iteration_30000").mkdir(parents=True)
    (model_dir / "point_cloud" / "iteration_7000" / "point_cloud.ply").write_text("x")
    (model_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply").write_text("x")

    latest = find_latest_ply(str(model_dir))
    assert "iteration_30000" in latest
