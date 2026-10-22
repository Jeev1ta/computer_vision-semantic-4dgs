import numpy as np
import pytest

from integration.colmap_io import Camera, _find_sparse_dir, _intrinsics_to_fx_fy_cx_cy
from integration.gaussian_io import load_gaussian_cloud
from integration.plyfile_lite import PlyData as LitePlyData, PlyElement as LitePlyElement
from integration.ply_compat import PlyData, PlyElement


# ---------------------------------------------------------------------------
# real COLMAP text model round-trip (official format)
# ---------------------------------------------------------------------------

def test_colmap_text_model_roundtrip(tmp_path):
    from integration.colmap_io import load_cameras

    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)

    # cameras.txt in the OFFICIAL format
    (sparse / "cameras.txt").write_text(
        "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
        "1 PINHOLE 640 480 500.0 500.0 320.0 240.0\n"
    )
    # images.txt: identity rotation, two cameras offset along x
    (sparse / "images.txt").write_text(
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
        "1 1 0 0 0 -0.5 0 0 1 00015.jpg\n\n"
        "2 1 0 0 0 0.5 0 0 1 00020.jpg\n\n"
    )

    cameras = load_cameras(str(tmp_path))
    assert set(cameras) == {"00015.jpg", "00020.jpg"}
    c15 = cameras["00015.jpg"]
    assert (c15.width, c15.height) == (640, 480)
    assert c15.fx == 500.0 and c15.cx == 320.0
    assert np.allclose(c15.R, np.eye(3), atol=1e-9)
    assert np.allclose(c15.t, [-0.5, 0, 0])
    assert np.allclose(c15.camera_center(), [0.5, 0, 0])


def test_colmap_sparse_dir_discovery(tmp_path):
    (tmp_path / "sparse" / "0").mkdir(parents=True)
    (tmp_path / "sparse" / "0" / "cameras.txt").write_text("")
    assert _find_sparse_dir(tmp_path) == (tmp_path / "sparse" / "0")

    with pytest.raises(FileNotFoundError):
        _find_sparse_dir(tmp_path / "nowhere")


def test_camera_model_extraction():
    p = np.array([500.0, 320.0, 240.0])
    assert _intrinsics_to_fx_fy_cx_cy("SIMPLE_PINHOLE", p) == (500.0, 500.0, 320.0, 240.0)
    p = np.array([500.0, 480.0, 320.0, 240.0, 0.01])
    fx, fy, cx, cy = _intrinsics_to_fx_fy_cx_cy("RADIAL", p)
    assert (fx, fy, cx, cy) == (500.0, 480.0, 320.0, 240.0)


# ---------------------------------------------------------------------------
# plyfile_lite fallback (ascii + binary round-trip)
# ---------------------------------------------------------------------------

def _synthetic_vertex_data(n=7):
    rng = np.random.default_rng(3)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    data = np.empty(n, dtype=dtype)
    data["x"], data["y"], data["z"] = rng.normal(size=(3, n))
    data["red"], data["green"], data["blue"] = 200, 100, 50
    return data


def test_plyfile_lite_ascii_roundtrip(tmp_path):
    data = _synthetic_vertex_data()
    el = LitePlyElement.describe(data, "vertex")
    LitePlyData([el]).write(str(tmp_path / "cloud.ply"))

    back = LitePlyData.read(str(tmp_path / "cloud.ply"))
    v = back.elements[0]
    assert np.allclose(np.stack([v["x"], v["y"], v["z"]], axis=1),
                       np.stack([data["x"], data["y"], data["z"]], axis=1), atol=1e-5)


def test_plyfile_lite_reads_binary_and_real_plyfile_reads_lite_ascii(tmp_path):
    data = _synthetic_vertex_data()
    el = LitePlyElement.describe(data, "vertex")
    LitePlyData([el]).write(str(tmp_path / "cloud.ply"), binary=True)
    back = LitePlyData.read(str(tmp_path / "cloud.ply"))
    assert len(back.elements[0].data) == len(data)

    # the compat layer (real plyfile when installed) must read the same file
    back2 = PlyData.read(str(tmp_path / "cloud.ply"))
    assert len(back2.elements[0].data) == len(data)


def test_gaussian_io_reads_colored_ply(tmp_path):
    """A xyz+rgb cloud (what the dashboard persists) loads fine: opacity
    defaults to 1, scale/rotation to None."""
    data = _synthetic_vertex_data()
    PlyData([PlyElement.describe(data, "vertex")]).write(str(tmp_path / "point_cloud.ply"))
    cloud = load_gaussian_cloud(str(tmp_path / "point_cloud.ply"))
    assert cloud.n == len(data)
    assert np.all(cloud.opacity == 1.0)
    assert cloud.scale is None and cloud.rotation is None


# ---------------------------------------------------------------------------
# classical segmentation + honest 2.5D lift
# ---------------------------------------------------------------------------

def _two_object_image():
    img = np.full((240, 320, 3), (30, 30, 40), dtype=np.uint8)
    img[40:120, 40:140] = (200, 50, 50)    # red block, upper-left
    img[140:220, 180:300] = (50, 200, 50)  # green block, lower-right
    return img


def test_classical_segmenter_recovers_two_shapes():
    from integration.segmentation import segment_image
    segments = segment_image(_two_object_image(), backend="classical", max_objects=6)
    assert 2 <= len(segments) <= 6
    areas = sorted((s.area for s in segments), reverse=True)
    # the two big colored blocks dominate the segment list
    assert areas[0] > 0.03 * 240 * 320
    assert all(s.mask.dtype == bool for s in segments)
    assert all(s.backend == "classical" for s in segments)
    # segments are deterministic
    again = segment_image(_two_object_image(), backend="classical", max_objects=6)
    assert [s.area for s in again] == [s.area for s in segments]


def test_lift_single_image_is_honest_2p5d():
    from integration.segmentation import segment_image
    from integration.single_image import lift_single_image

    img = _two_object_image()
    segments = segment_image(img, backend="classical", max_objects=4)
    result = lift_single_image(img, segments)

    assert result["mode"] == "single_image_2.5d"
    assert result["depth_source"] == "plane_assumption"
    n = len(result["xyz"])
    assert n > 0
    assert result["xyz"].shape == (n, 3)
    assert result["labels"].min() >= 1
    assert (result["confidence"] > 0).all() and (result["confidence"] <= 1).all()
    # nearer objects (smaller depth) are the lower/larger ones; z is bounded
    assert result["xyz"][:, 2].min() >= 1.0 and result["xyz"][:, 2].max() <= 4.0


def test_lift_rejects_unknown_depth_mode():
    from integration.single_image import lift_single_image
    with pytest.raises(ValueError, match="plane_assumption"):
        lift_single_image(_two_object_image(), [], depth_mode="magic_3d")
