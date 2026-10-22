import numpy as np

from integration.colmap_io import Camera
from integration.gaussian_io import GaussianCloud
from integration.association import assign_gaussian_labels, build_vote_matrix
from integration.voting import UNKNOWN_LABEL, single_view_labels


def _make_camera(name, tx=0.0):
    """A camera looking down +Z, offset along X by tx (world == camera frame otherwise)."""
    return Camera(
        camera_id=1, image_id=1, name=name,
        width=100, height=100,
        fx=100.0, fy=100.0, cx=50.0, cy=50.0,
        R=np.eye(3), t=np.array([-tx, 0.0, 0.0]),
    )


def _make_cloud():
    # Three Gaussians: one on the "left" (chair), one on the "right" (table), one far background.
    xyz = np.array([
        [-0.5, 0.0, 2.0],   # projects left-of-center
        [0.5, 0.0, 2.0],    # projects right-of-center
        [0.0, 0.0, 2.0],    # dead center -- ambiguous / background
    ])
    opacity = np.array([0.9, 0.9, 0.9])
    return GaussianCloud(xyz=xyz, opacity=opacity, scale=None, rotation=None, n=3)


def _make_masks():
    # In each 100x100 image: left half = object 0 ("chair"), right half = object 1 ("table").
    chair_mask = np.zeros((100, 100), dtype=bool)
    chair_mask[:, :50] = True
    table_mask = np.zeros((100, 100), dtype=bool)
    table_mask[:, 50:] = True

    masks = {
        "cam0.jpg": {0: chair_mask.copy(), 1: table_mask.copy()},
        "cam1.jpg": {0: chair_mask.copy(), 1: table_mask.copy()},
        "cam2.jpg": {0: chair_mask.copy(), 1: table_mask.copy()},
    }
    return masks


def test_build_vote_matrix_shapes():
    cloud = _make_cloud()
    cameras = {f"cam{i}.jpg": _make_camera(f"cam{i}.jpg") for i in range(3)}
    masks = _make_masks()

    votes, weights, used_views, stats = build_vote_matrix(cloud, cameras, masks)
    assert votes.shape == (3, 3)
    assert weights.shape == (3, 3)
    assert set(used_views) == {"cam0.jpg", "cam1.jpg", "cam2.jpg"}
    assert stats["views"]["cam0.jpg"]["masks_used"] == 2
    assert votes[0, 0] == 0 and votes[1, 0] == 1  # left -> chair, right -> table


def test_build_vote_matrix_overlap_priority():
    """When a pixel is inside several masks the vote goes to the most
    credible mask (higher SAM2 score) regardless of dict iteration order."""
    cloud = _make_cloud()
    cam = _make_camera("cam0.jpg")
    cameras = {cam.name: cam}
    everything = np.ones((100, 100), dtype=bool)
    left = np.zeros((100, 100), dtype=bool); left[:, :50] = True

    # object 9 has the best score and comes LAST in dict order -> it must win
    masks = {"cam0.jpg": {0: left, 9: everything}}
    scores = {"cam0.jpg": {0: 0.9, 9: 0.99}}
    votes, _, _, _ = build_vote_matrix(cloud, cameras, masks, scores)
    assert (votes[:, 0] == 9).all()

    # with equal scores, the LARGER mask wins; with equal area, lowest id wins
    masks = {"cam0.jpg": {7: left, 3: everything}}
    votes, _, _, _ = build_vote_matrix(cloud, cameras, masks)
    assert (votes[:, 0] == 3).all()


def test_build_vote_matrix_requires_matching_camera():
    from pytest import raises
    cloud = _make_cloud()
    masks = {"missing.jpg": {0: np.ones((10, 10), dtype=bool)}}
    with raises(ValueError, match="matching camera"):
        build_vote_matrix(cloud, {}, masks)


def test_assign_gaussian_labels_majority():
    cloud = _make_cloud()
    cameras = {f"cam{i}.jpg": _make_camera(f"cam{i}.jpg") for i in range(3)}
    masks = _make_masks()

    labels, confidence, n_sup, used_views, stats = assign_gaussian_labels(
        cloud, cameras, masks, strategy="majority", min_views=1
    )

    # Gaussian 0 projects left -> chair (0) in every camera
    assert labels[0] == 0
    # Gaussian 1 projects right -> table (1) in every camera
    assert labels[1] == 1
    assert confidence[0] > 0.9
    assert confidence[1] > 0.9


def test_assign_gaussian_labels_weighted_runs():
    cloud = _make_cloud()
    cameras = {f"cam{i}.jpg": _make_camera(f"cam{i}.jpg") for i in range(3)}
    masks = _make_masks()

    labels, confidence, n_sup, used_views, stats = assign_gaussian_labels(
        cloud, cameras, masks, strategy="weighted", min_views=1
    )
    assert labels[0] == 0
    assert labels[1] == 1


def test_assign_gaussian_labels_single_view_noisier():
    cloud = _make_cloud()
    cameras = {f"cam{i}.jpg": _make_camera(f"cam{i}.jpg") for i in range(3)}
    masks = _make_masks()

    labels, confidence, n_sup, used_views, stats = assign_gaussian_labels(
        cloud, cameras, masks, strategy="single_view"
    )
    # single view still gets an answer here since all views agree in this toy scene,
    # but only 1 supporting view is ever counted.
    assert (n_sup <= 1).all()
    assert stats["single_view_chosen"] in used_views


def test_single_view_picks_most_informative_view():
    """The single-view baseline should use the view with the most object
    votes, not whatever column happened to be first."""
    cloud = _make_cloud()
    cameras = {"empty.jpg": _make_camera("empty.jpg"), "full.jpg": _make_camera("full.jpg")}
    # empty view: no masks at all; full view: full coverage
    masks = {"empty.jpg": {}, "full.jpg": {0: np.ones((100, 100), dtype=bool)}}
    votes, _, used_views, _ = build_vote_matrix(cloud, cameras, masks)
    labels, confidence, n_sup, chosen = single_view_labels(votes)
    assert used_views[chosen] == "full.jpg"
    assert (labels == 0).all()
